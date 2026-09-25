package org.fog.dynacolgnn.learn;

import org.fog.dynacol.model.ColonyResourceEntry;
import org.fog.dynacol.model.ResourceVector;
import org.fog.dynacol.model.ServiceRequest;
import org.fog.dynacol.util.FogTopologyUtil;
import org.fog.dynacolgnn.EncoderKind;
import org.fog.dynacolgnn.graph.ServiceDagFeatures;
import org.fog.entities.FogDevice;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Builds per-candidate features.
 * <ul>
 *   <li>VECTOR (A1): local host features + service/DAG scalars (no message passing)</li>
 *   <li>GNN (A2): structure-weighted CRT hops (RTT + coloc + producer pull)
 *       + service-graph message from placed producers</li>
 * </ul>
 */
public final class ColonyFeatureEncoder {

    public static final int HOST_DIM = 8;
    public static final int SERVICE_DIM = 8;
    /** host_emb + service + bias */
    public static final int FEATURE_DIM = HOST_DIM + SERVICE_DIM + 1;
    public static final int GNN_HOPS = 2;
    /** Pairwise coloc co-support added to CRT edges (GNN-v2). */
    private static final double LAMBDA_COLOC_PAIR = 0.35;
    /** Pull message mass toward producer-aligned (high-coloc) hosts. */
    private static final double LAMBDA_COLOC_TARGET = 0.50;

    private ColonyFeatureEncoder() {
    }

    public static double[][] encodeCandidates(EncoderKind kind,
                                             ServiceRequest request,
                                             List<ColonyResourceEntry> candidates,
                                             ServiceDagFeatures dag,
                                             ResourceVector maxResources,
                                             Map<Integer, FogDevice> deviceIndex,
                                             double[] colocAffinity,
                                             Map<String, Integer> moduleHosts) {
        int n = candidates.size();
        double[] service = serviceFeatures(request, dag, maxResources);
        // Vector ablation: demand/deadline only — no DAG topology scalars.
        // GNN keeps full service-graph channels (degrees, depth, pull, fan, coloc).
        if (kind != EncoderKind.GNN) {
            service = new double[]{
                    service[0], service[1], service[2],
                    0.0, 0.0, 0.0, 0.0, 0.0
            };
        }
        double[][] hostRaw = new double[n][];
        for (int i = 0; i < n; i++) {
            hostRaw[i] = hostLocalFeatures(candidates.get(i), maxResources);
        }

        double[][] hostEmb;
        if (kind == EncoderKind.GNN && n > 1) {
            double[][] weights = structureAffinityWeights(candidates, deviceIndex, colocAffinity);
            hostEmb = weightedAggregateHops(hostRaw, weights, GNN_HOPS);
            for (int i = 0; i < n; i++) {
                hostEmb[i][6] = neighborPressure(hostRaw, weights, i, 0);
                hostEmb[i][7] = weightEntropy(weights[i]);
            }
            // Service-graph → host: pull features from hosts of already-placed producers.
            injectProducerHostMessages(hostEmb, candidates, request, dag, moduleHosts, deviceIndex);
        } else {
            hostEmb = hostRaw;
            for (int i = 0; i < n; i++) {
                hostEmb[i][6] = 0.0;
                hostEmb[i][7] = 0.0;
            }
        }

        double[][] out = new double[n][FEATURE_DIM];
        for (int i = 0; i < n; i++) {
            System.arraycopy(hostEmb[i], 0, out[i], 0, HOST_DIM);
            System.arraycopy(service, 0, out[i], HOST_DIM, SERVICE_DIM);
            double coloc = (colocAffinity != null && i < colocAffinity.length) ? colocAffinity[i] : 0.0;
            if (kind == EncoderKind.GNN) {
                double producerPull = producerRttAffinity(
                        candidates.get(i).getFogDeviceId(), request, dag, moduleHosts, deviceIndex);
                double fanJoin = dag != null ? dag.fanInNorm(request.getModuleName()) : 0.0;
                // GNN-only service slots: producer RTT pull and fan-in join pressure.
                out[i][HOST_DIM + 5] = producerPull;
                out[i][HOST_DIM + 6] = fanJoin;
            }
            out[i][HOST_DIM + 7] = coloc;
            out[i][FEATURE_DIM - 1] = 1.0;
        }
        return out;
    }

    private static double[] hostLocalFeatures(ColonyResourceEntry entry, ResourceVector maxResources) {
        ResourceVector avail = entry.getAvailable();
        double maxCpu = Math.max(1.0, maxResources.getCpu());
        double maxRam = Math.max(1.0, maxResources.getRam());
        double maxSto = Math.max(1.0, maxResources.getStorage());
        double maxBw = Math.max(1.0, maxResources.getBandwidth());
        return new double[]{
                avail.getCpu() / maxCpu,
                avail.getRam() / maxRam,
                avail.getStorage() / maxSto,
                avail.getBandwidth() / maxBw,
                Math.min(1.0, entry.getRttToFcmMs() / 100.0),
                Math.max(-1.0, Math.min(1.0, entry.getAttractiveness())),
                0.0,
                0.0
        };
    }

    private static double[] serviceFeatures(ServiceRequest request,
                                           ServiceDagFeatures dag,
                                           ResourceVector maxResources) {
        ResourceVector demand = request.getDemand();
        double maxCpu = Math.max(1.0, maxResources.getCpu());
        double maxRam = Math.max(1.0, maxResources.getRam());
        double deadline = Math.max(1.0, request.getDeadlineMs());
        String module = request.getModuleName();
        double inDeg = dag != null ? dag.inDegreeNorm(module) : 0.0;
        double outDeg = dag != null ? dag.outDegreeNorm(module) : 0.0;
        double depth = dag != null ? dag.depthNorm(module) : 0.0;
        double source = dag != null ? dag.isSource(module) : 0.0;
        double sink = dag != null ? dag.isSink(module) : 0.0;
        return new double[]{
                demand.getCpu() / maxCpu,
                demand.getRam() / maxRam,
                Math.min(1.0, deadline / 500.0),
                inDeg,
                outDeg,
                depth,
                Math.max(source, sink),
                0.0 // coloc filled per candidate
        };
    }

    /**
     * Mix candidate host embedding with mean features of hosts that already
     * run directed producers — A1 never sees this cross-graph channel.
     */
    private static void injectProducerHostMessages(double[][] hostEmb,
                                                   List<ColonyResourceEntry> candidates,
                                                   ServiceRequest request,
                                                   ServiceDagFeatures dag,
                                                   Map<String, Integer> moduleHosts,
                                                   Map<Integer, FogDevice> deviceIndex) {
        if (dag == null || moduleHosts == null || moduleHosts.isEmpty()) {
            return;
        }
        Set<String> producers = dag.producers(request.getModuleName());
        if (producers.isEmpty()) {
            return;
        }
        double[] msg = new double[6];
        int hits = 0;
        for (String producer : producers) {
            Integer hostId = moduleHosts.get(producer);
            if (hostId == null) {
                continue;
            }
            for (int j = 0; j < candidates.size(); j++) {
                if (candidates.get(j).getFogDeviceId() == hostId) {
                    for (int d = 0; d < 6; d++) {
                        msg[d] += hostEmb[j][d];
                    }
                    hits++;
                    break;
                }
            }
        }
        if (hits == 0) {
            return;
        }
        for (int d = 0; d < 6; d++) {
            msg[d] /= hits;
        }
        for (int i = 0; i < hostEmb.length; i++) {
            for (int d = 0; d < 6; d++) {
                hostEmb[i][d] = 0.5 * hostEmb[i][d] + 0.5 * msg[d];
            }
            // Boost structural channels with same-host producer fraction.
            double sameHost = 0.0;
            int total = 0;
            int candId = candidates.get(i).getFogDeviceId();
            for (String producer : producers) {
                Integer hostId = moduleHosts.get(producer);
                if (hostId == null) {
                    continue;
                }
                total++;
                if (hostId == candId) {
                    sameHost += 1.0;
                }
            }
            if (total > 0) {
                hostEmb[i][6] = 0.5 * hostEmb[i][6] + 0.5 * (sameHost / total);
            }
            hostEmb[i][7] = 0.5 * hostEmb[i][7]
                    + 0.5 * producerRttAffinity(candId, request, dag, moduleHosts, deviceIndex);
        }
    }

    private static double producerRttAffinity(int candidateHostId,
                                              ServiceRequest request,
                                              ServiceDagFeatures dag,
                                              Map<String, Integer> moduleHosts,
                                              Map<Integer, FogDevice> deviceIndex) {
        if (dag == null || moduleHosts == null || deviceIndex == null) {
            return 0.0;
        }
        Set<String> producers = dag.producers(request.getModuleName());
        if (producers.isEmpty()) {
            return 0.0;
        }
        FogDevice cand = deviceIndex.get(candidateHostId);
        if (cand == null) {
            return 0.0;
        }
        double sum = 0.0;
        int n = 0;
        for (String producer : producers) {
            Integer hostId = moduleHosts.get(producer);
            if (hostId == null) {
                continue;
            }
            FogDevice prodHost = deviceIndex.get(hostId);
            if (prodHost == null) {
                continue;
            }
            double rtt = FogTopologyUtil.estimateRttMs(cand, prodHost, deviceIndex);
            sum += 1.0 / (1.0 + rtt / 20.0);
            n++;
        }
        return n == 0 ? 0.0 : sum / n;
    }

    /**
     * Row-stochastic CRT affinity for GNN message passing.
     * Base = RTT affinity; with structure on (default), add coloc co-support and
     * pull toward producer-aligned hosts. Ablate with {@code -Ddynacolgnn.structureAdj=false}.
     */
    public static double[][] structureAffinityWeights(List<ColonyResourceEntry> candidates,
                                                      Map<Integer, FogDevice> deviceIndex,
                                                      double[] colocAffinity) {
        int n = candidates == null ? 0 : candidates.size();
        double[][] w = new double[n][n];
        if (n == 0) {
            return w;
        }
        boolean structureOn = structureAdjEnabled();
        for (int i = 0; i < n; i++) {
            FogDevice di = deviceIndex != null ? deviceIndex.get(candidates.get(i).getFogDeviceId()) : null;
            double colocI = colocAt(colocAffinity, i);
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    w[i][j] = 1.0;
                } else {
                    FogDevice dj = deviceIndex != null ? deviceIndex.get(candidates.get(j).getFogDeviceId()) : null;
                    double rtt;
                    if (di != null && dj != null && deviceIndex != null) {
                        rtt = FogTopologyUtil.estimateRttMs(di, dj, deviceIndex);
                    } else {
                        rtt = Math.abs(candidates.get(i).getRttToFcmMs() - candidates.get(j).getRttToFcmMs()) + 1.0;
                    }
                    double edge = 1.0 / (1.0 + rtt / 20.0);
                    if (structureOn) {
                        double colocJ = colocAt(colocAffinity, j);
                        edge += LAMBDA_COLOC_PAIR * Math.min(colocI, colocJ);
                        edge += LAMBDA_COLOC_TARGET * colocJ;
                    }
                    w[i][j] = edge;
                }
                sum += w[i][j];
            }
            if (sum > 0) {
                for (int j = 0; j < n; j++) {
                    w[i][j] /= sum;
                }
            }
        }
        return w;
    }

    private static boolean structureAdjEnabled() {
        String prop = System.getProperty("dynacolgnn.structureAdj", "true");
        return prop == null || !"false".equalsIgnoreCase(prop.trim());
    }

    private static double colocAt(double[] coloc, int i) {
        if (coloc == null || i < 0 || i >= coloc.length) {
            return 0.0;
        }
        double v = coloc[i];
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            return 0.0;
        }
        return Math.max(0.0, Math.min(1.0, v));
    }

    private static double[][] weightedAggregateHops(double[][] raw, double[][] weights, int hops) {
        double[][] cur = raw;
        for (int h = 0; h < hops; h++) {
            cur = oneWeightedHop(cur, weights);
        }
        return cur;
    }

    private static double[][] oneWeightedHop(double[][] raw, double[][] weights) {
        int n = raw.length;
        int dim = Math.min(6, raw[0].length);
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            out[i] = raw[i].clone();
            for (int d = 0; d < dim; d++) {
                double acc = 0.0;
                for (int j = 0; j < n; j++) {
                    acc += weights[i][j] * raw[j][d];
                }
                out[i][d] = 0.8 * raw[i][d] + 0.2 * acc;
            }
        }
        return out;
    }

    private static double neighborPressure(double[][] hostRaw, double[][] weights, int i, int channel) {
        double acc = 0.0;
        for (int j = 0; j < hostRaw.length; j++) {
            if (j == i) {
                continue;
            }
            acc += weights[i][j] * (1.0 - hostRaw[j][channel]);
        }
        return Math.max(0.0, Math.min(1.0, acc));
    }

    private static double weightEntropy(double[] row) {
        double h = 0.0;
        for (double p : row) {
            if (p > 1e-12) {
                h -= p * Math.log(p);
            }
        }
        return Math.min(1.0, h / Math.log(Math.max(2, row.length)));
    }
}
