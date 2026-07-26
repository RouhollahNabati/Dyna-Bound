package org.fog.dynacolgnn;

import org.fog.application.Application;
import org.fog.dynacol.DynaColConfig;
import org.fog.dynacol.DynaColFeatureFlags;
import org.fog.dynacol.metrics.ControlOverheadMonitor;
import org.fog.dynacol.model.ColonyResourceEntry;
import org.fog.dynacol.model.ColonySummary;
import org.fog.dynacol.model.FogNodeState;
import org.fog.dynacol.model.ResourceVector;
import org.fog.dynacol.model.ServiceRequest;
import org.fog.dynacol.placement.AttractivenessModel;
import org.fog.dynacol.placement.DCBOPlacement;
import org.fog.dynacol.placement.ObjectiveFunction;
import org.fog.dynacol.placement.PlacementCandidateUtil;
import org.fog.dynacol.placement.PlacementDepth;
import org.fog.dynacol.placement.ReconcilablePlacement;
import org.fog.dynacol.table.ColonyResourceTable;
import org.fog.dynacol.table.GlobalResourceTable;
import org.fog.dynacol.util.FogTopologyUtil;
import org.fog.dynacolgnn.graph.ServiceDagFeatures;
import org.fog.dynacolgnn.learn.ColonyFeatureEncoder;
import org.fog.dynacolgnn.learn.OnlineColonyGnn;
import org.fog.dynacolgnn.learn.OnlineLinearPolicy;
import org.fog.dynacolgnn.learn.TrajectoryLogger;
import org.fog.entities.FogDevice;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

/**
 * Colony-bounded L1/L2/L3 placement for DynaCol-Hybrid.
 * Vector ablation uses {@link OnlineLinearPolicy}; GNN uses {@link OnlineColonyGnn}.
 * When learn-blend is ~0 (large-N default), delegates to {@link DCBOPlacement} for parity.
 */
public final class ColonyBoundedPlacement implements ReconcilablePlacement {

    private static final double LEARNING_RATE = 0.05;
    private static final double GNN_LEARNING_RATE = 0.055;
    private static final double EPSILON_SMALL = 0.05;
    private static final double EPSILON_LARGE = 0.0;
    /** Shared scale-aware blend so A1/A2 differ by encoder, not by blend schedule. */
    private static final double LEARN_BLEND_SMALL = 0.40;
    private static final double LEARN_BLEND_MID = 0.25;
    private static final double LEARN_BLEND_LARGE = 0.0;
    /** Stronger learner voice for GNN when graph-SLA channels are active. */
    private static final double LEARN_BLEND_GNN_SMALL = 0.60;
    /**
     * iFogSim grids use target N but actual fog count is slightly lower
     * (e.g. target 500 → ~497). Treat ≥450 as large scale.
     */
    private static final int LARGE_SCALE_FOG_NODES = 450;
    private static final int MID_SCALE_FOG_NODES = 250;

    private final Map<Integer, FogNodeState> nodeStates;
    private final Map<Integer, FogDevice> deviceIndex;
    private final List<FogDevice> allDevices;
    private final ObjectiveFunction objectiveFunction;
    private final AttractivenessModel attractivenessModel;
    private final DCBOPlacement dcboDelegate;
    private final DynaColFeatureFlags flags;
    private final EncoderKind encoderKind;
    private final OnlineLinearPolicy vectorPolicy;
    private final OnlineColonyGnn gnnPolicy;
    private final ResourceVector maxResources;
    private final Map<String, Integer> moduleHosts = new HashMap<>();
    private final int fogNodeCount;

    private Application application;
    private ServiceDagFeatures dagFeatures = ServiceDagFeatures.from(null);

    public ColonyBoundedPlacement(Map<Integer, FogNodeState> nodeStates,
                                  List<FogDevice> allDevices,
                                  EncoderKind encoderKind,
                                  Random random,
                                  DynaColFeatureFlags flags) {
        this.nodeStates = nodeStates;
        this.allDevices = allDevices;
        this.deviceIndex = FogTopologyUtil.indexById(allDevices);
        this.objectiveFunction = new ObjectiveFunction(allDevices);
        this.flags = flags != null ? flags : DynaColFeatureFlags.full();
        this.encoderKind = encoderKind != null ? encoderKind : EncoderKind.GNN;
        this.attractivenessModel = new AttractivenessModel(this.flags);
        this.dcboDelegate = new DCBOPlacement(nodeStates, allDevices, this.flags);
        this.fogNodeCount = (int) allDevices.stream().filter(d -> d.getParentId() != -1).count();
        double epsilon = fogNodeCount >= LARGE_SCALE_FOG_NODES ? EPSILON_LARGE : EPSILON_SMALL;
        String epsProp = System.getProperty("dynacolgnn.epsilon");
        if (epsProp != null && !epsProp.isBlank()) {
            try {
                epsilon = Math.max(0.0, Math.min(1.0, Double.parseDouble(epsProp.trim())));
            } catch (NumberFormatException ignored) {
                // keep default
            }
        }
        String collectEps = System.getProperty("dynacolgnn.collectEpsilon");
        if (collectEps != null && !collectEps.isBlank()) {
            try {
                epsilon = Math.max(epsilon, Double.parseDouble(collectEps.trim()));
            } catch (NumberFormatException ignored) {
                // keep default
            }
        }
        if (this.encoderKind == EncoderKind.GNN) {
            this.gnnPolicy = new OnlineColonyGnn(GNN_LEARNING_RATE, epsilon, random);
            this.vectorPolicy = null;
            TrajectoryLogger.getInstance().ensureOpen();
        } else {
            this.vectorPolicy = new OnlineLinearPolicy(
                    ColonyFeatureEncoder.FEATURE_DIM, LEARNING_RATE, epsilon, random);
            this.gnnPolicy = null;
        }
        this.maxResources = FogTopologyUtil.maxResources(allDevices);
    }

    public void setApplication(Application application) {
        this.application = application;
        this.dagFeatures = ServiceDagFeatures.from(application);
    }

    /**
     * Seed / refresh known module→host mapping from the runtime area table so
     * producer-RTT and co-location channels are populated on the first L2 step
     * (not only after this placer itself has updated hosts).
     */
    public void syncModuleHosts(Map<String, Integer> hosts) {
        moduleHosts.clear();
        if (hosts == null || hosts.isEmpty()) {
            return;
        }
        moduleHosts.putAll(hosts);
    }

    public EncoderKind getEncoderKind() {
        return encoderKind;
    }

    @Override
    public Optional<Integer> placeService(ServiceRequest request, FogNodeState localFcmState) {
        return reconcileService(request, localFcmState, null, PlacementDepth.L3_GLOBAL);
    }

    @Override
    public Optional<Integer> reconcileService(ServiceRequest request,
                                              FogNodeState localFcmState,
                                              Integer currentHostId,
                                              PlacementDepth maxDepth) {
        if (localFcmState == null) {
            return Optional.empty();
        }
        // Exact DCBO parity when learner is disabled by blend (large-N default).
        if (resolveLearnBlend() <= 1e-12) {
            return dcboDelegate.reconcileService(request, localFcmState, currentHostId, maxDepth);
        }

        if (maxDepth.ordinal() >= PlacementDepth.L1_FAST.ordinal() && currentHostId != null) {
            Optional<Integer> sticky = tryFastPath(request, localFcmState, currentHostId);
            if (sticky.isPresent()) {
                moduleHosts.put(request.getModuleName(), sticky.get());
                return sticky;
            }
        }

        if (maxDepth.ordinal() >= PlacementDepth.L2_COLONY.ordinal()) {
            ControlOverheadMonitor.getInstance().recordQuery();
            if (flags.isLearningEnabled()) {
                attractivenessModel.applyTemporalDecay(localFcmState.getCrt(), localFcmState.getGrt());
            }
            if (flags.isCrtEnabled()) {
                Optional<Integer> local = placeLocalLearned(request, localFcmState.getCrt());
                if (local.isPresent()) {
                    return local;
                }
            }
            if (flags.isGrtEnabled()) {
                Optional<Integer> remote = placeRemote(request, localFcmState.getGrt());
                if (remote.isPresent()) {
                    return remote;
                }
            }
            if (currentHostId != null && hostStillFeasible(request, localFcmState, currentHostId)) {
                return Optional.of(currentHostId);
            }
        }

        if (maxDepth == PlacementDepth.L3_GLOBAL) {
            return placeCloud();
        }
        return currentHostId != null ? Optional.of(currentHostId) : Optional.empty();
    }

    private Optional<Integer> tryFastPath(ServiceRequest request,
                                          FogNodeState localFcmState,
                                          int currentHostId) {
        if (!hostStillFeasible(request, localFcmState, currentHostId)) {
            return Optional.empty();
        }
        double latency = estimateLatencyMs(request, currentHostId);
        double deadline = request.getDeadlineMs();
        if (latency <= deadline * DynaColConfig.SLA_STICKY_LOW_RATIO) {
            return Optional.of(currentHostId);
        }
        if (latency > deadline * DynaColConfig.SLA_ESCALATE_HIGH_RATIO) {
            return Optional.empty();
        }
        ColonyResourceTable crt = localFcmState.getCrt();
        if (crt == null) {
            return Optional.of(currentHostId);
        }
        double currentScore = scoreEntry(request, crt.get(currentHostId));
        List<ColonyResourceEntry> feasible = feasibleTopK(request, crt);
        if (feasible.isEmpty()) {
            return Optional.of(currentHostId);
        }
        double bestScore = scoreEntry(request, feasible.get(0));
        if (bestScore == Double.MAX_VALUE
                || currentScore <= bestScore * (1.0 + DynaColConfig.STICKY_SCORE_MARGIN)) {
            return Optional.of(currentHostId);
        }
        return Optional.empty();
    }

    private Optional<Integer> placeLocalLearned(ServiceRequest request, ColonyResourceTable crt) {
        List<ColonyResourceEntry> feasible = feasibleTopK(request, crt);
        if (feasible.isEmpty()) {
            return Optional.empty();
        }
        double[] coloc = colocAffinity(request.getModuleName(), feasible);
        // A1 ablation: no service-graph co-location channel (flat host+service only).
        if (encoderKind != EncoderKind.GNN) {
            coloc = new double[feasible.size()];
        }
        double[][] features = ColonyFeatureEncoder.encodeCandidates(
                encoderKind, request, feasible, dagFeatures, maxResources, deviceIndex, coloc, moduleHosts);
        double[] base = buildBaseUtilities(request, feasible, features, coloc);
        int idx = selectIndex(feasible, features, base);
        if (idx < 0 || idx >= feasible.size()) {
            return Optional.empty();
        }
        ColonyResourceEntry chosen = feasible.get(idx);
        ControlOverheadMonitor.getInstance().recordPlacement();
        LearningSignal signal;
        if (encoderKind == EncoderKind.GNN) {
            signal = shapeLearningReward(request, feasible, idx, features, coloc);
        } else {
            // Vector ablation: classical 1-step relative attractiveness only.
            double reward = attractivenessModel.reward(
                    request, deviceIndex.get(chosen.getFogDeviceId()), deviceIndex);
            double dcboReward = attractivenessModel.reward(
                    request, deviceIndex.get(feasible.get(0).getFogDeviceId()), deviceIndex);
            double shaped = Math.max(0.0, Math.min(1.0, 0.5 + 0.5 * (reward - dcboReward)));
            signal = new LearningSignal(shaped, -1);
        }
        double rawReward = attractivenessModel.reward(
                request, deviceIndex.get(chosen.getFogDeviceId()), deviceIndex);
        if (gnnPolicy != null) {
            gnnPolicy.update(idx, signal.shaped, signal.bestIndex);
            if (TrajectoryLogger.getInstance().isEnabled()) {
                TrajectoryLogger.getInstance().logDecision(
                        gnnPolicy.lastX(),
                        gnnPolicy.lastCtx(),
                        gnnPolicy.lastAdj(),
                        base,
                        resolveLearnBlend(),
                        idx,
                        signal.shaped,
                        request.getModuleName());
            }
        } else if (vectorPolicy != null) {
            vectorPolicy.update(features[idx], signal.shaped);
        }
        attractivenessModel.updateColonyEntry(chosen, request, rawReward);
        chosen.setAvailable(chosen.getAvailable().subtract(request.getDemand()));
        moduleHosts.put(request.getModuleName(), chosen.getFogDeviceId());
        return Optional.of(chosen.getFogDeviceId());
    }

    private static final class LearningSignal {
        final double shaped;
        final int bestIndex;

        LearningSignal(double shaped, int bestIndex) {
            this.shaped = shaped;
            this.bestIndex = bestIndex;
        }
    }

    /**
     * Multi-candidate learning target. Vector: attr+util only.
     * GNN: graph-SLA score (producer-RTT pull, coloc, fan) dominates so the
     * signal cannot be copied by the flat ablation.
     */
    private LearningSignal shapeLearningReward(ServiceRequest request,
                                               List<ColonyResourceEntry> feasible,
                                               int chosenIdx,
                                               double[][] features,
                                               double[] coloc) {
        int n = feasible.size();
        if (n <= 1 || chosenIdx < 0 || chosenIdx >= n) {
            return new LearningSignal(0.5, chosenIdx);
        }
        double[] scores = new double[n];
        double sum = 0.0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        int best = 0;
        for (int i = 0; i < n; i++) {
            FogDevice device = deviceIndex.get(feasible.get(i).getFogDeviceId());
            double attr = attractivenessModel.reward(request, device, deviceIndex);
            double j = scoreEntry(request, feasible.get(i));
            double util = (j == Double.MAX_VALUE) ? 0.0 : 1.0 / (1.0 + j);
            double colocI = (coloc != null && i < coloc.length) ? coloc[i] : 0.0;
            double pull = 0.0;
            double fan = 0.0;
            if (features != null && i < features.length
                    && features[i].length > ColonyFeatureEncoder.HOST_DIM + 6) {
                pull = features[i][ColonyFeatureEncoder.HOST_DIM + 5];
                fan = features[i][ColonyFeatureEncoder.HOST_DIM + 6];
            }
            if (encoderKind == EncoderKind.GNN) {
                // Graph-SLA channels Vector cannot see; keep attr/util so SLA stays aligned.
                double graphSla = 0.45 * pull + 0.35 * colocI + 0.20 * (fan * Math.max(pull, colocI));
                scores[i] = 0.35 * attr + 0.25 * util + 0.40 * graphSla;
            } else {
                scores[i] = 0.60 * attr + 0.40 * util;
            }
            sum += scores[i];
            min = Math.min(min, scores[i]);
            max = Math.max(max, scores[i]);
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        double mean = sum / n;
        double range = Math.max(1e-4, max - min);
        double z = (scores[chosenIdx] - mean) / range;

        int better = 0;
        for (int i = 0; i < n; i++) {
            if (scores[i] > scores[chosenIdx] + 1e-12) {
                better++;
            }
        }
        double rankFrac = 1.0 - (better / (double) Math.max(1, n - 1));
        double shaped = 0.5 + 0.40 * z + 0.30 * (rankFrac - 0.5);
        shaped = Math.max(0.0, Math.min(1.0, shaped));
        return new LearningSignal(shaped, best);
    }

    private List<ColonyResourceEntry> feasibleTopK(ServiceRequest request, ColonyResourceTable crt) {
        List<ColonyResourceEntry> feasible = new ArrayList<>();
        if (crt == null) {
            return feasible;
        }
        for (ColonyResourceEntry entry : PlacementCandidateUtil.fogCandidates(
                crt.asList(), deviceIndex, request)) {
            if (entry.getAvailable().canFit(request.getDemand())) {
                feasible.add(entry);
            }
        }
        feasible.sort(Comparator.comparingDouble(e -> scoreEntry(request, e)));
        if (feasible.size() > DynaColConfig.TOP_K_CRT_CANDIDATES) {
            return new ArrayList<>(feasible.subList(0, DynaColConfig.TOP_K_CRT_CANDIDATES));
        }
        return feasible;
    }

    private double scoreEntry(ServiceRequest request, ColonyResourceEntry entry) {
        if (entry == null || !entry.getAvailable().canFit(request.getDemand())) {
            return Double.MAX_VALUE;
        }
        double score = objectiveFunction.score(request, entry);
        if (flags.isLearningEnabled()) {
            score -= DynaColConfig.LAMBDA_ATTRACTIVENESS * entry.getAttractiveness();
        }
        return score;
    }

    private double[] colocAffinity(String moduleName, List<ColonyResourceEntry> feasible) {
        double[] out = new double[feasible.size()];
        // Prefer directed producers (co-locate with upstream); fall back to undirected.
        Set<String> related = dagFeatures.producers(moduleName);
        if (related.isEmpty()) {
            related = dagFeatures.neighbors(moduleName);
        }
        if (related.isEmpty() || moduleHosts.isEmpty()) {
            return out;
        }
        for (int i = 0; i < feasible.size(); i++) {
            int hostId = feasible.get(i).getFogDeviceId();
            int hits = 0;
            int total = 0;
            for (String neigh : related) {
                Integer placed = moduleHosts.get(neigh);
                if (placed == null) {
                    continue;
                }
                total++;
                if (placed == hostId) {
                    hits++;
                }
            }
            out[i] = total == 0 ? 0.0 : hits / (double) total;
        }
        return out;
    }

    private double[] buildBaseUtilities(ServiceRequest request,
                                        List<ColonyResourceEntry> feasible,
                                        double[][] features,
                                        double[] coloc) {
        double[] base = new double[feasible.size()];
        for (int i = 0; i < feasible.size(); i++) {
            double j = scoreEntry(request, feasible.get(i));
            base[i] = (j == Double.MAX_VALUE) ? 0.0 : 1.0 / (1.0 + j);
            if (encoderKind == EncoderKind.GNN) {
                double pull = features[i][ColonyFeatureEncoder.HOST_DIM + 5];
                double fan = features[i][ColonyFeatureEncoder.HOST_DIM + 6];
                base[i] += 0.40 * coloc[i] + 0.45 * pull + 0.15 * fan;
            }
        }
        return base;
    }

    private int selectIndex(List<ColonyResourceEntry> feasible,
                            double[][] features,
                            double[] base) {
        double blend = resolveLearnBlend();
        if (gnnPolicy != null) {
            return gnnPolicy.selectHybridIndex(feasible, deviceIndex, features, base, blend);
        }
        return vectorPolicy.selectHybridIndex(features, base, blend);
    }

    private double resolveLearnBlend() {
        String prop = System.getProperty("dynacolgnn.learnBlend");
        if (prop != null && !prop.isBlank()) {
            try {
                return Double.parseDouble(prop.trim());
            } catch (NumberFormatException ignored) {
                // fall through
            }
        }
        if (fogNodeCount >= LARGE_SCALE_FOG_NODES) {
            return LEARN_BLEND_LARGE;
        }
        if (fogNodeCount >= MID_SCALE_FOG_NODES) {
            return LEARN_BLEND_MID;
        }
        if (encoderKind == EncoderKind.GNN) {
            return LEARN_BLEND_GNN_SMALL;
        }
        return LEARN_BLEND_SMALL;
    }

    private Optional<Integer> placeRemote(ServiceRequest request, GlobalResourceTable grt) {
        if (grt == null) {
            return Optional.empty();
        }
        List<ColonySummary> ranked = new ArrayList<>();
        for (ColonySummary summary : grt.asList()) {
            if (!summary.getAggregateAvailable().canFit(request.getDemand())) {
                continue;
            }
            ranked.add(summary);
        }
        ranked.sort(Comparator.comparingDouble(s -> {
            double score = objectiveFunction.score(request, s);
            if (flags.isLearningEnabled()) {
                score -= DynaColConfig.LAMBDA_COLONY * s.getAttractiveness();
            }
            return score;
        }));
        int limit = Math.min(DynaColConfig.TOP_K_GRT_COLONIES, ranked.size());
        for (int i = 0; i < limit; i++) {
            ColonySummary summary = ranked.get(i);
            FogNodeState remoteFcm = nodeStates.get(summary.getFcmDeviceId());
            if (remoteFcm == null || remoteFcm.getCrt() == null) {
                continue;
            }
            Optional<Integer> delegated = placeLocalLearned(request, remoteFcm.getCrt());
            if (delegated.isPresent()) {
                if (flags.isLearningEnabled()) {
                    double reward = attractivenessModel.reward(
                            request, deviceIndex.get(delegated.get()), deviceIndex);
                    attractivenessModel.updateColonySummary(summary, request, reward);
                }
                return delegated;
            }
        }
        return Optional.empty();
    }

    private Optional<Integer> placeCloud() {
        FogDevice cloud = allDevices.stream()
                .filter(d -> d.getParentId() == -1)
                .findFirst()
                .orElse(null);
        if (cloud == null) {
            return Optional.empty();
        }
        ControlOverheadMonitor.getInstance().recordPlacement();
        return Optional.of(cloud.getId());
    }

    private boolean hostStillFeasible(ServiceRequest request, FogNodeState localFcmState, int hostId) {
        if (localFcmState.getCrt() == null) {
            return false;
        }
        ColonyResourceEntry entry = localFcmState.getCrt().get(hostId);
        return entry != null && entry.getAvailable().canFit(request.getDemand());
    }

    private double estimateLatencyMs(ServiceRequest request, int hostId) {
        FogDevice source = deviceIndex.get(request.getSourceDeviceId());
        FogDevice target = deviceIndex.get(hostId);
        if (source == null || target == null) {
            return Double.MAX_VALUE;
        }
        return FogTopologyUtil.estimateRttMs(source, target, deviceIndex);
    }
}
