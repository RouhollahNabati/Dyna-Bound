package org.fog.dynacolgnn.graph;

import org.fog.application.AppEdge;
import org.fog.application.AppModule;
import org.fog.application.Application;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

/**
 * Lightweight service-DAG statistics for the module being placed.
 */
public final class ServiceDagFeatures {

    private final Map<String, Integer> inDegree = new HashMap<>();
    private final Map<String, Integer> outDegree = new HashMap<>();
    private final Map<String, Integer> depthFromSource = new HashMap<>();
    private final Map<String, Set<String>> producers = new HashMap<>();
    private final Map<String, Set<String>> consumers = new HashMap<>();
    private final Map<String, Set<String>> undirected = new HashMap<>();
    private int maxDepth = 1;
    private int maxFanIn = 1;
    private int maxFanOut = 1;

    private ServiceDagFeatures() {
    }

    public static ServiceDagFeatures from(Application application) {
        ServiceDagFeatures features = new ServiceDagFeatures();
        if (application == null) {
            return features;
        }
        List<AppModule> modules = application.getModules();
        if (modules != null) {
            for (AppModule module : modules) {
                features.inDegree.putIfAbsent(module.getName(), 0);
                features.outDegree.putIfAbsent(module.getName(), 0);
            }
        }
        Map<String, Set<String>> adj = new HashMap<>();
        List<AppEdge> edges = application.getEdges();
        if (edges != null) {
            for (AppEdge edge : edges) {
                String src = edge.getSource();
                String dst = edge.getDestination();
                if (src == null || dst == null) {
                    continue;
                }
                // Prefer MODULE edges; also accept module↔module links if edgeType is unset/wrong.
                boolean srcIsModule = features.inDegree.containsKey(src);
                boolean dstIsModule = features.inDegree.containsKey(dst);
                boolean moduleEdge = edge.getEdgeType() == AppEdge.MODULE
                        || (srcIsModule && dstIsModule && edge.getEdgeType() != AppEdge.SENSOR
                        && edge.getEdgeType() != AppEdge.ACTUATOR);
                if (moduleEdge && srcIsModule && dstIsModule) {
                    features.outDegree.merge(src, 1, Integer::sum);
                    features.inDegree.merge(dst, 1, Integer::sum);
                    features.producers.computeIfAbsent(dst, k -> new HashSet<>()).add(src);
                    features.consumers.computeIfAbsent(src, k -> new HashSet<>()).add(dst);
                    adj.computeIfAbsent(src, k -> new HashSet<>()).add(dst);
                }
                features.inDegree.putIfAbsent(src, features.inDegree.getOrDefault(src, 0));
                features.outDegree.putIfAbsent(dst, features.outDegree.getOrDefault(dst, 0));
                features.undirected.computeIfAbsent(src, k -> new HashSet<>()).add(dst);
                features.undirected.computeIfAbsent(dst, k -> new HashSet<>()).add(src);
            }
        }
        for (int v : features.inDegree.values()) {
            features.maxFanIn = Math.max(features.maxFanIn, v);
        }
        for (int v : features.outDegree.values()) {
            features.maxFanOut = Math.max(features.maxFanOut, v);
        }
        features.computeDepths(adj);
        return features;
    }

    private void computeDepths(Map<String, Set<String>> adj) {
        Queue<String> queue = new ArrayDeque<>();
        for (Map.Entry<String, Integer> e : inDegree.entrySet()) {
            if (e.getValue() == 0) {
                depthFromSource.put(e.getKey(), 0);
                queue.add(e.getKey());
            }
        }
        if (queue.isEmpty()) {
            for (String name : inDegree.keySet()) {
                depthFromSource.put(name, 0);
            }
            maxDepth = 1;
            return;
        }
        while (!queue.isEmpty()) {
            String u = queue.poll();
            int du = depthFromSource.getOrDefault(u, 0);
            maxDepth = Math.max(maxDepth, du);
            for (String v : adj.getOrDefault(u, Collections.emptySet())) {
                int nd = du + 1;
                Integer old = depthFromSource.get(v);
                if (old == null || nd < old) {
                    depthFromSource.put(v, nd);
                    queue.add(v);
                    maxDepth = Math.max(maxDepth, nd);
                }
            }
        }
        if (maxDepth <= 0) {
            maxDepth = 1;
        }
    }

    public double inDegreeNorm(String moduleName) {
        return maxFanIn <= 0 ? 0.0 : inDegree.getOrDefault(moduleName, 0) / (double) maxFanIn;
    }

    public double outDegreeNorm(String moduleName) {
        return maxFanOut <= 0 ? 0.0 : outDegree.getOrDefault(moduleName, 0) / (double) maxFanOut;
    }

    /** 0 = source-side module, 1 = deepest sink-side module. */
    public double depthNorm(String moduleName) {
        int d = depthFromSource.getOrDefault(moduleName, 0);
        return d / (double) maxDepth;
    }

    public double isSource(String moduleName) {
        return inDegree.getOrDefault(moduleName, 0) == 0 ? 1.0 : 0.0;
    }

    public double isSink(String moduleName) {
        return outDegree.getOrDefault(moduleName, 0) == 0 ? 1.0 : 0.0;
    }

    /** Branching / join pressure used only by the GNN encoder. */
    public double fanInNorm(String moduleName) {
        return inDegreeNorm(moduleName);
    }

    public double fanOutNorm(String moduleName) {
        return outDegreeNorm(moduleName);
    }

    /** Directed producers (MODULE edges into this module). */
    public Set<String> producers(String moduleName) {
        return producers.getOrDefault(moduleName, Collections.emptySet());
    }

    /** Directed consumers (MODULE edges out of this module). */
    public Set<String> consumers(String moduleName) {
        return consumers.getOrDefault(moduleName, Collections.emptySet());
    }

    /** Undirected DAG neighbors (producers/consumers + sensor/actuator links). */
    public Set<String> neighbors(String moduleName) {
        return undirected.getOrDefault(moduleName, Collections.emptySet());
    }
}
