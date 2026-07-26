package org.fog.dynacolgnn.learn;

import java.util.Arrays;
import java.util.Random;

/**
 * Online linear scorer with optional ε-greedy exploration.
 * Used by the vector ablation; GNN path uses {@link OnlineColonyGnn}.
 */
public final class OnlineLinearPolicy {

    private final double[] weights;
    private final double learningRate;
    private final double epsilon;
    private final Random random;

    public OnlineLinearPolicy(int dim, double learningRate, double epsilon, Random random) {
        this.weights = new double[dim];
        Arrays.fill(this.weights, 0.0);
        for (int i = 0; i < dim; i++) {
            this.weights[i] = 0.01 * ((i % 5) - 2);
        }
        this.learningRate = learningRate;
        this.epsilon = Math.max(0.0, Math.min(1.0, epsilon));
        this.random = random != null ? random : new Random(0L);
    }

    public int dimension() {
        return weights.length;
    }

    public double score(double[] features) {
        requireDim(features);
        double sum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * features[i];
        }
        return sum;
    }

    /**
     * Higher score is better for selection. Returns index into candidates.
     */
    public int selectIndex(double[][] candidateFeatures) {
        if (candidateFeatures == null || candidateFeatures.length == 0) {
            return -1;
        }
        if (candidateFeatures.length == 1) {
            return 0;
        }
        if (random.nextDouble() < epsilon) {
            return random.nextInt(candidateFeatures.length);
        }
        int best = 0;
        double bestScore = score(candidateFeatures[0]);
        for (int i = 1; i < candidateFeatures.length; i++) {
            double s = score(candidateFeatures[i]);
            if (s > bestScore) {
                bestScore = s;
                best = i;
            }
        }
        return best;
    }

    /**
     * Higher score is better. Combines baseUtilities (e.g. inverted DCBO J)
     * with the learned linear score so an untrained policy stays near DCBO.
     */
    public int selectHybridIndex(double[][] candidateFeatures,
                                 double[] baseUtilities,
                                 double learnBlend) {
        if (candidateFeatures == null || candidateFeatures.length == 0) {
            return -1;
        }
        if (candidateFeatures.length == 1) {
            return 0;
        }
        if (baseUtilities == null || baseUtilities.length != candidateFeatures.length) {
            return selectIndex(candidateFeatures);
        }
        if (random.nextDouble() < epsilon) {
            return random.nextInt(candidateFeatures.length);
        }
        double blend = Math.max(0.0, learnBlend);
        int best = 0;
        double bestScore = baseUtilities[0] + blend * score(candidateFeatures[0]);
        for (int i = 1; i < candidateFeatures.length; i++) {
            double s = baseUtilities[i] + blend * score(candidateFeatures[i]);
            if (s > bestScore) {
                bestScore = s;
                best = i;
            }
        }
        return best;
    }

    /** Reinforce chosen features toward observed reward in [0, 1]. */
    public void update(double[] features, double reward) {
        requireDim(features);
        double clipped = Math.max(0.0, Math.min(1.0, reward));
        double prediction = sigmoid(score(features));
        double error = clipped - prediction;
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error * features[i];
        }
    }

    public double[] weightsCopy() {
        return Arrays.copyOf(weights, weights.length);
    }

    private void requireDim(double[] features) {
        if (features == null || features.length != weights.length) {
            throw new IllegalArgumentException("feature dim mismatch");
        }
    }

    private static double sigmoid(double x) {
        if (x > 30) {
            return 1.0;
        }
        if (x < -30) {
            return 0.0;
        }
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
