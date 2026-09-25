package org.fog.dynacolgnn.learn;

import java.util.Random;

/**
 * Lightweight CRT-bounded PPO-style policy: shared MLP scores candidates, softmax
 * samples an index, and a clipped likelihood-ratio update uses a moving baseline.
 */
public final class OnlineColonyPpo {

    private static final int HID = 16;
    private static final double CLIP = 0.20;

    private final double[][] w1;
    private final double[] b1;
    private final double[] w2;
    private double b2;
    private final double learningRate;
    private final double epsilon;
    private final Random random;
    private final int inputDim;

    private double[][] lastFeatures;
    private double[] lastProbs;
    private int lastAction = -1;
    private double baseline = 0.5;

    public OnlineColonyPpo(int inputDim, double learningRate, double epsilon, Random random) {
        this.inputDim = inputDim;
        this.learningRate = learningRate;
        this.epsilon = Math.max(0.0, Math.min(1.0, epsilon));
        this.random = random != null ? random : new Random(0L);
        this.w1 = new double[HID][inputDim];
        this.b1 = new double[HID];
        this.w2 = new double[HID];
        this.b2 = 0.0;
        Random init = new Random(this.random.nextLong());
        for (int h = 0; h < HID; h++) {
            for (int i = 0; i < inputDim; i++) {
                w1[h][i] = 0.02 * (init.nextDouble() - 0.5);
            }
            b1[h] = 0.0;
            w2[h] = 0.02 * (init.nextDouble() - 0.5);
        }
    }

    public int selectIndex(double[][] candidateFeatures) {
        if (candidateFeatures == null || candidateFeatures.length == 0) {
            return -1;
        }
        lastFeatures = candidateFeatures;
        if (candidateFeatures.length == 1) {
            lastAction = 0;
            lastProbs = new double[]{1.0};
            return 0;
        }
        if (random.nextDouble() < epsilon) {
            lastAction = random.nextInt(candidateFeatures.length);
            lastProbs = softmaxLogits(candidateFeatures);
            return lastAction;
        }
        lastProbs = softmaxLogits(candidateFeatures);
        double u = random.nextDouble();
        double cum = 0.0;
        int chosen = candidateFeatures.length - 1;
        for (int i = 0; i < lastProbs.length; i++) {
            cum += lastProbs[i];
            if (u <= cum) {
                chosen = i;
                break;
            }
        }
        lastAction = chosen;
        return chosen;
    }

    public void update(double reward) {
        if (lastFeatures == null || lastProbs == null
                || lastAction < 0 || lastAction >= lastFeatures.length) {
            return;
        }
        double r = Math.max(0.0, Math.min(1.0, reward));
        double advantage = r - baseline;
        baseline = 0.95 * baseline + 0.05 * r;
        double[] newProbs = softmaxLogits(lastFeatures);
        double oldP = Math.max(1e-8, lastProbs[lastAction]);
        double newP = Math.max(1e-8, newProbs[lastAction]);
        double ratio = newP / oldP;
        double clipped = Math.max(1.0 - CLIP, Math.min(1.0 + CLIP, ratio));
        double surrogate = Math.min(ratio * advantage, clipped * advantage);
        // Ascend on surrogate via finite-difference-style nudge on chosen logit path.
        double scale = learningRate * surrogate;
        double[] x = lastFeatures[lastAction];
        double[] h = hidden(x);
        for (int i = 0; i < HID; i++) {
            w2[i] += scale * h[i];
        }
        b2 += scale;
        for (int i = 0; i < HID; i++) {
            double dh = (1.0 - h[i] * h[i]) * (scale * w2[i]);
            for (int j = 0; j < inputDim; j++) {
                w1[i][j] += dh * x[j];
            }
            b1[i] += dh;
        }
    }

    private double[] softmaxLogits(double[][] features) {
        int n = features.length;
        double[] logits = new double[n];
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            logits[i] = score(features[i]);
            if (logits[i] > max) {
                max = logits[i];
            }
        }
        double sum = 0.0;
        double[] probs = new double[n];
        for (int i = 0; i < n; i++) {
            probs[i] = Math.exp(logits[i] - max);
            sum += probs[i];
        }
        if (sum <= 0.0) {
            ArraysFill(probs, 1.0 / n);
            return probs;
        }
        for (int i = 0; i < n; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    private double score(double[] x) {
        double[] h = hidden(x);
        double s = b2;
        for (int i = 0; i < HID; i++) {
            s += w2[i] * h[i];
        }
        return s;
    }

    private double[] hidden(double[] x) {
        if (x == null || x.length != inputDim) {
            throw new IllegalArgumentException("PPO feature dim mismatch");
        }
        double[] h = new double[HID];
        for (int i = 0; i < HID; i++) {
            double s = b1[i];
            for (int j = 0; j < inputDim; j++) {
                s += w1[i][j] * x[j];
            }
            h[i] = Math.tanh(s);
        }
        return h;
    }

    private static void ArraysFill(double[] a, double v) {
        for (int i = 0; i < a.length; i++) {
            a[i] = v;
        }
    }
}
