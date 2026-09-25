package org.fog.dynacolgnn.learn;

import java.util.Arrays;
import java.util.Random;

/**
 * Lightweight online DQN over CRT candidates. Each candidate is scored by a shared
 * one-hidden-layer Q-network on flat features; action set size equals CRT size.
 */
public final class OnlineColonyDqn {

    private static final int HID = 16;

    private final double[][] w1;
    private final double[] b1;
    private final double[] w2;
    private double b2;
    private final double learningRate;
    private final double epsilon;
    private final Random random;
    private final int inputDim;

    private double[][] lastFeatures;
    private int lastAction = -1;

    public OnlineColonyDqn(int inputDim, double learningRate, double epsilon, Random random) {
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
            return 0;
        }
        if (random.nextDouble() < epsilon) {
            lastAction = random.nextInt(candidateFeatures.length);
            return lastAction;
        }
        int best = 0;
        double bestQ = qValue(candidateFeatures[0]);
        for (int i = 1; i < candidateFeatures.length; i++) {
            double q = qValue(candidateFeatures[i]);
            if (q > bestQ) {
                bestQ = q;
                best = i;
            }
        }
        lastAction = best;
        return best;
    }

    /** One-step TD update toward observed shaped reward in [0, 1]. */
    public void update(double reward) {
        if (lastFeatures == null || lastAction < 0 || lastAction >= lastFeatures.length) {
            return;
        }
        double target = Math.max(0.0, Math.min(1.0, reward));
        double[] x = lastFeatures[lastAction];
        double[] h = hidden(x);
        double q = b2;
        for (int i = 0; i < HID; i++) {
            q += w2[i] * h[i];
        }
        double err = target - q;
        for (int i = 0; i < HID; i++) {
            w2[i] += learningRate * err * h[i];
        }
        b2 += learningRate * err;
        for (int i = 0; i < HID; i++) {
            double dh = (1.0 - h[i] * h[i]) * (learningRate * err * w2[i]);
            for (int j = 0; j < inputDim; j++) {
                w1[i][j] += dh * x[j];
            }
            b1[i] += dh;
        }
    }

    private double qValue(double[] x) {
        double[] h = hidden(x);
        double q = b2;
        for (int i = 0; i < HID; i++) {
            q += w2[i] * h[i];
        }
        return q;
    }

    private double[] hidden(double[] x) {
        requireDim(x);
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

    private void requireDim(double[] x) {
        if (x == null || x.length != inputDim) {
            throw new IllegalArgumentException("DQN feature dim mismatch");
        }
    }
}
