package org.fog.dynacolgnn.learn;

import org.fog.dynacol.model.ColonyResourceEntry;
import org.fog.entities.FogDevice;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Online 2-layer GraphSAGE-style GNN over CRT candidates.
 * Adjacency is structure-aware (RTT + coloc + producer pull) via
 * {@link ColonyFeatureEncoder#structureAffinityWeights}; ablates to RTT-only
 * with {@code -Ddynacolgnn.structureAdj=false}.
 *
 * <p>Forward: {@code h' = tanh(W_self h + W_neigh A h)}, two layers, then linear
 * readout on {@code concat(h, context)} plus a mild coloc/pull residual.
 * Update: softmax policy-gradient (REINFORCE) over all CRT candidates.
 */
public final class OnlineColonyGnn {

    /** Full host embedding (incl. neighbor-pressure / producer-message channels). */
    public static final int IN_DIM = ColonyFeatureEncoder.HOST_DIM;
    public static final int HID = 16;
    public static final int CTX_DIM = 8;

    private final double[][] w1Self = new double[HID][IN_DIM];
    private final double[][] w1Neigh = new double[HID][IN_DIM];
    private final double[][] w2Self = new double[HID][HID];
    private final double[][] w2Neigh = new double[HID][HID];
    private final double[] readout = new double[HID + CTX_DIM];

    private final double learningRate;
    private final double epsilon;
    private final Random random;
    private boolean onlineUpdatesEnabled = true;
    /** EMA of recent rewards for centered advantages. */
    private double rewardBaseline = 0.5;

    /** Cached forward for the last {@link #selectHybridIndex} call. */
    private double[][] cacheX;
    private double[][] cacheAgg1;
    private double[][] cacheH1;
    private double[][] cacheAgg2;
    private double[][] cacheH2;
    private double[][] cacheCtx;
    private double[][] cacheAdj;
    private int cacheN;

    public OnlineColonyGnn(double learningRate, double epsilon, Random random) {
        this.learningRate = learningRate;
        this.epsilon = Math.max(0.0, Math.min(1.0, epsilon));
        this.random = random != null ? random : new Random(0L);
        initWeights();
        maybeLoadImportedWeights();
    }

    /** When offline weights are loaded, skip online delta updates. */
    public void setOnlineUpdatesEnabled(boolean enabled) {
        this.onlineUpdatesEnabled = enabled;
    }

    public boolean isOnlineUpdatesEnabled() {
        return onlineUpdatesEnabled;
    }

    public double[][] lastX() {
        return cacheX;
    }

    public double[][] lastCtx() {
        return cacheCtx;
    }

    public double[][] lastAdj() {
        return cacheAdj;
    }

    private void maybeLoadImportedWeights() {
        String prop = System.getProperty("dynacolgnn.gnnWeights");
        if (prop == null || prop.isBlank()) {
            return;
        }
        try {
            loadWeights(Path.of(prop.trim()));
            String fineTune = System.getProperty("dynacolgnn.gnnFineTune", "false");
            onlineUpdatesEnabled = "true".equalsIgnoreCase(fineTune.trim());
            System.out.println("[dynacolgnn] loaded GNN weights from " + prop.trim()
                    + (onlineUpdatesEnabled ? " (online fine-tune ON)" : " (online updates frozen)"));
        } catch (IOException e) {
            System.err.println("[dynacolgnn] gnnWeights load failed: " + e.getMessage());
        }
    }

    /**
     * Load JSON exported by {@code dyna-bound/offline/train_gin_ppo.py}.
     * Keys: w1_self, w1_neigh, w2_self, w2_neigh, readout (row-major matrices).
     */
    public void loadWeights(Path jsonPath) throws IOException {
        String raw = Files.readString(jsonPath, StandardCharsets.UTF_8);
        copyMatrix(parseMatrix(raw, "w1_self"), w1Self, HID, IN_DIM);
        copyMatrix(parseMatrix(raw, "w1_neigh"), w1Neigh, HID, IN_DIM);
        copyMatrix(parseMatrix(raw, "w2_self"), w2Self, HID, HID);
        copyMatrix(parseMatrix(raw, "w2_neigh"), w2Neigh, HID, HID);
        double[] ro = parseVector(raw, "readout");
        if (ro.length != readout.length) {
            throw new IOException("readout dim mismatch: " + ro.length + " vs " + readout.length);
        }
        System.arraycopy(ro, 0, readout, 0, readout.length);
    }

    public void saveWeights(Path jsonPath) throws IOException {
        if (jsonPath.getParent() != null) {
            Files.createDirectories(jsonPath.getParent());
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"in_dim\": ").append(IN_DIM).append(",\n");
        sb.append("  \"hid\": ").append(HID).append(",\n");
        sb.append("  \"ctx_dim\": ").append(CTX_DIM).append(",\n");
        sb.append("  \"w1_self\": ");
        appendMatrixJson(sb, w1Self);
        sb.append(",\n  \"w1_neigh\": ");
        appendMatrixJson(sb, w1Neigh);
        sb.append(",\n  \"w2_self\": ");
        appendMatrixJson(sb, w2Self);
        sb.append(",\n  \"w2_neigh\": ");
        appendMatrixJson(sb, w2Neigh);
        sb.append(",\n  \"readout\": ");
        appendVectorJson(sb, readout);
        sb.append("\n}\n");
        Files.writeString(jsonPath, sb.toString(), StandardCharsets.UTF_8);
    }

    private void initWeights() {
        // Small orthogonal-ish init; bias coloc/producer context in readout.
        for (int i = 0; i < HID; i++) {
            for (int j = 0; j < IN_DIM; j++) {
                w1Self[i][j] = 0.08 * (((i + j) % 5) - 2);
                w1Neigh[i][j] = 0.05 * (((i * 3 + j) % 5) - 2);
            }
            for (int j = 0; j < HID; j++) {
                w2Self[i][j] = (i == j) ? 0.15 : 0.02 * (((i + j) % 3) - 1);
                w2Neigh[i][j] = 0.04 * (((i + 2 * j) % 5) - 2);
            }
        }
        Arrays.fill(readout, 0.0);
        for (int i = 0; i < HID; i++) {
            readout[i] = 0.02 * ((i % 5) - 2);
        }
        // Context layout from ColonyFeatureEncoder GNN slots: [..., pull, fan, coloc]
        readout[HID + 5] = 0.55; // producer-RTT pull (graph-SLA)
        readout[HID + 6] = 0.20; // fan-in
        readout[HID + 7] = 0.60; // coloc
    }

    /**
     * @param features {@link ColonyFeatureEncoder#FEATURE_DIM} rows (host || service || bias)
     * @param baseUtilities inverted DCBO utilities (same length)
     * @param colocAffinity per-candidate producer co-location in \([0,1]\), or null
     */
    public int selectHybridIndex(List<ColonyResourceEntry> candidates,
                                 Map<Integer, FogDevice> deviceIndex,
                                 double[][] features,
                                 double[] baseUtilities,
                                 double learnBlend,
                                 double[] colocAffinity) {
        if (features == null || features.length == 0) {
            return -1;
        }
        if (features.length == 1) {
            forward(candidates, deviceIndex, features, colocAffinity);
            return 0;
        }
        if (random.nextDouble() < epsilon) {
            forward(candidates, deviceIndex, features, colocAffinity);
            return random.nextInt(features.length);
        }
        double[] scores = forward(candidates, deviceIndex, features, colocAffinity);
        double blend = Math.max(0.0, learnBlend);
        int best = 0;
        double bestScore = (baseUtilities != null ? baseUtilities[0] : 0.0) + blend * scores[0];
        for (int i = 1; i < scores.length; i++) {
            double base = (baseUtilities != null && i < baseUtilities.length) ? baseUtilities[i] : 0.0;
            double s = base + blend * scores[i];
            if (s > bestScore) {
                bestScore = s;
                best = i;
            }
        }
        return best;
    }

    /** Backward-compatible overload (RTT/structure without explicit coloc vector). */
    public int selectHybridIndex(List<ColonyResourceEntry> candidates,
                                 Map<Integer, FogDevice> deviceIndex,
                                 double[][] features,
                                 double[] baseUtilities,
                                 double learnBlend) {
        return selectHybridIndex(candidates, deviceIndex, features, baseUtilities, learnBlend, null);
    }

    /** Online contrastive update over all CRT candidates (REINFORCE-style). */
    public void update(int chosenIndex, double reward) {
        update(chosenIndex, reward, -1);
    }

    /**
     * @param bestIndex locally best CRT candidate by attractiveness+utility, or -1
     */
    public void update(int chosenIndex, double reward, int bestIndex) {
        if (!onlineUpdatesEnabled) {
            return;
        }
        if (cacheH2 == null || chosenIndex < 0 || chosenIndex >= cacheN || cacheN <= 1) {
            return;
        }
        double clipped = Math.max(0.0, Math.min(1.0, reward));
        rewardBaseline = 0.95 * rewardBaseline + 0.05 * clipped;
        double advantage = clipped - rewardBaseline;
        if (Math.abs(advantage) < 0.02) {
            advantage = clipped >= 0.5 ? 0.02 : -0.02;
        }

        double[] logits = new double[cacheN];
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < cacheN; i++) {
            logits[i] = scoreFromCache(i);
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
            }
        }
        double[] probs = new double[cacheN];
        double z = 0.0;
        for (int i = 0; i < cacheN; i++) {
            probs[i] = Math.exp(Math.min(30.0, logits[i] - maxLogit));
            z += probs[i];
        }
        if (z <= 0.0) {
            return;
        }
        for (int i = 0; i < cacheN; i++) {
            probs[i] /= z;
        }

        double step = learningRate * Math.max(0.5, Math.min(2.5, Math.abs(advantage) * 5.0));
        for (int i = 0; i < cacheN; i++) {
            double indicator = (i == chosenIndex) ? 1.0 : 0.0;
            double coef = advantage * (indicator - probs[i]);
            if (bestIndex >= 0 && bestIndex < cacheN) {
                double bestInd = (i == bestIndex) ? 1.0 : 0.0;
                coef += 0.40 * (bestInd - probs[i]);
            }
            if (Math.abs(coef) < 1e-12) {
                continue;
            }
            applyCandidateGradient(i, step * coef);
        }
    }

    /** Match {@link #forward} scoring, including structure residual. */
    private double scoreFromCache(int i) {
        return dot(readout, concat(cacheH2[i], cacheCtx[i])) + structureResidual(cacheCtx[i]);
    }

    private static double structureResidual(double[] ctx) {
        return 0.18 * ctx[5] + 0.08 * ctx[6] + 0.15 * ctx[7];
    }

    /** Push gradient into readout + both GNN layers for one candidate. */
    private void applyCandidateGradient(int idx, double scale) {
        double[] concat = concat(cacheH2[idx], cacheCtx[idx]);
        for (int k = 0; k < readout.length; k++) {
            readout[k] += scale * concat[k];
        }

        double[] dh2 = new double[HID];
        for (int h = 0; h < HID; h++) {
            double h2 = cacheH2[idx][h];
            dh2[h] = scale * readout[h] * (1.0 - h2 * h2);
        }
        for (int h = 0; h < HID; h++) {
            for (int k = 0; k < HID; k++) {
                w2Self[h][k] += dh2[h] * cacheH1[idx][k];
                w2Neigh[h][k] += dh2[h] * cacheAgg2[idx][k];
            }
        }

        double[] dh1 = new double[HID];
        for (int k = 0; k < HID; k++) {
            double acc = 0.0;
            for (int h = 0; h < HID; h++) {
                acc += w2Self[h][k] * dh2[h];
            }
            double h1 = cacheH1[idx][k];
            dh1[k] = acc * (1.0 - h1 * h1);
        }
        for (int h = 0; h < HID; h++) {
            for (int k = 0; k < IN_DIM; k++) {
                w1Self[h][k] += dh1[h] * cacheX[idx][k];
                w1Neigh[h][k] += dh1[h] * cacheAgg1[idx][k];
            }
        }
    }

    private double[] forward(List<ColonyResourceEntry> candidates,
                             Map<Integer, FogDevice> deviceIndex,
                             double[][] features,
                             double[] colocAffinity) {
        int n = features.length;
        cacheN = n;
        cacheAdj = ColonyFeatureEncoder.structureAffinityWeights(candidates, deviceIndex, colocAffinity);
        cacheX = new double[n][IN_DIM];
        cacheCtx = new double[n][CTX_DIM];
        for (int i = 0; i < n; i++) {
            System.arraycopy(features[i], 0, cacheX[i], 0, IN_DIM);
            // service / structure context starts at HOST_DIM (= IN_DIM)
            int hostDim = ColonyFeatureEncoder.HOST_DIM;
            for (int c = 0; c < CTX_DIM; c++) {
                cacheCtx[i][c] = features[i][hostDim + c];
            }
        }

        cacheAgg1 = aggregate(cacheX, cacheAdj);
        cacheH1 = new double[n][HID];
        for (int i = 0; i < n; i++) {
            cacheH1[i] = layer(w1Self, w1Neigh, cacheX[i], cacheAgg1[i], IN_DIM);
        }

        cacheAgg2 = aggregate(cacheH1, cacheAdj);
        cacheH2 = new double[n][HID];
        double[] scores = new double[n];
        for (int i = 0; i < n; i++) {
            cacheH2[i] = layer(w2Self, w2Neigh, cacheH1[i], cacheAgg2[i], HID);
            scores[i] = dot(readout, concat(cacheH2[i], cacheCtx[i])) + structureResidual(cacheCtx[i]);
        }
        return scores;
    }

    private static double[] layer(double[][] wSelf, double[][] wNeigh,
                                  double[] self, double[] neigh, int inDim) {
        double[] out = new double[wSelf.length];
        for (int h = 0; h < wSelf.length; h++) {
            double sum = 0.0;
            for (int k = 0; k < inDim; k++) {
                sum += wSelf[h][k] * self[k] + wNeigh[h][k] * neigh[k];
            }
            out[h] = Math.tanh(sum);
        }
        return out;
    }

    private static double[][] aggregate(double[][] nodes, double[][] adj) {
        int n = nodes.length;
        int dim = nodes[0].length;
        double[][] out = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) {
                double acc = 0.0;
                for (int j = 0; j < n; j++) {
                    acc += adj[i][j] * nodes[j][d];
                }
                out[i][d] = acc;
            }
        }
        return out;
    }

    private static double[] concat(double[] a, double[] b) {
        double[] out = new double[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
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

    private static void copyMatrix(double[][] src, double[][] dst, int rows, int cols) throws IOException {
        if (src.length != rows || src[0].length != cols) {
            throw new IOException("matrix shape " + src.length + "x" + src[0].length
                    + " expected " + rows + "x" + cols);
        }
        for (int i = 0; i < rows; i++) {
            System.arraycopy(src[i], 0, dst[i], 0, cols);
        }
    }

    private static double[][] parseMatrix(String json, String key) throws IOException {
        String body = extractArrayBody(json, key);
        // Split top-level rows: [[...],[...]]
        List<double[]> rows = new java.util.ArrayList<>();
        int depth = 0;
        int start = -1;
        for (int i = 0; i < body.length(); i++) {
            char c = body.charAt(i);
            if (c == '[') {
                if (depth == 0) {
                    start = i + 1;
                }
                depth++;
            } else if (c == ']') {
                depth--;
                if (depth == 0 && start >= 0) {
                    rows.add(parseNumbers(body.substring(start, i)));
                    start = -1;
                }
            }
        }
        if (rows.isEmpty()) {
            throw new IOException("empty matrix for " + key);
        }
        return rows.toArray(new double[0][]);
    }

    private static double[] parseVector(String json, String key) throws IOException {
        return parseNumbers(extractArrayBody(json, key));
    }

    private static String extractArrayBody(String json, String key) throws IOException {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[", Pattern.DOTALL);
        Matcher m = p.matcher(json);
        if (!m.find()) {
            throw new IOException("missing key " + key);
        }
        int i = m.end() - 1; // at '['
        int depth = 0;
        int start = i;
        for (; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '[') {
                depth++;
            } else if (c == ']') {
                depth--;
                if (depth == 0) {
                    return json.substring(start + 1, i);
                }
            }
        }
        throw new IOException("unclosed array for " + key);
    }

    private static double[] parseNumbers(String body) {
        String[] parts = body.split(",");
        java.util.List<Double> vals = new java.util.ArrayList<>();
        for (String part : parts) {
            String t = part.trim();
            if (t.isEmpty() || t.equals("[") || t.equals("]")) {
                continue;
            }
            t = t.replace("[", "").replace("]", "").trim();
            if (t.isEmpty()) {
                continue;
            }
            vals.add(Double.parseDouble(t));
        }
        double[] out = new double[vals.size()];
        for (int i = 0; i < vals.size(); i++) {
            out[i] = vals.get(i);
        }
        return out;
    }

    private static void appendMatrixJson(StringBuilder sb, double[][] m) {
        sb.append('[');
        for (int i = 0; i < m.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            appendVectorJson(sb, m[i]);
        }
        sb.append(']');
    }

    private static void appendVectorJson(StringBuilder sb, double[] v) {
        sb.append('[');
        for (int i = 0; i < v.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(String.format(Locale.US, "%.8f", v[i]));
        }
        sb.append(']');
    }
}
