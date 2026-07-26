package org.fog.dynacolgnn.learn;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Appends placement decisions as JSONL for offline GIN/PPO training.
 * Enable with {@code -Ddynacolgnn.trajLog=/path/to/traj.jsonl}.
 */
public final class TrajectoryLogger {

    private static final TrajectoryLogger INSTANCE = new TrajectoryLogger();

    private final Object lock = new Object();
    private Path path;
    private BufferedWriter writer;
    private final AtomicLong step = new AtomicLong();

    private TrajectoryLogger() {
    }

    public static TrajectoryLogger getInstance() {
        return INSTANCE;
    }

    public void ensureOpen() {
        if (writer != null) {
            return;
        }
        String prop = System.getProperty("dynacolgnn.trajLog");
        if (prop == null || prop.isBlank()) {
            return;
        }
        synchronized (lock) {
            if (writer != null) {
                return;
            }
            try {
                path = Path.of(prop.trim());
                if (path.getParent() != null) {
                    Files.createDirectories(path.getParent());
                }
                writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } catch (IOException e) {
                System.err.println("[dynacolgnn] trajLog open failed: " + e.getMessage());
                writer = null;
            }
        }
    }

    public boolean isEnabled() {
        ensureOpen();
        return writer != null;
    }

    public void logDecision(double[][] x,
                            double[][] ctx,
                            double[][] adj,
                            double[] base,
                            double blend,
                            int action,
                            double reward,
                            String module) {
        if (!isEnabled() || x == null || action < 0 || action >= x.length) {
            return;
        }
        synchronized (lock) {
            if (writer == null) {
                return;
            }
            try {
                long t = step.getAndIncrement();
                StringBuilder sb = new StringBuilder(512);
                sb.append("{\"t\":").append(t);
                sb.append(",\"module\":\"").append(escape(module)).append('"');
                sb.append(",\"n\":").append(x.length);
                sb.append(",\"action\":").append(action);
                sb.append(",\"reward\":").append(fmt(reward));
                sb.append(",\"blend\":").append(fmt(blend));
                String scenario = System.getProperty("dynacolgnn.trajScenario", "");
                if (scenario != null && !scenario.isBlank()) {
                    sb.append(",\"scenario\":\"").append(escape(scenario.trim())).append('"');
                }
                sb.append(",\"x\":");
                appendMatrix(sb, x);
                sb.append(",\"ctx\":");
                appendMatrix(sb, ctx);
                sb.append(",\"adj\":");
                appendMatrix(sb, adj);
                if (base != null) {
                    sb.append(",\"base\":");
                    appendVector(sb, base);
                }
                sb.append("}\n");
                writer.write(sb.toString());
                writer.flush();
            } catch (IOException e) {
                System.err.println("[dynacolgnn] trajLog write failed: " + e.getMessage());
            }
        }
    }

    public void close() {
        synchronized (lock) {
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException ignored) {
                    // ignore
                }
                writer = null;
            }
        }
    }

    private static void appendMatrix(StringBuilder sb, double[][] m) {
        sb.append('[');
        for (int i = 0; i < m.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            appendVector(sb, m[i]);
        }
        sb.append(']');
    }

    private static void appendVector(StringBuilder sb, double[] v) {
        sb.append('[');
        for (int i = 0; i < v.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(fmt(v[i]));
        }
        sb.append(']');
    }

    private static String fmt(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            return "0";
        }
        return String.format(Locale.US, "%.6f", v);
    }

    private static String escape(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
