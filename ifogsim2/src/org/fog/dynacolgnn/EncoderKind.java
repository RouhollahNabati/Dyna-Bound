package org.fog.dynacolgnn;

/**
 * Encoder / learner used at L2 inside {@link ColonyBoundedPlacement}.
 */
public enum EncoderKind {
    /** Flat feature vector + online linear policy (hybrid ablation). */
    VECTOR,
    /** Online 2-layer GraphSAGE over CRT + service-graph context (hybrid GNN). */
    GNN,
    /** Flat features + online DQN; CRT-bounded action set (same-plane baseline). */
    DQN,
    /** Flat features + lightweight PPO-style policy; CRT-bounded (same-plane baseline). */
    PPO;

    public boolean usesFlatFeatures() {
        return this == VECTOR || this == DQN || this == PPO;
    }

    public boolean isPureLearnerBaseline() {
        return this == DQN || this == PPO;
    }
}
