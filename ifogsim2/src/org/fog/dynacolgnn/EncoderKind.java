package org.fog.dynacolgnn;

/**
 * Encoder used at L2 inside {@link ColonyBoundedPlacement}.
 */
public enum EncoderKind {
    /** Flat feature vector + online linear policy (hybrid ablation). */
    VECTOR,
    /** Online 2-layer GraphSAGE over CRT + service-graph context (hybrid GNN). */
    GNN
}
