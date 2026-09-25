package org.fog.dynacol.placement;

import org.fog.dynacol.model.FogNodeState;
import org.fog.dynacol.model.ServiceRequest;

import java.util.Optional;

/**
 * Placement strategies that support hierarchical L1/L2/L3 reconcile (DCBO and Dyna-Bound).
 */
public interface ReconcilablePlacement extends PlacementStrategy {

    Optional<Integer> reconcileService(ServiceRequest request,
                                       FogNodeState localFcmState,
                                       Integer currentHostId,
                                       PlacementDepth maxDepth);
}
