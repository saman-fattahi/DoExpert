"""doexpert.pareto_selection
============================

Preference-based selection helpers operating directly on a Pareto front
``F`` (minimisation convention): augmented scalarising functions (ASF),
pseudo-weights, and knee/high-trade-off point detection. pymoo is used
when available; pure-NumPy fallbacks are provided otherwise.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compromise_programming_asf", "manual_asf",
    "pseudo_weights_method", "manual_pseudo_weights",
    "high_tradeoff_points", "manual_knee_detection", "manual_tradeoff_detection",
]

def compromise_programming_asf(F, weights):
    """
    PyMOO Compromise Programming using Achievement Scalarizing Function (ASF)
    F: objective values (n_solutions x n_objectives)
    weights: preference weights for objectives
    """
    try:
        from pymoo.decomposition.asf import ASF
        
        decomp = ASF()
        scalarized = decomp(F, weights)
        best_idx = scalarized.argmin()
        
        # Convert to ranking scores (higher is better)
        scores = 1 / (scalarized + 1e-10)  # Inverse ranking
        ranking = np.argsort(scalarized)  # Best has minimum ASF value
        
        return scores, ranking, best_idx
    except ImportError:
        # Fallback to manual ASF implementation
        return manual_asf(F, weights)

def manual_asf(F, weights):
    """
    Manual implementation of Achievement Scalarizing Function
    """
    # Normalize objectives
    F_norm = (F - np.min(F, axis=0)) / (np.max(F, axis=0) - np.min(F, axis=0) + 1e-10)
    
    # Calculate ASF: max over objectives of weighted normalized distance
    asf_values = np.max(weights * F_norm, axis=1)
    best_idx = asf_values.argmin()
    
    scores = 1 / (asf_values + 1e-10)
    ranking = np.argsort(asf_values)
    
    return scores, ranking, best_idx

def pseudo_weights_method(F, preferred_weights):
    """
    PyMOO Pseudo-Weights method for solution selection
    Calculates normalized distance to worst solution
    """
    try:
        from pymoo.mcdm.pseudo_weights import PseudoWeights
        
        pw = PseudoWeights(preferred_weights)
        best_idx, pseudo_weights = pw.do(F, return_pseudo_weights=True)
        
        # Calculate scores based on pseudo weights similarity
        scores = np.sum(pseudo_weights * preferred_weights, axis=1)
        ranking = np.argsort(scores)[::-1]
        
        return scores, ranking, best_idx, pseudo_weights
    except ImportError:
        # Manual implementation
        return manual_pseudo_weights(F, preferred_weights)

def manual_pseudo_weights(F, preferred_weights):
    """
    Manual implementation of pseudo-weights method
    """
    n_solutions, n_objectives = F.shape
    
    # Calculate ranges
    f_max = np.max(F, axis=0)
    f_min = np.min(F, axis=0)
    
    # Calculate pseudo weights for each solution
    pseudo_weights = np.zeros_like(F)
    for i in range(n_solutions):
        numerator = (f_max - F[i]) / (f_max - f_min + 1e-10)
        denominator = np.sum(numerator)
        pseudo_weights[i] = numerator / (denominator + 1e-10)
    
    # Find solution with pseudo weights closest to preferred weights
    distances = np.linalg.norm(pseudo_weights - preferred_weights, axis=1)
    best_idx = distances.argmin()
    
    scores = 1 / (distances + 1e-10)
    ranking = np.argsort(distances)
    
    return scores, ranking, best_idx, pseudo_weights

def high_tradeoff_points(F):
    """
    PyMOO High Trade-off Points (Knee Points) identification
    """
    try:
        from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
        
        dm = HighTradeoffPoints()
        knee_indices = dm(F)
        
        # Create scores: knee points get higher scores
        scores = np.zeros(len(F))
        scores[knee_indices] = 1.0
        
        # Ranking: knee points first, then by original order
        ranking = list(knee_indices) + [i for i in range(len(F)) if i not in knee_indices]
        
        return scores, ranking, knee_indices
    except ImportError:
        # Manual knee point detection
        return manual_knee_detection(F)

def manual_knee_detection(F):
    """
    Manual implementation of knee point detection
    Based on maximum curvature in normalized objective space
    """
    if F.shape[1] != 2:
        # For more than 2 objectives, use trade-off based approach
        return manual_tradeoff_detection(F)
    
    # Normalize objectives
    F_norm = (F - np.min(F, axis=0)) / (np.max(F, axis=0) - np.min(F, axis=0) + 1e-10)
    
    # Sort by first objective
    sorted_indices = np.argsort(F_norm[:, 0])
    F_sorted = F_norm[sorted_indices]
    
    # Calculate curvature (simplified approach)
    curvatures = np.zeros(len(F_sorted))
    for i in range(1, len(F_sorted) - 1):
        # Calculate angle change
        v1 = F_sorted[i] - F_sorted[i-1]
        v2 = F_sorted[i+1] - F_sorted[i]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        curvatures[i] = 1 - cos_angle  # Higher for sharper angles
    
    # Find knee points (local maxima of curvature)
    knee_threshold = np.percentile(curvatures, 75)  # Top 25% curvature
    knee_indices = sorted_indices[curvatures > knee_threshold]
    
    scores = np.zeros(len(F))
    scores[knee_indices] = curvatures[curvatures > knee_threshold]
    
    ranking = knee_indices[np.argsort(scores[knee_indices])[::-1]].tolist()
    ranking += [i for i in range(len(F)) if i not in knee_indices]
    
    return scores, ranking, knee_indices

def manual_tradeoff_detection(F):
    """
    Manual trade-off detection for many-objective problems
    """
    n_solutions, n_objectives = F.shape
    
    # Normalize objectives
    F_norm = (F - np.min(F, axis=0)) / (np.max(F, axis=0) - np.min(F, axis=0) + 1e-10)
    
    # Calculate trade-off scores based on balanced performance
    # Solutions with moderate values in all objectives (avoiding extremes)
    trade_off_scores = np.zeros(n_solutions)
    
    for i in range(n_solutions):
        # Calculate how "balanced" this solution is
        # Good trade-off solutions avoid being too extreme in any objective
        extremeness = np.max(F_norm[i]) - np.min(F_norm[i])
        trade_off_scores[i] = 1 / (extremeness + 1e-10)
    
    # Select top trade-off points
    threshold = np.percentile(trade_off_scores, 75)
    tradeoff_indices = np.where(trade_off_scores > threshold)[0]
    
    ranking = tradeoff_indices[np.argsort(trade_off_scores[tradeoff_indices])[::-1]].tolist()
    ranking += [i for i in range(n_solutions) if i not in tradeoff_indices]
    
    return trade_off_scores, ranking, tradeoff_indices

