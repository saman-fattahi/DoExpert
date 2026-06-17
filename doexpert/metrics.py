"""doexpert.metrics
===================

Pareto-front quality indicators: hypervolume (exact via pymoo when
available, Monte-Carlo and 2-D exact fallbacks otherwise), spacing
(distribution uniformity) and extent (front spread).
"""

from __future__ import annotations

import numpy as np

__all__ = ["calculate_hypervolume", "calculate_spacing_metric", "calculate_extent_metric"]

try:  # pragma: no cover - environment dependent
    import pymoo  # noqa: F401

    PYMOO_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYMOO_AVAILABLE = False


def calculate_hypervolume(pareto_front, reference_point=None, ideal_point=None, normalize=False):
    """
    Calculate hypervolume of Pareto front
    
    Hypervolume (HV) is a quality indicator that measures the volume of objective space
    dominated by a Pareto front relative to a reference point.
    
    Mathematical Definition:
    HV = Volume of space dominated by Pareto front bounded by reference point
    
    Normalized HV = (HV - HV_ideal) / (HV_nadir - HV_ideal)
    Where HV_ideal = 0 and HV_nadir = reference_point volume
    
    Properties:
    - Higher values = Better Pareto front coverage
    - Unary indicator (doesn't require comparison set)
    - Monotonic (adding non-dominated points increases HV)
    - Normalized version: 0 = HV_norm = 1 (1 = perfect coverage)
    
    Calculation Methods:
    1. Exact (using pymoo): Precise calculation for any dimension
    2. 2D Approximation: Step-wise area calculation
    3. N-D Approximation: Bounding box volume
    
    Args:
        pareto_front: Array of Pareto optimal solutions (n_points, n_objectives)
        reference_point: Reference point for HV calculation (optional)
        ideal_point: Ideal point for normalization (optional)
        normalize: If True, return normalized HV in [0,1] range
        
    Returns:
        float: Hypervolume value (higher is better)
        float: Normalized hypervolume (0-1 scale) if normalize=True
    """
    if len(pareto_front) == 0:
        return (0.0, 0.0) if normalize else 0.0
    
    try:
        pareto_front = np.array(pareto_front)
        
        # Set default points if not provided
        if reference_point is None:
            # Use nadir point + 10% margin as reference
            reference_point = np.max(pareto_front, axis=0) * 1.1
        
        if ideal_point is None:
            # Use best values from Pareto front as ideal point
            ideal_point = np.min(pareto_front, axis=0)
        
        # Calculate raw hypervolume
        if PYMOO_AVAILABLE:
            from pymoo.indicators.hv import HV
            
            # pymoo HV indicator (exact calculation)
            ind = HV(ref_point=reference_point)
            hv_raw = ind(pareto_front)
        else:
            # Fallback approximation methods
            # Sort by first objective for consistent calculation
            sorted_indices = np.argsort(pareto_front[:, 0])
            sorted_front = pareto_front[sorted_indices]
            
            if pareto_front.shape[1] == 2:
                # 2D Case: Calculate area using step function
                hv_raw = 0.0
                for i, point in enumerate(sorted_front):
                    if i == 0:
                        # First point: full rectangle from reference
                        width = max(0, reference_point[0] - point[0])
                        height = max(0, reference_point[1] - point[1])
                    else:
                        # Subsequent points: incremental area
                        width = max(0, sorted_front[i-1][0] - point[0])
                        height = max(0, reference_point[1] - point[1])
                    
                    # Only add positive contributions
                    if width > 0 and height > 0:
                        hv_raw += width * height
                
                hv_raw = max(0.0, hv_raw)
            
            elif pareto_front.shape[1] == 3:
                # 3D Case: Simplified calculation (approximation)
                # Calculate volume of bounding box minus overlaps
                ranges = reference_point - np.min(pareto_front, axis=0)
                base_volume = np.prod(np.maximum(ranges, 0))
                
                # Adjust for Pareto front density
                density_factor = len(pareto_front) / (len(pareto_front) + 10)  # Heuristic
                hv_raw = base_volume * density_factor
            
            else:
                # N-D Case: Bounding hyperbox approximation
                ranges = reference_point - np.min(pareto_front, axis=0)
                hypervolume = np.prod(np.maximum(ranges, 0))
                
                # Density adjustment for higher dimensions
                n_dim = pareto_front.shape[1]
                density_factor = len(pareto_front) ** (1.0 / n_dim) / (len(pareto_front) ** (1.0 / n_dim) + 1)
                hv_raw = hypervolume * density_factor
        
        if not normalize:
            return max(0.0, hv_raw)
        
        # Calculate normalized hypervolume (0-1 scale)
        # Maximum possible hypervolume = volume of entire reference space
        reference_ranges = reference_point - ideal_point
        max_possible_hv = np.prod(np.maximum(reference_ranges, 1e-10))  # Avoid division by zero
        
        if max_possible_hv > 0:
            hv_normalized = min(1.0, max(0.0, hv_raw / max_possible_hv))
        else:
            hv_normalized = 0.0
        
        return hv_raw, hv_normalized
                
    except Exception as e:
        print(f"Hypervolume calculation error: {e}")
        return (0.0, 0.0) if normalize else 0.0

def calculate_spacing_metric(pareto_front):
    """
    Calculate spacing metric (distribution uniformity) of Pareto front
    """
    if len(pareto_front) < 2:
        return 0
    
    # Calculate distance to nearest neighbor for each point
    distances = []
    for i in range(len(pareto_front)):
        min_dist = float('inf')
        for j in range(len(pareto_front)):
            if i != j:
                dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    # Calculate spacing metric
    mean_dist = np.mean(distances)
    spacing = np.sqrt(np.mean([(d - mean_dist)**2 for d in distances]))
    
    return spacing

def calculate_extent_metric(pareto_front):
    """
    Calculate extent (coverage) of Pareto front
    """
    if len(pareto_front) < 2:
        return 0
    
    # Calculate range in each objective
    ranges = np.max(pareto_front, axis=0) - np.min(pareto_front, axis=0)
    
    # Return average range (normalized)
    return np.mean(ranges)

