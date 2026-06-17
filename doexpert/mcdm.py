"""doexpert.mcdm
=================

Multi-criteria decision-making (MCDM) methods for ranking Pareto-optimal
solutions. All functions share the same interface:

    scores, ranking = method(decision_matrix, weights, criteria_types, **kwargs)

Parameters
----------
decision_matrix : np.ndarray, shape (n_alternatives, n_criteria)
    Objective values of the candidate solutions.
weights : np.ndarray, shape (n_criteria,)
    Non-negative criterion weights (normalised internally where required).
criteria_types : list[str]
    'max' for benefit criteria, 'min' for cost criteria.

Returns
-------
scores : np.ndarray
    Higher is better (for VIKOR the returned score is 1 - Q so that the
    "higher is better" convention holds across all methods).
ranking : np.ndarray
    Indices of alternatives sorted from best to worst.

Extracted from the original DoExpert application (v1.0) into a standalone,
test-covered module in v1.1.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "topsis", "vikor", "promethee_ii", "saw", "waspas", "fuzzy_topsis",
    "AVAILABLE_METHODS", "rank_solutions",
]


def _to_int_types(criteria_types):
    """Normalise criteria types to the integer convention (1=max, -1=min)."""
    out = []
    for c in criteria_types:
        if isinstance(c, str):
            out.append(1 if c.lower().startswith("max") else -1)
        else:
            out.append(1 if int(c) > 0 else -1)
    return out


def _to_str_types(criteria_types):
    """Normalise criteria types to the string convention ('max'/'min')."""
    out = []
    for c in criteria_types:
        if isinstance(c, str):
            out.append("max" if c.lower().startswith("max") else "min")
        else:
            out.append("max" if int(c) > 0 else "min")
    return out


def topsis(decision_matrix, weights, criteria_types):
    criteria_types = _to_str_types(criteria_types)
    normalized_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix**2, axis=0))
    weighted_matrix = normalized_matrix * weights
    
    ideal_solution = np.zeros(weighted_matrix.shape[1])
    negative_ideal_solution = np.zeros(weighted_matrix.shape[1])
    
    for i, criterion_type in enumerate(criteria_types):
        if criterion_type == 'max':
            ideal_solution[i] = np.max(weighted_matrix[:, i])
            negative_ideal_solution[i] = np.min(weighted_matrix[:, i])
        else:
            ideal_solution[i] = np.min(weighted_matrix[:, i])
            negative_ideal_solution[i] = np.max(weighted_matrix[:, i])
    
    distance_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution)**2, axis=1))
    distance_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution)**2, axis=1))
    
    scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal + 1e-10)
    ranking = np.argsort(scores)[::-1]
    
    return scores, ranking

def vikor(decision_matrix, weights, criteria_types, v=0.5):
    """
    VIKOR method for compromise solution
    v: weight for group utility (0.5 = balanced approach)
    """
    criteria_types = _to_int_types(criteria_types)
    # Normalize decision matrix
    normalized_matrix = np.zeros_like(decision_matrix)
    
    for i, criterion_type in enumerate(criteria_types):
        if criterion_type == 1:  # maximize
            f_best = np.max(decision_matrix[:, i])
            f_worst = np.min(decision_matrix[:, i])
        else:  # minimize
            f_best = np.min(decision_matrix[:, i])
            f_worst = np.max(decision_matrix[:, i])
        
        if f_best != f_worst:
            if criterion_type == 1:  # maximize
                normalized_matrix[:, i] = (f_best - decision_matrix[:, i]) / (f_best - f_worst)
            else:  # minimize
                normalized_matrix[:, i] = (decision_matrix[:, i] - f_best) / (f_worst - f_best)
    
    # Calculate S (group utility) and R (individual regret)
    S = np.sum(weights * normalized_matrix, axis=1)
    R = np.max(weights * normalized_matrix, axis=1)
    
    # Calculate VIKOR index
    S_best, S_worst = np.min(S), np.max(S)
    R_best, R_worst = np.min(R), np.max(R)
    
    Q = np.zeros(len(S))
    if S_worst != S_best and R_worst != R_best:
        Q = v * (S - S_best) / (S_worst - S_best) + (1 - v) * (R - R_best) / (R_worst - R_best)
    
    ranking = np.argsort(Q)
    scores = 1 - Q  # Convert to score (higher is better)
    
    return scores, ranking, S, R, Q

def promethee_ii(decision_matrix, weights, criteria_types, preference_functions=None):
    """
    PROMETHEE II method for complete ranking
    preference_functions: list of preference function types for each criterion
    """
    criteria_types = _to_int_types(criteria_types)
    n_alternatives, n_criteria = decision_matrix.shape
    
    # Default preference functions (linear)
    if preference_functions is None:
        preference_functions = ['linear'] * n_criteria
    
    # Calculate preference matrix
    preference_matrix = np.zeros((n_alternatives, n_alternatives))
    
    for a in range(n_alternatives):
        for b in range(n_alternatives):
            if a != b:
                pref_sum = 0
                for j in range(n_criteria):
                    d = decision_matrix[a, j] - decision_matrix[b, j]
                    
                    # Adjust for criteria type
                    if criteria_types[j] == -1:  # minimize
                        d = -d
                    
                    # Apply preference function
                    if preference_functions[j] == 'linear':
                        # Linear preference function
                        pref = max(0, d) / (np.max(decision_matrix[:, j]) - np.min(decision_matrix[:, j]) + 1e-10)
                    else:  # default usual function
                        pref = 1 if d > 0 else 0
                    
                    pref_sum += weights[j] * pref
                
                preference_matrix[a, b] = pref_sum
    
    # Calculate positive and negative flows
    phi_plus = np.mean(preference_matrix, axis=1)  # Positive flow
    phi_minus = np.mean(preference_matrix, axis=0)  # Negative flow
    
    # Calculate net flow (PROMETHEE II)
    phi_net = phi_plus - phi_minus
    
    ranking = np.argsort(phi_net)[::-1]
    scores = (phi_net - np.min(phi_net)) / (np.max(phi_net) - np.min(phi_net) + 1e-10)
    
    return scores, ranking, phi_plus, phi_minus, phi_net

def saw(decision_matrix, weights, criteria_types):
    """
    Simple Additive Weighting (SAW) method
    """
    criteria_types = _to_int_types(criteria_types)
    normalized_matrix = np.zeros_like(decision_matrix)
    
    for i, criterion_type in enumerate(criteria_types):
        if criterion_type == 1:  # maximize
            normalized_matrix[:, i] = decision_matrix[:, i] / np.max(decision_matrix[:, i])
        else:  # minimize
            normalized_matrix[:, i] = np.min(decision_matrix[:, i]) / decision_matrix[:, i]
    
    scores = np.sum(weights * normalized_matrix, axis=1)
    ranking = np.argsort(scores)[::-1]
    
    return scores, ranking

# PyMOO-based MCDM methods

def waspas(decision_matrix, weights, criteria_types, lambda_param=0.5):
    """
    WASPAS method - combination of SAW and WPM
    lambda_param: weight between SAW and WPM (0.5 = balanced)
    """
    criteria_types = _to_int_types(criteria_types)
    # SAW component
    normalized_matrix_saw = np.zeros_like(decision_matrix)
    for i, criterion_type in enumerate(criteria_types):
        if criterion_type == 1:  # maximize
            normalized_matrix_saw[:, i] = decision_matrix[:, i] / np.max(decision_matrix[:, i])
        else:  # minimize
            normalized_matrix_saw[:, i] = np.min(decision_matrix[:, i]) / decision_matrix[:, i]
    
    q1 = np.sum(weights * normalized_matrix_saw, axis=1)
    
    # WPM component
    normalized_matrix_wpm = np.zeros_like(decision_matrix)
    for i, criterion_type in enumerate(criteria_types):
        if criterion_type == 1:  # maximize
            normalized_matrix_wpm[:, i] = decision_matrix[:, i] / np.max(decision_matrix[:, i])
        else:  # minimize
            normalized_matrix_wpm[:, i] = np.min(decision_matrix[:, i]) / decision_matrix[:, i]
    
    q2 = np.prod(normalized_matrix_wpm ** weights, axis=1)
    
    # Combined WASPAS score
    scores = lambda_param * q1 + (1 - lambda_param) * q2
    ranking = np.argsort(scores)[::-1]
    
    return scores, ranking, q1, q2

def fuzzy_topsis(decision_matrix, weights, criteria_types, alpha=0.5):
    """
    Fuzzy TOPSIS for handling uncertainty
    alpha: alpha-cut level for defuzzification
    """
    criteria_types = _to_int_types(criteria_types)
    # For simplicity, we'll use triangular fuzzy numbers
    # Each value becomes (value-std, value, value+std)
    std_dev = np.std(decision_matrix, axis=0)
    
    # Create fuzzy decision matrix (lower, middle, upper)
    fuzzy_lower = decision_matrix - alpha * std_dev
    fuzzy_middle = decision_matrix
    fuzzy_upper = decision_matrix + alpha * std_dev
    
    # Normalize fuzzy decision matrix
    for i in range(decision_matrix.shape[1]):
        max_upper = np.max(fuzzy_upper[:, i])
        if max_upper > 0:
            fuzzy_lower[:, i] = fuzzy_lower[:, i] / max_upper
            fuzzy_middle[:, i] = fuzzy_middle[:, i] / max_upper
            fuzzy_upper[:, i] = fuzzy_upper[:, i] / max_upper
    
    # Weight the normalized fuzzy decision matrix
    weighted_lower = fuzzy_lower * weights
    weighted_middle = fuzzy_middle * weights
    weighted_upper = fuzzy_upper * weights
    
    # Find fuzzy positive and negative ideal solutions
    fpis_lower = np.zeros(decision_matrix.shape[1])
    fpis_middle = np.zeros(decision_matrix.shape[1])
    fpis_upper = np.zeros(decision_matrix.shape[1])
    
    fnis_lower = np.zeros(decision_matrix.shape[1])
    fnis_middle = np.zeros(decision_matrix.shape[1])
    fnis_upper = np.zeros(decision_matrix.shape[1])
    
    for i, criterion_type in enumerate(criteria_types):
        if criterion_type == 1:  # maximize
            fpis_lower[i] = np.max(weighted_lower[:, i])
            fpis_middle[i] = np.max(weighted_middle[:, i])
            fpis_upper[i] = np.max(weighted_upper[:, i])
            
            fnis_lower[i] = np.min(weighted_lower[:, i])
            fnis_middle[i] = np.min(weighted_middle[:, i])
            fnis_upper[i] = np.min(weighted_upper[:, i])
        else:  # minimize
            fpis_lower[i] = np.min(weighted_lower[:, i])
            fpis_middle[i] = np.min(weighted_middle[:, i])
            fpis_upper[i] = np.min(weighted_upper[:, i])
            
            fnis_lower[i] = np.max(weighted_lower[:, i])
            fnis_middle[i] = np.max(weighted_middle[:, i])
            fnis_upper[i] = np.max(weighted_upper[:, i])
    
    # Calculate distances using fuzzy distance formula
    d_pos = np.zeros(decision_matrix.shape[0])
    d_neg = np.zeros(decision_matrix.shape[0])
    
    for i in range(decision_matrix.shape[0]):
        # Distance to FPIS
        d_pos[i] = np.sqrt(np.sum([
            ((weighted_lower[i, j] - fpis_lower[j])**2 + 
             (weighted_middle[i, j] - fpis_middle[j])**2 + 
             (weighted_upper[i, j] - fpis_upper[j])**2) / 3
            for j in range(decision_matrix.shape[1])
        ]))
        
        # Distance to FNIS
        d_neg[i] = np.sqrt(np.sum([
            ((weighted_lower[i, j] - fnis_lower[j])**2 + 
             (weighted_middle[i, j] - fnis_middle[j])**2 + 
             (weighted_upper[i, j] - fnis_upper[j])**2) / 3
            for j in range(decision_matrix.shape[1])
        ]))
    
    # Calculate closeness coefficient
    scores = d_neg / (d_pos + d_neg + 1e-10)
    ranking = np.argsort(scores)[::-1]
    
    return scores, ranking



#: Registry of available MCDM methods (name -> callable).
AVAILABLE_METHODS = {
    "TOPSIS": topsis,
    "VIKOR": vikor,
    "PROMETHEE II": promethee_ii,
    "SAW": saw,
    "WASPAS": waspas,
    "Fuzzy TOPSIS": fuzzy_topsis,
}


def rank_solutions(decision_matrix, weights, criteria_types, method="TOPSIS", **kwargs):
    """Rank alternatives with the selected MCDM method.

    A thin dispatcher around :data:`AVAILABLE_METHODS` that normalises the
    weight vector and harmonises the score convention (higher = better)
    across methods, enabling side-by-side comparison of ranking outcomes.

    Parameters
    ----------
    method : str
        One of ``AVAILABLE_METHODS`` keys (case-insensitive).
    **kwargs
        Method-specific parameters, e.g. ``v`` for VIKOR (default 0.5),
        ``lambda_param`` for WASPAS (default 0.5), ``alpha`` for Fuzzy
        TOPSIS (default 0.5).

    Returns
    -------
    scores : np.ndarray  (higher is better)
    ranking : np.ndarray (best-to-worst indices)
    """
    key = {k.lower(): k for k in AVAILABLE_METHODS}.get(str(method).lower())
    if key is None:
        raise ValueError(
            f"Unknown MCDM method '{method}'. "
            f"Available: {sorted(AVAILABLE_METHODS)}"
        )
    decision_matrix = np.asarray(decision_matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    s = weights.sum()
    if s <= 0:
        weights = np.ones(decision_matrix.shape[1]) / decision_matrix.shape[1]
    else:
        weights = weights / s
    result = AVAILABLE_METHODS[key](decision_matrix, weights, criteria_types, **kwargs)
    # All methods return (scores, ranking, *diagnostics) with scores already
    # following the higher-is-better convention (VIKOR returns 1 - Q).
    scores, ranking = result[0], result[1]
    return np.asarray(scores, dtype=float), np.asarray(ranking)
