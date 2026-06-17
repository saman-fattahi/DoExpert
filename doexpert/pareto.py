"""doexpert.pareto
==================

Pareto-dominance utilities: non-dominance tests and Pareto-set extraction
supporting mixed minimisation/maximisation objectives and optional
constraint-violation filtering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["is_pareto_optimal", "is_pareto_efficient", "get_pareto"]

def is_pareto_optimal(costs, maximize_objectives=None):
    """
    Find Pareto-optimal solutions
    costs: 2D array where each row is a solution and each column is an objective
    maximize_objectives: List of boolean flags indicating which objectives to maximize
    Returns: Boolean array indicating which solutions are Pareto-optimal
    """
    try:
        costs = np.array(costs)
    except ValueError as e:
        # Handle inhomogeneous arrays by ensuring consistent shapes
        if "inhomogeneous" in str(e).lower() or "sequence" in str(e).lower():
            costs_list = []
            for cost in costs:
                cost_arr = np.array(cost)
                if cost_arr.ndim == 0:
                    cost_arr = np.array([cost_arr])
                elif cost_arr.ndim > 1:
                    cost_arr = cost_arr.flatten()
                costs_list.append(cost_arr)
            
            # Ensure all have same length
            if costs_list:
                max_len = max(len(c) for c in costs_list)
                padded_costs = []
                for c in costs_list:
                    if len(c) < max_len:
                        padded_c = np.pad(c, (0, max_len - len(c)), mode='edge')
                    else:
                        padded_c = c[:max_len]
                    padded_costs.append(padded_c)
                costs = np.array(padded_costs)
            else:
                return np.array([], dtype=bool)
        else:
            raise e
    
    if costs.ndim == 1:
        costs = costs.reshape(1, -1)
    
    # Handle maximize objectives by negating them for minimization comparison
    if maximize_objectives is not None:
        costs_adj = costs.copy()
        maximize_objectives = np.array(maximize_objectives)
        for i, maximize in enumerate(maximize_objectives):
            if maximize:
                costs_adj[:, i] = -costs_adj[:, i]
    else:
        costs_adj = costs
    
    n_solutions = costs_adj.shape[0]
    is_pareto = np.ones(n_solutions, dtype=bool)
    
    for i in range(n_solutions):
        # Check if solution i is dominated by any other solution
        # Solution i is dominated by j if j is better or equal in all objectives and strictly better in at least one
        for j in range(n_solutions):
            if i != j:
                # j dominates i if j <= i in all objectives AND j < i in at least one objective
                dominates = np.all(costs_adj[j] <= costs_adj[i]) and np.any(costs_adj[j] < costs_adj[i])
                if dominates:
                    is_pareto[i] = False
                    break
    
    return is_pareto

def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    costs = np.array(costs)
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        # Check if point i is dominated by any other point
        # Point i is dominated if there exists another point j such that:
        # all objectives of j are <= objectives of i AND at least one is strictly <
        for j in range(n_points):
            if i != j:
                # j dominates i if j <= i in all objectives AND j < i in at least one objective
                if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                    is_efficient[i] = False
                    break
    
    if return_mask:
        return is_efficient
    else:
        return np.where(is_efficient)[0]

def get_pareto(X, F, maximize_objectives=None, CV=None, return_indices=True):
    """
    Compute Pareto set for arbitrary arrays using the app's dominance logic.
    - X: array-like (n, d) of decision/control variables (optional, can be None)
    - F: array-like (n, m) of objective values in their original semantics
         (i.e., larger is better for 'maximize' objectives; smaller is better for 'minimize').
    - maximize_objectives: list/array of booleans length m indicating which objectives are to be maximized
    - CV: array-like (n,) of total constraint violation (<= 0 means feasible). If provided, only feasible
          solutions are considered for Pareto. If None, all are treated as feasible.
    Returns dict with:
      mask: boolean mask (n,) True where solution is Pareto non-dominated among feasibles
      indices: integer indices of the non-dominated solutions in the original arrays
      X_nd: X filtered to Pareto solutions (None if X is None)
      F_nd: F filtered to Pareto solutions
    """
    # Ensure arrays
    F = np.array(F)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    X_arr = np.array(X) if X is not None else None
    n = len(F)
    # Feasibility mask
    if CV is not None:
        CV = np.array(CV).reshape(-1)
        feasible = (CV <= 0)
    else:
        feasible = np.ones(n, dtype=bool)
    # Subselect feasible
    idx_map = np.where(feasible)[0]
    if len(idx_map) == 0:
        mask_full = np.zeros(n, dtype=bool)
        return {
            'mask': mask_full,
            'indices': np.array([], dtype=int),
            'X_nd': None if X_arr is None else X_arr[[]],
            'F_nd': F[[]]
        }
    F_feas = F[idx_map]
    # Compute Pareto mask on feasible set, respecting directions
    mask_feas = is_pareto_optimal(F_feas, maximize_objectives=maximize_objectives)
    nd_indices = idx_map[mask_feas]
    # Build full-length mask
    mask_full = np.zeros(n, dtype=bool)
    mask_full[nd_indices] = True
    return {
        'mask': mask_full,
        'indices': nd_indices,
        'X_nd': None if X_arr is None else X_arr[nd_indices],
        'F_nd': F[nd_indices]
    }
