# DoExpert API Reference

Developer documentation for the `doexpert` analytical engine package
(v1.1). All modules are importable independently of the Streamlit GUI.

```python
import doexpert
print(doexpert.__version__)
```

---

## doexpert.mcdm

Multi-criteria decision-making methods. All methods share the
interface:

```python
scores, ranking, *diagnostics = method(decision_matrix, weights, criteria_types, **kwargs)
```

- `decision_matrix` — `ndarray (n_alternatives, n_criteria)`
- `weights` — `ndarray (n_criteria,)`, non-negative
- `criteria_types` — list of `'max'`/`'min'` (or `1`/`-1`)
- `scores` — higher is better for every method (VIKOR returns `1 − Q`;
  the raw `Q`, `S`, `R` arrays are returned as diagnostics)
- `ranking` — indices, best to worst

### Functions

| Function | Extra parameters | Diagnostics returned |
|---|---|---|
| `topsis(M, w, t)` | — | — |
| `vikor(M, w, t, v=0.5)` | `v`: group-utility weight | `S`, `R`, `Q` |
| `promethee_ii(M, w, t, preference_functions=None)` | per-criterion preference functions | `phi_plus`, `phi_minus`, `phi_net` |
| `saw(M, w, t)` | — | — |
| `waspas(M, w, t, lambda_param=0.5)` | `lambda_param`: WSM/WPM mix | `wsm`, `wpm` |
| `fuzzy_topsis(M, w, t, alpha=0.5)` | `alpha`: fuzzification spread | — |

### Dispatcher

```python
from doexpert.mcdm import rank_solutions, AVAILABLE_METHODS

scores, ranking = rank_solutions(M, w, ['min', 'min', 'max'],
                                 method="VIKOR", v=0.5)
```

`rank_solutions` normalizes the weight vector, validates the method
name against `AVAILABLE_METHODS`, and guarantees the higher-is-better
score convention, enabling direct cross-method comparison.

### Example — ranking a Pareto front with two methods

```python
import numpy as np
from doexpert.mcdm import rank_solutions

F = np.array([[0.30, 12.1, 5.5],
              [0.42,  9.8, 7.0],
              [0.55,  8.0, 8.2]])          # Ra, Fn, MRR
w = np.array([0.5, 0.3, 0.2])
t = ['min', 'min', 'max']

for m in ("TOPSIS", "VIKOR", "PROMETHEE II"):
    s, r = rank_solutions(F, w, t, method=m)
    print(m, "best:", r[0])
```

---

## doexpert.pareto

| Function | Description |
|---|---|
| `is_pareto_efficient(costs, return_mask=True)` | Non-dominance mask/indices for a minimization cost matrix |
| `is_pareto_optimal(costs, maximize_objectives=None)` | Non-dominance with per-objective min/max flags |
| `get_pareto(X, F, maximize_objectives=None, CV=None, return_indices=True)` | Pareto-set extraction with optional constraint-violation filtering; returns `dict(mask, indices, X_nd, F_nd)` |

---

## doexpert.pareto_selection

Preference-based single-solution selection on a Pareto front `F`
(minimization convention). pymoo is used when installed; pure-NumPy
fallbacks are included.

| Function | Description |
|---|---|
| `compromise_programming_asf(F, weights)` | Augmented scalarising function (pymoo `ASF`) |
| `manual_asf(F, weights)` | NumPy ASF fallback |
| `pseudo_weights_method(F, preferred_weights)` | pymoo pseudo-weights selection |
| `manual_pseudo_weights(F, preferred_weights)` | NumPy fallback |
| `high_tradeoff_points(F)` | pymoo high-trade-off (knee) detection |
| `manual_knee_detection(F)` / `manual_tradeoff_detection(F)` | NumPy knee detection fallbacks |

---

## doexpert.metrics

Pareto-front quality indicators.

| Function | Description |
|---|---|
| `calculate_hypervolume(front, reference_point=None, ideal_point=None, normalize=False)` | Exact (pymoo), 2-D exact, or Monte-Carlo hypervolume |
| `calculate_spacing_metric(front)` | Schott spacing (0 = perfectly uniform) |
| `calculate_extent_metric(front)` | Mean per-objective range (spread) |

---

## doexpert.doe

Taguchi orthogonal-array engine (`doexpert.doe.orthogonal_arrays`).

| Function | Description |
|---|---|
| `list_catalog()` | All catalogued arrays (`OAInfo` objects, L4–L81 incl. mixed-level) |
| `get_oa_info(label)` | Look up one array, e.g. `"L8(2)"` |
| `auto_select_oa(levels_vector)` | Smallest valid array accommodating the factor levels, e.g. `[2,2,2,2,2] → L8(2)` |
| `find_compatible_oas(levels_vector)` | All compatible arrays |
| `search_oas_by_criteria(min_factors, max_runs, ...)` | Filtered catalogue search |
| `generate_basic_oa_matrix(oa_info)` | Raw 0-based design matrix |
| `build_design(oa_info, factor_names, factor_levels)` / `generate_design(...)` | `pandas.DataFrame` design table with real factor levels |

### Example — regenerate the paper's L8 design

```python
from doexpert.doe import auto_select_oa, build_design

oa = auto_select_oa([2, 2, 2, 2, 2])
design = build_design(
    oa,
    ["vc", "ae", "vw", "qd", "Ud"],
    [["20", "30"], ["50", "200"], ["250", "1000"], ["0.7", "0.9"], ["2", "4"]],
)
```

---

## Testing

The analytical modules are covered by the unit-test suite in `tests/`
(dominance properties, ranking invariants, OA balance and strength-2
orthogonality, hypervolume reference values). Run with:

```bash
pytest tests/ --cov=doexpert
```

Continuous integration executes the suite on Python 3.9–3.12
(`.github/workflows/tests.yml`).
