# DoExpert User Guide

This guide describes what DoExpert can and cannot do at each workflow
stage, the degree of user interaction required, and all user-configurable
options exposed in version 1.1.

## 1. Workflow overview

DoExpert organizes a study as a sequence of stages sharing one internal
state (Streamlit session state):

```
Problem definition → Experimental design → Data upload/validation
→ Variable analysis → Surrogate modeling → Predictive analysis
→ Multi-objective optimization → MCDM ranking → Export
```

Stages can be revisited at any time. Re-training a surrogate model or
re-running the optimization does **not** require restarting the study:
downstream stages automatically operate on the most recent upstream
results held in the shared state (see §9, Iterative use).

## 2. Problem definition

| Item | User-configurable? | Details |
|---|---|---|
| Number of control variables (CVs) | Yes | No hard upper limit; designs constrain practical counts |
| CV type | Yes | Numeric (continuous within bounds) or categorical |
| CV bounds / levels | Yes | Lower/upper bounds and discrete levels per CV |
| Number of evaluation variables (EVs) | Yes | Multiple responses supported |
| EV objective direction | Yes (later, in MOO stage) | minimize / maximize per EV |

**Continuous vs. discrete:** CVs are defined with continuous admissible
ranges. For DoE generation, the user discretizes each CV into levels
(e.g., 2 levels for screening, 3+ for response-surface designs). In the
optimization stage, CVs are treated as continuous within their bounds —
the optimizer is not restricted to the DoE levels. Categorical CVs are
mapped to discrete levels throughout.

## 3. Experimental design

Available design strategies (selectable in the GUI):

- Full Factorial
- Orthogonal Array (Taguchi) — automatic array selection from a
  validated catalogue (L4–L81, including mixed-level arrays)
- Fractional Factorial (2-level)
- Box-Behnken Design (RSM)
- Central Composite Design (CCD)
- Latin Hypercube Sampling
- D-Optimal Design
- Plackett-Burman (Screening)
- Sobol Sequence (Quasi-Monte Carlo)
- Halton Sequence (Low-Discrepancy)

The software also provides design *recommendations* based on the number
of factors, levels, study goal (screening / modeling / optimization),
and experimental budget. Designs are exported as editable tables
(CSV/Excel); responses can be typed directly into the table or imported
later.

**Not configurable:** the internal construction algorithms of each
design family (e.g., the orthogonal-array generators) are fixed;
users select among designs, levels, and run counts, but cannot inject
custom design generators through the GUI. Programmatic users can call
`doexpert.doe` directly.

## 4. Data upload and validation

- CSV/Excel import with automatic consistency checks against the
  defined CVs/EVs.
- Cleaning utilities: removal of empty rows/columns, EV-aware filtering
  of incomplete runs, type coercion.
- New or corrected data can be re-imported at any time; downstream
  stages then re-run on the updated dataset without redefining the
  problem (see §9).

## 5. Variable analysis

| Tool | Configurable options |
|---|---|
| Descriptive statistics | Response selection |
| ANOVA significance testing | Response selection, main effects and two-way interactions, significance level |
| Main-effects analysis | Response and factor selection |

**Not configurable:** the statistical engines themselves (ANOVA
implementation) are fixed; alternative analysis backends cannot be
swapped through the GUI.

## 6. Surrogate modeling

Supported regression models (one final model selected per EV):

Random Forest, Gradient Boosting, Extra Trees, Support Vector Machine,
Neural Network (MLP), Gaussian Process Regression, Ridge, Lasso,
Linear Regression, and XGBoost (if installed).

| Option | User-configurable? |
|---|---|
| Models included in comparison | Yes (any subset) |
| Hyperparameters | Yes — manual per-model interface, or automated tuning |
| Auto-tuning budget | Yes — fast / balanced / thorough (controls search space and iterations) |
| Train/test split & cross-validation metrics | Reported automatically (R², RMSE, MAE) |
| Final model per EV | Yes — automatic best-model proposal, manual override |

Models can be re-trained at any time; the new final models immediately
replace the previous ones for prediction and optimization.

## 7. Multi-objective optimization

Available algorithms (subject to the installed pymoo version):

NSGA-II, NSGA-III, MOEA/D, GDE3/MODE, AGE-MOEA, OMOPSO, SPEA2, RVEA,
SMS-EMOA.

| Option | User-configurable? |
|---|---|
| Algorithm selection (incl. multiple in one run) | Yes |
| Objective direction per EV (min/max) | Yes |
| Population size, number of generations | Yes (with adaptive recommendations) |
| Algorithm-specific parameters (e.g., F/CR for GDE3, decomposition for MOEA/D) | Yes |
| CV bounds | Derived from problem definition / data range, editable |
| Constraints on EVs | Yes |
| Pareto-quality indicators | Reported: hypervolume, spacing, extent |

## 8. MCDM ranking and selection

Since version 1.1, the ranking method is selectable. Implemented
methods (module `doexpert.mcdm`):

- TOPSIS
- VIKOR (compromise parameter *v* configurable)
- PROMETHEE II
- SAW (Simple Additive Weighting)
- WASPAS (λ configurable)
- Fuzzy TOPSIS (α configurable)

| Option | User-configurable? |
|---|---|
| Ranking method | Yes |
| Objective weights | Yes (free weights, normalized internally) |
| Cross-method comparison | Yes — top-ranked solution per method and Spearman rank-correlation matrix |

The cross-method comparison addresses the known sensitivity of ranking
outcomes to the choice of MCDM method: users can verify whether the
recommended operating point is stable across methods before adopting it.

## 9. Iterative use and feedback between stages

DoExpert is sequential in presentation but not one-directional in use.
Because all stages share one persistent study state:

- New or cleaned data can be imported mid-study; variable analysis,
  modeling, and optimization re-run on the updated dataset.
- Surrogate models can be re-trained or replaced after inspecting
  optimization results, and the optimization re-executed against the
  new models — supporting the common "model → optimize → inspect →
  re-model" refinement loop.
- Weights and ranking methods can be changed freely on a fixed Pareto
  set to study ranking sensitivity.

What is **not** automated in the current version: closed-loop adaptive
sampling (automatic suggestion of new experiments from surrogate
uncertainty) and automatic model re-selection triggered by optimization
diagnostics. These remain user-initiated steps; the modular layout
(`doexpert` package) is designed so such modules can be added without
modifying existing stages.

## 10. Degree of user interaction

| Stage | Interaction level | Why not fully automatic |
|---|---|---|
| Problem definition | Required | Encodes domain knowledge (variables, bounds, units) |
| Experimental design | Guided choice | Depends on budget and study goal; recommendations provided |
| Data entry/import | Required | Physical experiments are external to the software |
| Variable analysis | Optional inspection | Fully automatic defaults; user interprets |
| Surrogate modeling | Optional | Automatic comparison + best-model proposal; manual override possible |
| Predictive analysis | Optional | What-if exploration tool |
| Optimization | Configurable, sensible defaults | Adaptive parameter recommendations provided |
| MCDM ranking | Required (weights) | Preferences are inherently subjective |

## 11. Running the test suite

```bash
pip install -e .[dev]
pytest tests/
```
