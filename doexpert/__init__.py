"""DoExpert — analytical engine package.

Reusable, test-covered analytical modules underlying the DoExpert
platform:

- :mod:`doexpert.mcdm` — multi-criteria decision-making methods
  (TOPSIS, VIKOR, PROMETHEE II, SAW, WASPAS, Fuzzy TOPSIS) with a
  unified ``rank_solutions`` dispatcher.
- :mod:`doexpert.pareto` — Pareto-dominance tests and Pareto-set
  extraction.
- :mod:`doexpert.pareto_selection` — preference-based selection on a
  Pareto front (ASF, pseudo-weights, knee detection).
- :mod:`doexpert.metrics` — Pareto-front quality indicators
  (hypervolume, spacing, extent).
- :mod:`doexpert.doe` — design-of-experiments utilities, including the
  Taguchi orthogonal-array engine.

The Streamlit application (``streamlit_app_doexpert.py``) provides the
graphical workflow on top of these modules; the modules themselves can
be used independently in scripts and notebooks.
"""

from doexpert import mcdm, metrics, pareto, pareto_selection  # noqa: F401
from doexpert.mcdm import AVAILABLE_METHODS, rank_solutions  # noqa: F401

__version__ = "1.1.0"
