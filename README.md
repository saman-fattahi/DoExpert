# DoExpert — Version 1.1

**DoExpert** is an interactive web application built with [Streamlit](https://streamlit.io/) for end-to-end experiment-driven process optimization. It covers the full workflow from experimental design and data analysis to surrogate modeling, forward prediction, multi-objective optimization, and decision-making.

---

## Overview

DoExpert is designed for engineers and researchers who need to:
- Plan structured experiments using Design of Experiments (DoE) techniques
- Analyze experimental results with statistical methods (ANOVA, main effects)
- Build accurate surrogate (machine learning) models from experimental data
- Predict process outputs based on new control variable inputs
- Optimize multiple conflicting objectives simultaneously
- Rank and select optimal solutions using a systematic decision-making method (MCDM)

---

## Key Features

### 1. Design of Experiments (DoE)
- Generate structured experimental designs (Full Factorial, Central Composite, Box-Behnken, Latin Hypercube, etc.)
- Define control variables (CVs) with their ranges and levels
- Export designs for lab or simulation use

### 2. Data Upload & Validation
- Upload experimental results in CSV/Excel format
- Automatic data validation and consistency checks
- Preview and inspect loaded datasets

### 3. Variable Analysis
- **ANOVA & Statistics:** Assess the statistical significance of each control variable on each experimental variable (EV) using Analysis of Variance (ANOVA)
- **Main Effects Analysis:** Visualize the directional influence of each CV on EVs through main effect plots

### 4. Surrogate Modeling
- Train multiple machine learning models simultaneously on experimental data
- Supported models (default hyperparameters):
  - Random Forest
  - Gradient Boosting
  - Extra Trees
  - Support Vector Machine (SVM)
  - Neural Network (MLP)
  - Gaussian Process Regression
  - Ridge Regression
  - Lasso Regression
  - Linear Regression
  - XGBoost (if installed)
- Automatic model comparison with cross-validation metrics (R², RMSE, MAE)
- Best model per EV is selected automatically for downstream tasks

### 5. Forward Predictive Analysis
- Predict experimental variable (EV) values from new control variable (CV) inputs
- Uses the best-performing surrogate model per EV
- Interactive input sliders and instant predictions

### 6. Multi-Objective Optimization
- Optimize multiple EVs simultaneously with user-defined objectives (minimize / maximize)
- Powered by [PyMOO](https://pymoo.org/) — available algorithms:
  - **NSGA-II** (Non-dominated Sorting Genetic Algorithm II)
  - **NSGA-III** (Reference-point based NSGA-III)
- Control variable bounds automatically derived from experimental data range
- Pareto front visualization and result export

### 7. MCDM — Multi-Criteria Decision Making
- Rank Pareto-optimal solutions using a **selectable MCDM method**:
  **TOPSIS**, **VIKOR**, **PROMETHEE II**, **SAW**, **WASPAS**, **Fuzzy TOPSIS**
- Cross-method comparison: top-ranked solution per method and Spearman
  rank-correlation matrix to assess ranking stability
- User-defined weights per objective (automatically normalized)
- Ranked solution table with visual indicators for best trade-off solutions

---

## Repository structure

```
doexpert/                  # Reusable analytical engine (importable package)
├── mcdm.py                # TOPSIS, VIKOR, PROMETHEE II, SAW, WASPAS, Fuzzy TOPSIS
├── pareto.py              # Pareto-dominance utilities
├── pareto_selection.py    # ASF, pseudo-weights, knee detection
├── metrics.py             # Hypervolume, spacing, extent
└── doe/                   # Taguchi orthogonal-array engine
streamlit_app_doexpert.py  # Streamlit GUI (presentation + workflow layer)
tests/                     # Unit tests (pytest), run in CI on Python 3.9–3.12
examples/                  # Flat grinding case-study dataset (L8 design)
docs/                      # User guide and API reference
```

The Streamlit application provides the graphical workflow; the
`doexpert` package can be used independently in scripts and notebooks
(see [docs/API_REFERENCE.md](docs/API_REFERENCE.md)).

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

**Option A — pip (recommended):**

```bash
pip install doexpert
doexpert          # launches the web app at http://localhost:8501
```

**Option B — from source:**

```bash
git clone https://github.com/saman-fattahi/DoExpert.git
cd DoExpert

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run streamlit_app_doexpert.py
```

Open your browser at **http://localhost:8501**

---

## Docker (Optional)

```bash
# Build the Docker image
docker build -t doexpert:latest .

# Run the container
docker run -p 8501:8501 doexpert:latest
```

Open your browser at **http://localhost:8501**

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

| Package | Purpose |
|---|---|
| streamlit | Web application framework |
| pandas / numpy | Data handling |
| scikit-learn | Surrogate modeling |
| pymoo | Multi-objective optimization |
| scipy | Statistical analysis (ANOVA) |
| plotly | Interactive charts |
| xgboost *(optional)* | Additional surrogate model |

---

## Typical Workflow

```
1. Define CVs and generate DoE design
      ↓
2. Upload experimental results
      ↓
3. Analyze variable significance (ANOVA)
      ↓
4. Train surrogate models
      ↓
5. Predict outputs for new CV settings
      ↓
6. Run multi-objective optimization (NSGA-II / NSGA-III)
      ↓
7. Rank Pareto solutions with TOPSIS (MCDM)
      ↓
8. Export best solution
```

---

## Example dataset

The flat grinding case study from the accompanying SoftwareX paper is
provided in [`examples/flat_grinding_L8.csv`](examples/flat_grinding_L8.csv)
together with reproduction instructions ([`examples/README.md`](examples/README.md)).

---

## Documentation

- [User Guide](docs/USER_GUIDE.md) — configurable options per workflow
  stage, iterative use, degree of automation
- [API Reference](docs/API_REFERENCE.md) — developer documentation for
  the `doexpert` package

---

## Testing

```bash
pip install -e .[dev]
pytest tests/
```

Unit tests cover the MCDM methods (ranking invariants, weight
sensitivity), Pareto utilities, front-quality metrics (hypervolume
reference values), and the orthogonal-array engine (balance and
strength-2 orthogonality). CI runs the suite on Python 3.9–3.12.

---

## Developer Contact

**Dr. Saman Fattahi**  
Team Leader – Data-Driven Manufacturing (DDM) Group
Postdoctoral Researcher
Furtwangen University
KSF – Institute for Advanced Manufacturing

📧 Saman.Fattahi(at)hs-furtwangen.de  
📞 +49 7461 1502 6736

---

## License

MIT License — see [LICENSE](LICENSE) for details.
