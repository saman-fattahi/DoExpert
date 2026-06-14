# Example dataset — flat grinding case study (L8 Taguchi design)

`flat_grinding_L8.csv` reproduces the experimental setup of the flat
grinding case study presented in the accompanying SoftwareX paper. The
design matrix was generated with DoExpert's own orthogonal-array engine
(`doexpert.doe`) using the L8 two-level Taguchi array and the variable
levels reported in Table 1 of the paper:

| Variable | Symbol | Unit | Levels |
|---|---|---|---|
| Grinding wheel speed | v_c | m/s | 20, 30 |
| Depth of cut | a_e | µm | 50, 200 |
| Workpiece feed rate | v_w | mm/min | 250, 1000 |
| Dressing speed ratio | q_d | – | 0.7, 0.9 |
| Overlap ratio | U_d | – | 2, 4 |

Response columns (evaluation variables):

| Variable | Symbol | Unit |
|---|---|---|
| Surface roughness | Ra | µm |
| Normal grinding force | F_n | N |
| Material removal rate | MRR | mm³/(mm·min) |

> **NOTE TO MAINTAINERS:** the response columns (`Ra_um`, `Fn_N`,
> `MRR_mm3_per_mm_min`) must be populated with the measured values used
> in the paper before release, so that readers can reproduce the
> reported analysis, surrogate models, optimization, and ranking.

## How to reproduce the paper workflow

1. Launch the app: `streamlit run streamlit_app_doexpert.py`
2. In **Problem Definition**, define the five control variables and
   three evaluation variables with the levels above.
3. In **Experimental Design**, choose *Orthogonal Array (Taguchi)* and
   generate the L8 design — or skip design generation and directly
   upload `flat_grinding_L8.csv` in the **Data Upload** stage.
4. Run **Variable Analysis** (ANOVA on Ra), train surrogate models,
   perform multi-objective optimization (NSGA-II / NSGA-III; minimize
   Ra and F_n, maximize MRR), and rank Pareto solutions in the
   **MCDM Analysis** stage.

## Programmatic regeneration of the design matrix

```python
from doexpert.doe import auto_select_oa, build_design

names = ["vc_m_per_s", "ae_um", "vw_mm_per_min", "qd", "Ud"]
levels = [["20", "30"], ["50", "200"], ["250", "1000"],
          ["0.7", "0.9"], ["2", "4"]]
oa = auto_select_oa([2, 2, 2, 2, 2])   # -> L8(2)
design = build_design(oa, names, levels)
```
