# STaR-GNN for Multi-DMA Water-Demand Forecasting

[中文](README.md) | [Documentation](docs/README.md) | [Method](docs/METHOD_CN.md) | [Final experiment design](docs/EXPERIMENT_DESIGN_FINAL_CN.md) | [Results](docs/RESULTS_AND_ARTIFACTS_CN.md) | [Submission plotting guide](docs/PLOTTING_CN.md)

This repository provides the reproducible implementation of STaR-GNN for 24 h day-ahead and 168 h week-ahead hourly water-demand forecasting over ten district metered areas (DMAs). It includes split-aware preprocessing, a training-only Pearson functional graph, DCRNN/STGCN baselines, SAS-Norm and FA-DPR factorial ablations, frozen common-46 test artifacts, and the submission tables/figures used by the Journal of Hydrology manuscript.

> **Manuscript-facing results use a publisher-compatible total convention.** Internal aggregate-demand MAE is retained for diagnostics but is never mixed with the manuscript total MAE.

## Formal factorial ablation

| Model | Internal variant | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |

**STGCN is an independent graph baseline and is not an ablation variant.**

## Manuscript metric convention

Aligned with the total convention reported in the Que et al. (2024) supplementary tables:

- total MAE = sum of DMA A--J MAEs;
- total MAPE/RMSE/NSE = metrics on the hourly aggregate-demand series;
- primary test protocol = `common_46`;
- test-time teacher forcing = 0;
- test targets are not used for training, early stopping, or component selection.

Final STaR-GNN manuscript-facing results:

| Horizon | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| 24 h | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | **12.234** | **2.014** | **6.161** | **0.976** |

Internal aggregate-demand MAE is `4.360841 / 4.919812` and is used only for aggregate-demand diagnostics.

## Four-model ablation

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| 24 h | DCRNN + SAS-Norm | 10.468 | 2.010 | 6.134 | 0.976 |
| 24 h | DCRNN + FA-DPR | 11.238 | 1.945 | 6.079 | 0.977 |
| 24 h | **STaR-GNN** | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| 168 h | **DCRNN + SAS-Norm** | **12.208** | 2.102 | 6.468 | 0.974 |
| 168 h | DCRNN + FA-DPR | 14.086 | 3.278 | 9.332 | 0.945 |
| 168 h | STaR-GNN | 12.234 | **2.014** | **6.161** | **0.976** |

The 168 h publisher-compatible MAE point estimates of SAS-Norm-only and STaR-GNN differ by only `0.025755` (about `0.21%`). Because adjacent week-ahead forecast origins strongly overlap, the manuscript uses an ordered seven-origin moving-block bootstrap to bound interpretation. The mean-difference interval includes zero, so the repository does not describe the point-estimate difference as a stable performance gap.

## Journal of Hydrology submission evidence chain

The final Results section no longer spreads the story over five separate main-result figures. The canonical submission structure is **two main tables + three main result figures**:

```text
Main Table 1
Overall predictive performance
        ↓
Main Table 2 + Main Fig. 1
Factorial ablation + lead-time stability
        ↓
Main Fig. 2
Temporal + spatial robustness
        ↓
Main Fig. 3
Population-to-instance week-ahead dynamics
```

### Main Figure 1 — Ablation mechanism and lead-time stability

- absolute Day-1--Day-7 publisher-compatible MAE for the four factorial variants with seven-origin moving-block 95% CIs;
- Day-1-relative degradation, with Day-7 changes of approximately `+38.25%`, `+11.93%`, `+2.64%`, and `+1.70%` for DCRNN, FA-DPR, SAS-Norm, and STaR-GNN, respectively.

### Main Figure 2 — Temporal and spatial robustness

- paired MAE improvements across the 46 common forecast origins, with win counts `45/46`, `45/46`, `46/46`, and `40/46` against DCRNN/STGCN at 24/168 h;
- DMA-level robustness over 10 DMAs × 2 horizons × 2 graph baselines: all 40 comparisons are positive, ranging from about `1.26%` to `61.20%`.

### Main Figure 3 — Week-ahead demand dynamics

- population-level diurnal aggregate-demand error profile over 46 origins × 7 forecast days;
- a representative 168 h trajectory selected with a pre-specified median-error rule;
- the corresponding hourly aggregate-demand absolute error. Demand units are `L s⁻¹`.

Supplementary material:

- **Table S1**: detailed DMA A--J metrics;
- **Fig. S1**: relative improvement over all comparison models;
- **Fig. S2**: per-origin ECDF.

See [`docs/EXPERIMENT_DESIGN_FINAL_CN.md`](docs/EXPERIMENT_DESIGN_FINAL_CN.md) for the complete claim-driven design.

## Submission artifact paths

```text
paper/tables/submission/
  table1_overall_performance.md
  table2_factorial_ablation.md
  tableS1_dma_metrics.md

paper/figures/submission/
  main_fig1_ablation_leadtime.{pdf,svg,png}
  main_fig2_temporal_spatial_robustness.{pdf,svg,png}
  main_fig3_week_ahead_dynamics.{pdf,svg,png}

paper/figures/supplementary/
  supp_figS1_relative_improvement.{pdf,svg,png}
  supp_figS2_origin_ecdf.{pdf,svg,png}
```

STaR-GNN is the visual hero method (`#0F4D92` deep blue); DCRNN/STGCN use gray baselines, while SAS-Norm and FA-DPR use restrained variant colors. The shared style is centralized in `scripts/reproduce/manuscript_plot_style.py`.

## Data, graph, and frozen setting

- hourly period: 2021-01-01 to 2023-03-05;
- train through 2022-12-15 23:00; test from 2022-12-16;
- 10 DMAs; 672 h history; 24 h and 168 h horizons; 24 h stride;
- training-only Pearson functional graph;
- negative correlations clipped to zero; zero diagonal; no threshold or Top-K;
- random-walk normalization; one static graph shared by both tasks; diffusion step `K=2`;
- hidden size 32; one recurrent layer; batch size 16; early-stopping patience 15;
- learning rate `3e-4`; weight decay `0`; curriculum decay `500`; state-loss weight `0.03`; seed `0`.

## Reproduce and verify

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .

bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

Regenerate only the submission tables and figures:

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/render_submission_tables.py

python scripts/reproduce/render_submission_figures.py \
  --release results/paper/frozen_v1 \
  --block-length 7 \
  --bootstrap-iterations 50000 \
  --bootstrap-seed 20260821
```

Final figures are exported as vector PDF, editable SVG, and 300 dpi PNG previews.

Reported GRU/LSTM/MSNet/MSCMNet values are literature results from Que et al. (2024); DCRNN/STGCN/STaR-GNN are re-evaluated on common-46. These source categories are never described as having been retrained under identical code conditions.

Legacy `paper/figures/manuscript_fig1...5`, `paper/figures/test_*`, and old captions are retained for historical reproduction/diagnostics but are no longer the authoritative submission artifacts.
