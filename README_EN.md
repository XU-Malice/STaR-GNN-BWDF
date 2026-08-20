# STaR-GNN for Multi-DMA Water-Demand Forecasting

[中文](README.md) | [Documentation](docs/README.md) | [Full reproduction](docs/FULL_PIPELINE_CN.md) | [Method](docs/METHOD_CN.md) | [Results](docs/RESULTS_AND_ARTIFACTS_CN.md) | [Final figure plan](docs/MANUSCRIPT_FIGURES_FINAL_CN.md)

This repository provides the reproducible implementation of STaR-GNN for 24 h day-ahead and 168 h week-ahead hourly water-demand forecasting over ten district metered areas (DMAs). It includes split-aware preprocessing, a training-only Pearson functional graph, DCRNN/STGCN baselines, the SAS-Norm and FA-DPR factorial ablation, frozen common-46 test artifacts, and the tables/figures used by the Journal of Hydrology manuscript.

> **Manuscript-facing results use a publisher-compatible total convention.** Internal aggregate-demand MAE is retained for diagnostics but must not be mixed with manuscript total MAE.

## Factorial ablation

The formal ablation contains exactly four variants:

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

## Four-model ablation results

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

The 168 h publisher-compatible MAE point estimates of SAS-Norm-only and STaR-GNN differ by only `0.025755` (about `0.21%`). Because the 168 h forecast origins start 24 h apart and strongly overlap, the manuscript audit uses a seven-origin moving-block bootstrap rather than treating all 46 origins as independent. The confidence interval for the Full-minus-SAS mean difference includes zero. The repository therefore reports this as a small point-estimate difference, not as a stable performance gap.

## Main manuscript figures

- **Figure 1:** relative error reductions and NSE gains versus competing models;
- **Figure 2:** pure four-model factorial ablation over Day 1--Day 7; STGCN is excluded;
- **Figure 3:** DCRNN/STGCN/STaR-GNN ECDF robustness over the 46 common origins;
- **Figure 4:** DMA-level MAE improvements versus DCRNN/STGCN;
- **Figure 5:** a representative 168 h trajectory selected by a pre-specified median-error rule.

## Data and graph protocol

- hourly period: 2021-01-01 to 2023-03-05;
- train through 2022-12-15 23:00; test from 2022-12-16;
- 10 DMAs; 672 h history; 24 h and 168 h horizons; 24 h stride;
- training-only Pearson functional graph;
- negative correlations clipped to zero; zero diagonal; no threshold or Top-K;
- random-walk normalization; one static graph shared by both tasks;
- diffusion step `K=2`.

## Frozen manuscript setting

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

Shared model settings include hidden size 32, one recurrent layer, batch size 16, and early-stopping patience 15.

## Reproduce and verify

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .

bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

Regenerate the final manuscript tables and figures:

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820

python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures \
  --block-bootstrap-iterations 50000 \
  --block-bootstrap-length 7 \
  --block-bootstrap-seed 20260820
```

Expected final guards include:

```text
Factorial ablation model-set audit: PASS (4 models, no STGCN)
Publisher-compatible factorial cell audit: 30/32 PASS
Figure 2 factorial-model audit: PASS (4 models, no STGCN)
```

Reported GRU/LSTM/MSNet/MSCMNet values are literature results from Que et al. (2024); DCRNN/STGCN/STaR-GNN are re-evaluated on common-46. These two source categories are never described as having been retrained under identical code conditions.
