# STaR-GNN for Multi-DMA Water-Demand Forecasting

[中文](README.md) | [Documentation index](docs/README.md) | [Full reproduction guide](docs/FULL_PIPELINE_CN.md) | [Method](docs/METHOD_CN.md) | [Results and artifacts](docs/RESULTS_AND_ARTIFACTS_CN.md) | [Plotting guide](docs/PLOTTING_CN.md)

This repository provides the reproducible implementation of STaR-GNN for 24 h day-ahead and 168 h week-ahead hourly water-demand forecasting over ten district metered areas (DMAs). It includes split-aware preprocessing, a training-only Pearson functional graph, DCRNN/STGCN graph baselines, the SAS-Norm and FA-DPR ablations, frozen common-46 test artifacts, and the tables/figures used by the Journal of Hydrology manuscript.

> **Manuscript-facing comparisons use the publisher-compatible metric convention.** Internal aggregate-demand MAE values such as 4.360841/4.919812 are retained only for diagnostics and must not be mixed with the manuscript total MAE. See [`paper/tables/literature/METRIC_CONVENTIONS.md`](paper/tables/literature/METRIC_CONVENTIONS.md).

## 1. Manuscript metric convention

Aligned with the total convention reported by Que et al. (2024):

- **total MAE** = sum of the DMA A--J MAEs;
- **total MAPE/RMSE/NSE** = metrics computed on the hourly aggregate-demand series;
- primary evaluation = `common_46`, the 46 test origins shared by both forecast tasks;
- test-time teacher forcing is disabled and future demand is never used for model selection.

Final STaR-GNN manuscript-facing results are:

| Horizon | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| 24 h | **9.424199** | **1.804574** | **5.534656** | **0.980679** |
| 168 h | **12.233590** | **2.013774** | **6.160881** | **0.976176** |

The complete nine-model comparison is in [`table_literature_comparison_common46.md`](paper/tables/literature/table_literature_comparison_common46.md). GRU, LSTM, MSNet, and MSCMNet variants are reported values from Que et al. (2024); DCRNN, STGCN, and STaR-GNN are re-evaluated on the current common-46 protocol. They should not be described as all being retrained under one identical codebase.

## 2. Ablation naming and guardrails

| Manuscript name | Internal key | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | no | no |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | yes | no |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | no | yes |
| STaR-GNN | `full` / `Full` | yes | yes |

The publisher-compatible ablation audit is **30/32**. Two real exceptions are intentionally retained:

1. FA-DPR has a slightly worse 168 h MAPE than DCRNN (3.277716% vs. 3.248413%);
2. SAS-Norm-only has a marginally lower 168 h sum-of-DMA MAE than full STaR-GNN (12.207835 vs. 12.233590, about 0.21%).

Therefore the full model is best on all four metrics at 24 h; at 168 h it is best on MAPE/RMSE/NSE, while SAS-Norm-only is marginally lower on publisher-compatible MAE. See [`table_ablation_common46.md`](paper/tables/literature/table_ablation_common46.md).

## 3. Final manuscript tables and figures

The Results evidence chain is fixed as:

1. **Overall predictive accuracy** — Table 1 + Figure 1;
2. **Ablation and component contributions** — Table 2 + Figure 2;
3. **Robustness across forecast origins** — Figure 3;
4. **Spatial consistency across DMAs** — Table 3 + Figure 4;
5. **Representative weekly forecasting behavior** — Figure 5.

Main tables:

- `paper/tables/literature/table_literature_comparison_common46.*`
- `paper/tables/literature/table_ablation_common46.*`
- `paper/tables/literature/table_star_gnn_dma_common46.*`

Main figures:

- `paper/figures/manuscript_fig1_relative_improvement.*`
- `paper/figures/manuscript_fig2_day1_day7_publisher_mae.*`
- `paper/figures/manuscript_fig3_origin_ecdf.*`
- `paper/figures/manuscript_fig4_dma_mae_improvement.*`
- `paper/figures/manuscript_fig5_representative_168h_trajectory.*`

Final figure design is documented in [`docs/MANUSCRIPT_FIGURES_FINAL_CN.md`](docs/MANUSCRIPT_FIGURES_FINAL_CN.md); captions are in [`paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`](paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md).

## 4. Empirical checks captured by the figure audit

- Day-7 vs Day-1 publisher-compatible MAE change for the 168 h task: DCRNN +38.25%, FA-DPR +11.93%, SAS-Norm +2.64%, STaR-GNN +1.70%.
- Per-origin STaR-GNN win rate vs DCRNN: 45/46 at 24 h and 46/46 at 168 h.
- Per-origin win rate vs STGCN: 45/46 at 24 h and 40/46 at 168 h.
- DMA-level MAE improvement is positive in all 40 comparisons (10 DMAs × 2 horizons × 2 graph baselines), with heterogeneous magnitudes.
- SAS-Norm is the primary contributor to low long-horizon MAE drift; the manuscript should not claim that the full model strictly dominates SAS-Norm-only on 168 h MAE.

The underlying CSV/JSON evidence is stored in `paper/tables/manuscript/`.

## 5. Installation and frozen-result verification

Recorded environment: Python 3.11.15, PyTorch 2.9.1+cu128, CUDA 12.8, NumPy 2.4.6, pandas 3.0.5.

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .

bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

The GitHub Release contains frozen checkpoints, predictions, metrics, and checksums. Licensed BWDF raw/processed data are not redistributed; missing processed data and the training-period graph can be rebuilt by the pinned pipeline without retraining the models.

To train the full paper experiment from raw data:

```bash
bash scripts/reproduce/train_from_scratch.sh \
  --device auto \
  --evaluation-device cpu \
  --seeds 0
```

See [`docs/FULL_PIPELINE_CN.md`](docs/FULL_PIPELINE_CN.md) for the full data/graph/training/evaluation workflow.

## 6. Regenerate the final manuscript figures

Stage 1 builds the manuscript audit tables and base Figures 1--5 from the frozen predictions:

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

Stage 2 generates the **final** Figure 2 and Figure 3 layouts from the audited data:

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures
```

Stage 2 intentionally overwrites the Stage-1 versions of Figure 2 and Figure 3. See [`docs/PLOTTING_CN.md`](docs/PLOTTING_CN.md) for prerequisites, outputs, audit files, reproducibility checks, and troubleshooting.

## 7. Data and graph protocol

- Dataset period: 2021-01-01 to 2023-03-05, hourly.
- Training through 2022-12-15 23:00; test begins 2022-12-16.
- Input history: 672 h; forecast horizons: 24 h and 168 h.
- Graph: positive Pearson correlations computed from training demand only, zero diagonal, no threshold/Top-K, random-walk normalization.
- The same fixed functional graph is shared by both horizons.
- Primary test protocol: common-46.
- Hyperparameters are determined from validation only.

## 8. Repository layout

```text
configs/                 frozen data/graph/model/paper configuration
src/dma_wdf/             core data, graph, model, training, evaluation code
scripts/reproduce/       reproduction, frozen verification, manuscript tables/figures
paper/tables/literature/ manuscript Tables 1--3 and metric conventions
paper/tables/manuscript/ Figure 1--5 audit CSV/JSON artifacts
paper/figures/           manuscript and supplementary PNG/PDF figures
paper/captions/          final figure captions
docs/                    method, results, reproduction, plotting, release guides
```

Legacy `test_overall_*`, `test_ablation_*`, `test_star_gnn_dma_metrics.*`, and aggregate-demand Day-1--Day-7 plots are retained as supplementary/internal diagnostics; they are not the main manuscript evidence.

## 9. Tests, data, and citation

```bash
bash scripts/reproduce/smoke_test.sh
python -m pytest tests -q
```

See [`data/README.md`](data/README.md) for BWDF access, [`docs/RELEASE_CN.md`](docs/RELEASE_CN.md) for release/clean-room guidance, and `CITATION.cff` for citation metadata.