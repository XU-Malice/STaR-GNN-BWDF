# Legacy five-figure captions (superseded)

> **LEGACY / SUPERSEDED.** These captions correspond to the previous five-main-figure layout (`manuscript_fig1...5`). They are retained for historical reproduction only and are **not** the current Journal of Hydrology submission captions.
>
> Current authoritative captions: [`SUBMISSION_RESULT_FIGURE_CAPTIONS.md`](SUBMISSION_RESULT_FIGURE_CAPTIONS.md).
>
> Current experiment/figure architecture: [`../../docs/EXPERIMENT_DESIGN_FINAL_CN.md`](../../docs/EXPERIMENT_DESIGN_FINAL_CN.md).

## Previous Figure 1

**Relative performance improvement of STaR-GNN over competing models for the 24 h and 168 h forecasting tasks.** Panel (a) reports the relative reductions in MAE, MAPE and RMSE, calculated as \((E_{baseline}-E_{STaR})/E_{baseline}\times100\%\); panel (b) reports the absolute NSE gain \(NSE_{STaR}-NSE_{baseline}\). Positive values indicate an improvement of STaR-GNN. GRU, LSTM, MSNet and the MSCMNet variants (marked with †) are reported results from Que et al. (2024), whereas DCRNN, STGCN and STaR-GNN are evaluated on the common 46-sequence test period. The publisher-compatible total convention is used for cross-model comparison.

## Previous Figure 2

**Factorial ablation and long-horizon forecasting behavior over the seven forecast days of the 168 h task.** STGCN is excluded because it is an independent graph baseline rather than an ablation variant. Panel (a) reports the day-wise publisher-compatible MAE reduction of DCRNN + SAS-Norm, DCRNN + FA-DPR and STaR-GNN relative to DCRNN. Panel (b) reports the change in each factorial variant's day-wise MAE relative to its own Day 1 value. The Day-7 changes are approximately +38.25% for DCRNN, +11.93% for DCRNN + FA-DPR, +2.64% for DCRNN + SAS-Norm and +1.70% for STaR-GNN.

## Previous Figure 3

**Empirical cumulative distributions of per-origin publisher-compatible MAE across the 46 common test origins.** Results are shown separately for the 24 h and 168 h forecasting tasks for DCRNN, STGCN and STaR-GNN. Paired common-origin comparisons show that STaR-GNN outperforms DCRNN for 45/46 origins at 24 h and 46/46 origins at 168 h, and outperforms STGCN for 45/46 and 40/46 origins, respectively.

## Previous Figure 4

**DMA-level spatial consistency of the MAE improvements achieved by STaR-GNN.** Each cell reports the relative MAE reduction of STaR-GNN compared with DCRNN or STGCN for one DMA and forecast horizon. All 40 DMA-horizon-baseline comparisons are positive, with improvements ranging from approximately 1.26% to 61.20%.

## Previous Figure 5

**Representative 168 h aggregate-demand forecast trajectory and hourly absolute errors.** The displayed test origin is selected using the pre-specified median-error proximity rule. The selection does not depend on visual appearance. The aggregate-demand hourly error is distinct from the publisher-compatible sum-of-DMA MAE used for quantitative model comparison.
