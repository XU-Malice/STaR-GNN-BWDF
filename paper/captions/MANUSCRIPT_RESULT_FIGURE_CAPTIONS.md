# Manuscript result figure captions

## Figure 1

**Relative performance improvement of STaR-GNN over competing models for the 24 h and 168 h forecasting tasks.** Panel (a) reports the relative reductions in MAE, MAPE and RMSE, calculated as \((E_{baseline}-E_{STaR})/E_{baseline}\times100\%\); panel (b) reports the absolute NSE gain \(NSE_{STaR}-NSE_{baseline}\). Positive values indicate an improvement of STaR-GNN. GRU, LSTM, MSNet and the MSCMNet variants (marked with †) are reported results from Que et al. (2024), whereas DCRNN, STGCN and STaR-GNN are evaluated on the common 46-sequence test period. The publisher-compatible total convention is used for cross-model comparison.

## Figure 2

**Factorial ablation and long-horizon forecasting behavior over the seven forecast days of the 168 h task.** STGCN is intentionally excluded because it is an independent graph baseline rather than an ablation variant. Panel (a) reports the day-wise publisher-compatible MAE reduction of DCRNN + SAS-Norm, DCRNN + FA-DPR and STaR-GNN relative to DCRNN, calculated as \((MAE_{DCRNN,d}-MAE_{model,d})/MAE_{DCRNN,d}\times100\%\). Panel (b) reports the change in each factorial variant's day-wise MAE relative to its own Day 1 value. The Day-7 changes are approximately +38.25% for DCRNN, +11.93% for DCRNN + FA-DPR, +2.64% for DCRNN + SAS-Norm and +1.70% for STaR-GNN. The figure therefore emphasizes component contribution and lead-time stability rather than magnifying the small 168 h publisher-MAE point-estimate difference between SAS-Norm-only and the complete model.

## Figure 3

**Empirical cumulative distributions of per-origin publisher-compatible MAE across the 46 common test origins.** Results are shown separately for the 24 h and 168 h forecasting tasks for DCRNN, STGCN and STaR-GNN. For each origin, publisher-compatible MAE is the sum of DMA-level MAEs. Curves located further to the left indicate lower forecast errors over a larger fraction of test origins. Paired common-origin comparisons show that STaR-GNN outperforms DCRNN for 45/46 origins at 24 h and 46/46 origins at 168 h, and outperforms STGCN for 45/46 and 40/46 origins, respectively. Figure 3 is a baseline-robustness analysis and should not be interpreted as an ablation figure.

## Figure 4

**DMA-level spatial consistency of the MAE improvements achieved by STaR-GNN.** Each cell reports the relative MAE reduction of STaR-GNN compared with DCRNN or STGCN for one DMA and forecast horizon. Positive values indicate lower DMA-level MAE for STaR-GNN. All 40 DMA-horizon-baseline comparisons are positive, with improvements ranging from approximately 1.26% to 61.20%, indicating that the overall improvement is not dominated by a small subset of DMAs. The magnitude of the improvement remains spatially heterogeneous across DMAs.

## Figure 5

**Representative 168 h aggregate-demand forecast trajectory and hourly absolute errors.** The displayed test origin is selected using a pre-specified rule: among the 46 common test origins, the origin whose STaR-GNN publisher-compatible MAE is closest to the median is selected. The selection therefore does not depend on the visual appearance of the trajectories. The upper panel compares observed aggregate demand with predictions from STGCN, DCRNN and STaR-GNN; the lower panel shows the corresponding hourly absolute errors of the aggregate-demand trajectory. Vertical separators denote consecutive 24 h forecast days. The lower-panel absolute error is an aggregate-demand diagnostic and is distinct from the publisher-compatible sum-of-DMA MAE used for the quantitative model comparison.
