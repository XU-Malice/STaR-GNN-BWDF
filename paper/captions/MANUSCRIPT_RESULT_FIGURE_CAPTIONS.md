# Manuscript result figure captions

## Figure 1

**Relative performance improvement of STaR-GNN over competing models for the 24 h and 168 h forecasting tasks.** Panel (a) reports the relative reductions in MAE, MAPE and RMSE, calculated as \((E_{baseline}-E_{STaR})/E_{baseline}\times100\%\); panel (b) reports the absolute NSE gain \(NSE_{STaR}-NSE_{baseline}\). Positive values indicate an improvement of STaR-GNN. GRU, LSTM, MSNet and the MSCMNet variants are reported results from Que et al. (2024), whereas DCRNN, STGCN and STaR-GNN are evaluated on the common 46-sequence test period. The publisher-compatible total convention is used for cross-model comparison.

## Figure 2

**Evolution of publisher-compatible MAE across the seven forecast days of the 168 h task.** For each common test origin and each forecast day, MAE is first calculated separately for DMA A--J over the corresponding 24 h interval and then summed across DMAs. Lines show the mean across the 46 common test origins, and shaded bands denote deterministic nonparametric bootstrap 95% confidence intervals. This figure evaluates whether model errors accumulate as the forecast lead increases.

## Figure 3

**Empirical cumulative distributions of per-origin publisher-compatible MAE across the 46 common test origins.** Results are shown separately for the 24 h and 168 h forecasting tasks. For each origin, publisher-compatible MAE is the sum of DMA-level MAEs. Curves located further to the left indicate lower forecast errors over a larger fraction of test origins and therefore stronger sample-level robustness.

## Figure 4

**DMA-level spatial consistency of the MAE improvements achieved by STaR-GNN.** Each cell reports the relative MAE reduction of STaR-GNN compared with DCRNN or STGCN for one DMA and forecast horizon. Positive values indicate lower DMA-level MAE for STaR-GNN. The figure complements the absolute DMA-level metrics reported in Table 3 by showing whether the overall improvement is spatially consistent across DMA A--J.

## Figure 5

**Representative 168 h aggregate-demand forecast trajectory and hourly absolute errors.** The displayed test origin is selected using a pre-specified rule: among the 46 common test origins, the origin whose STaR-GNN publisher-compatible MAE is closest to the median is selected. The selection therefore does not depend on the visual appearance of the trajectories. The upper panel compares observed aggregate demand with predictions from STGCN, DCRNN and STaR-GNN; the lower panel shows the corresponding hourly absolute errors. Vertical separators denote consecutive 24 h forecast days.
