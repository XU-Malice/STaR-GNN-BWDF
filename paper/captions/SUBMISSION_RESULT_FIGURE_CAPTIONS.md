# Journal of Hydrology submission figure captions

## Main Figure 1

**Overall forecasting performance across prediction horizons and model families.** **a**, Relative reductions in MAE, MAPE and RMSE achieved by STaR-GNN against six published sequence/multiscale reference models and two re-evaluated graph baselines for the 24 h and 168 h tasks. **b**, Corresponding absolute NSE gains. Error-metric reductions are calculated as \((E_{baseline}-E_{STaR})/E_{baseline}\times100\%\); positive values consistently favor STaR-GNN. Values for GRU, LSTM, MSNet and the MSCMNet variants are reported by Que et al. (2024), whereas DCRNN, STGCN and STaR-GNN are evaluated using the common 46-origin protocol.

## Main Figure 2

**Component contributions and lead-time stability during 168 h forecasting.** Day-wise paired improvements of DCRNN + SAS-Norm, DCRNN + FA-DPR and STaR-GNN relative to DCRNN are shown for **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE. Positive values favor the variant. Points are horizontally offset within each forecast day to keep the similar SAS-Norm and STaR-GNN estimates visually distinguishable; vertical bars are 95% confidence intervals from an ordered seven-origin moving-block bootstrap. MAE is the sum of the ten DMA-level MAEs, while MAPE, RMSE and NSE are evaluated on aggregate demand.

## Main Figure 3

**Temporal and spatial robustness of STaR-GNN across all four metrics.** **a**, Common forecast-origin comparisons and **b**, DMA-level comparisons against DCRNN and STGCN for the 24 h and 168 h tasks. Color encodes the percentage of comparisons improved by STaR-GNN. Each cell gives the direction-aligned mean improvement followed by the number improved over the number compared. Error metrics are shown as relative reductions; NSE is shown as an absolute gain. STaR-GNN improves 158 of the 160 DMA–horizon–baseline–metric comparisons; the two exceptions are RMSE and NSE for DMA G against STGCN at 168 h.

## Main Figure 4

**Week-ahead demand dynamics from population-level error structure to a representative forecast.** **a**, Mean aggregate-demand absolute error over the 24 h daily cycle for the 168 h task, obtained by folding the seven forecast days for each of the 46 common origins; shaded bands are ordered moving-block 95% confidence intervals. **b**, Observed and predicted aggregate demand over a representative 168 h forecast selected using a pre-specified median-total-MAE rule. **c**, Corresponding hourly absolute errors. Vertical separators indicate consecutive 24 h forecast days. Demand and error are expressed in L s⁻¹.

## Supplementary Figure S1

**DMA-level improvement of STaR-GNN across four metrics.** Relative MAE, MAPE and RMSE reductions and absolute NSE gains are shown against DCRNN and STGCN for each DMA at 24 h and 168 h. Blue denotes improvement and red denotes deterioration. The detailed absolute values are reported in Supplementary Table S1.

## Supplementary Figure S2

**Distribution of per-origin total MAE across the common test period.** Empirical cumulative distributions are shown for DCRNN, STGCN and STaR-GNN for the **a**, 24 h and **b**, 168 h tasks. Curves further to the left indicate lower error over a larger fraction of test origins. Total MAE is the sum of the ten DMA-level MAEs.
