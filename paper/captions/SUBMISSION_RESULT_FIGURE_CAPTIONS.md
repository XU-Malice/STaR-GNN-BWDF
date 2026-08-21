# Journal of Hydrology submission figure captions

## Main Figure 1

**Component contributions and lead-time stability during 168 h forecasting.** **a**, Day-wise publisher-compatible MAE for DCRNN, DCRNN + SAS-Norm, DCRNN + FA-DPR and STaR-GNN across the seven consecutive forecast days. Shaded bands denote 95% confidence intervals obtained with an ordered seven-origin moving-block bootstrap. **b**, Change in day-wise MAE relative to each model's Day-1 value. The Day-7 changes are approximately +38.25% for DCRNN, +11.93% for DCRNN + FA-DPR, +2.64% for DCRNN + SAS-Norm and +1.70% for STaR-GNN. Publisher-compatible MAE is the sum of the ten DMA-level MAEs. Forecast origins are ordered chronologically; the block procedure preserves local temporal dependence among adjacent origins.

## Main Figure 2

**Temporal and spatial robustness of the STaR-GNN improvement.** **a**, Paired publisher-compatible MAE improvement for each of the 46 common forecast origins, calculated as MAE of the baseline minus MAE of STaR-GNN. Positive values indicate lower error for STaR-GNN. Diamonds and error bars denote the mean paired improvement and its 95% moving-block-bootstrap confidence interval, respectively; annotations give the number of origins for which STaR-GNN has lower MAE. **b**, DMA-level relative MAE reduction of STaR-GNN against DCRNN and STGCN for the 24 h and 168 h forecasting tasks. The 40 DMA–horizon–baseline comparisons are all positive; darker cells denote larger reductions.

## Main Figure 3

**Week-ahead demand dynamics from population-level error structure to a representative forecast.** **a**, Mean aggregate-demand absolute error over the 24 h daily cycle for the 168 h task, obtained by folding the seven forecast days for each of the 46 common origins. Shaded bands denote 95% moving-block-bootstrap confidence intervals. **b**, Observed and predicted aggregate demand over a representative 168 h forecast selected using a pre-specified median-error rule: the STaR-GNN origin whose publisher-compatible MAE is closest to the median across the 46 common origins. **c**, Corresponding hourly absolute error of the aggregate-demand trajectory. Vertical separators indicate consecutive 24 h forecast days. Demand and aggregate-demand absolute error are expressed in L s⁻¹.

## Supplementary Figure S1

**Relative performance improvement of STaR-GNN over the comparison models.** **a**, Relative reductions in MAE, MAPE and RMSE for the 24 h and 168 h tasks, calculated as \((E_{baseline}-E_{STaR})/E_{baseline}\times100\%\). **b**, Absolute NSE gain, \(NSE_{STaR}-NSE_{baseline}\). GRU, LSTM, MSNet and the MSCMNet variants (†) are values reported by Que et al. (2024), whereas DCRNN, STGCN and STaR-GNN are evaluated under the common 46-origin protocol. Positive values indicate an improvement of STaR-GNN.

## Supplementary Figure S2

**Distribution of per-origin publisher-compatible MAE across the common test period.** Empirical cumulative distributions are shown for DCRNN, STGCN and STaR-GNN for the **a**, 24 h and **b**, 168 h forecasting tasks. Curves positioned further to the left indicate lower forecast error over a larger fraction of test origins. This distributional view complements the paired origin-level differences reported in Main Fig. 2a.
