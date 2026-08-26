# Journal of Hydrology submission figure captions

## Main Figure 1

**Overall forecasting performance across prediction horizons and model families.** **a**, Relative reductions in MAE, MAPE and RMSE achieved by STaR-GNN against GRU, LSTM, MSNet, three MSCMNet variants, DCRNN and STGCN for the 24 h and 168 h tasks. **b**, Corresponding absolute improvements in NSE, defined as \(\Delta\mathrm{NSE}=\mathrm{NSE}_{\mathrm{STaR-GNN}}-\mathrm{NSE}_{\mathrm{baseline}}\). Error-metric reductions are calculated as \((E_{baseline}-E_{STaR-GNN})/E_{baseline}\times100\%\); positive values consistently favor STaR-GNN. Values for GRU, LSTM, MSNet and the MSCMNet variants are taken from Que et al. (2024), whereas DCRNN, STGCN and STaR-GNN are evaluated using the present study's pipeline.

## Main Figure 2

**Cross-DMA breadth of STaR-GNN improvements relative to individual baseline models.** Signed DMA-level improvements are shown for **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE. Each small point is one DMA-specific contrast with the baseline identified on the vertical axis; large markers and horizontal bars denote the median and interquartile range across the ten DMAs. Circles and squares represent the 24 h and 168 h horizons, respectively. Error metrics are expressed as relative reductions, whereas NSE is expressed as \(\Delta\mathrm{NSE}=\mathrm{NSE}_{\mathrm{STaR-GNN}}-\mathrm{NSE}_{\mathrm{baseline}}\). Positive values favor STaR-GNN. The upper six baselines are sequence models and the lower two are spatiotemporal graph models. At each horizon, the ten DMAs, eight baselines, and four metrics define 320 descriptive contrasts; they quantify coverage and are not treated as independent statistical observations. Complete absolute values are provided in Supplementary Table S1.

## Main Figure 3

**DMA-specific margins over the locally strongest competing model and their change with forecasting horizon.** Signed margins are shown for **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE. For every DMA, metric, and horizon, the comparator is independently selected as the best-performing method among the other eight models. Circles and squares denote 24 h and 168 h forecasts, and lines connect the two horizons for the same DMA. Error metrics are expressed as relative reductions and NSE as an absolute difference; positive values favor STaR-GNN. Red markers retain cases in which another method is locally superior rather than removing them through cross-DMA aggregation. Complete values and competitor identities are provided in Supplementary Table S1 and the figure audit data.

## Main Figure 4

**Component contributions and lead-time stability during 168 h forecasting.** Day-wise paired improvements of DCRNN + SAS-Norm, DCRNN + FA-DPR and STaR-GNN relative to DCRNN are shown for **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE. Error metrics are expressed as relative reductions, while NSE is expressed as the absolute change \(\Delta\mathrm{NSE}\); positive values favor the variant. Points are horizontally offset within each forecast day to keep the similar SAS-Norm and STaR-GNN estimates visually distinguishable; vertical bars are 95% confidence intervals from an ordered seven-window moving-block bootstrap. MAE is the sum of the ten DMA-level MAEs, while MAPE, RMSE and NSE are evaluated on aggregate demand.

## Main Figure 5

**Week-ahead demand dynamics from population-level error structure to a representative forecast.** **a**, Mean aggregate-demand absolute error over the 24 h daily cycle for the 168 h task, obtained by folding the seven forecast days across all common test windows; shaded bands are ordered moving-block 95% confidence intervals. **b**, Observed and predicted aggregate demand over a representative 168 h forecast selected using a pre-specified median-total-MAE rule. **c**, Corresponding hourly absolute errors. Vertical separators indicate consecutive 24 h forecast days. Demand and error are expressed in L s⁻¹.
