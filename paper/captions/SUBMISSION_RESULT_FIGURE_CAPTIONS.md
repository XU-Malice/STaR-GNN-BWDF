# Journal of Hydrology submission figure captions

## Main Figure 1

**Overall forecasting performance across prediction horizons and model families.** **a**, Relative reductions in MAE, MAPE and RMSE achieved by STaR-GNN against GRU, LSTM, MSNet, three MSCMNet variants, DCRNN and STGCN for the 24 h and 168 h tasks. **b**, Corresponding absolute improvements in NSE, defined as \(\Delta\mathrm{NSE}=\mathrm{NSE}_{\mathrm{STaR-GNN}}-\mathrm{NSE}_{\mathrm{baseline}}\). Error-metric reductions are calculated as \((E_{baseline}-E_{STaR-GNN})/E_{baseline}\times100\%\); positive values consistently favor STaR-GNN. MAE is the sum of the ten DMA-level MAEs, whereas MAPE, RMSE and NSE are evaluated on aggregate demand. Values for GRU, LSTM, MSNet and the MSCMNet variants are taken from Que et al. (2024), whereas DCRNN, STGCN and STaR-GNN are evaluated using the present study's pipeline; cross-source comparisons are used for performance positioning rather than paired inference.

## Main Figure 2

**Cross-DMA distribution of STaR-GNN improvements relative to individual baseline models.** Signed improvements are shown for **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE. Small points denote individual DMAs; large markers and horizontal bars denote the median and interquartile range across the ten DMAs. Circles and squares represent 24 h and 168 h forecasts, respectively. For the error metrics, the horizontal axis is \((E_{baseline}-E_{STaR-GNN})/E_{baseline}\times100\%\); for NSE, it is \(\Delta\mathrm{NSE}=\mathrm{NSE}_{STaR-GNN}-\mathrm{NSE}_{baseline}\). Positive values favor STaR-GNN. The horizontal separator distinguishes six sequence baselines from two spatiotemporal graph baselines.

## Main Figure 3

**DMA-level performance of STaR-GNN and the best-performing baseline for 24 h forecasting.** Panels show **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE across DMAs A–J. For each DMA and metric, the best-performing baseline is selected from GRU, LSTM, MSNet, MSCMNet-WM, MSCMNet-M, MSCMNet-W, DCRNN and STGCN. Blue squares denote STaR-GNN, gray open circles denote the best-performing baseline, and vertical segments connect the two absolute values. Circles and segments are orange when the baseline outperforms STaR-GNN. Focused vertical ranges are used for MAPE and NSE to improve the legibility of local differences. Lower MAE, MAPE and RMSE and higher NSE indicate better performance.

## Main Figure 4

**DMA-level performance of STaR-GNN and the best-performing baseline for 168 h forecasting.** Panels show **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE across DMAs A–J. For each DMA and metric, the best-performing baseline is selected from GRU, LSTM, MSNet, MSCMNet-WM, MSCMNet-M, MSCMNet-W, DCRNN and STGCN. Blue squares denote STaR-GNN, gray open circles denote the best-performing baseline, and vertical segments connect the two absolute values. Circles and segments are orange when the baseline outperforms STaR-GNN. Focused vertical ranges are used for MAPE and NSE to improve the legibility of local differences. Lower MAE, MAPE and RMSE and higher NSE indicate better performance.

## Main Figure 5

**Component contributions and lead-time stability during 168 h forecasting.** Day-wise paired improvements of DCRNN + SAS-Norm, DCRNN + FA-DPR and STaR-GNN relative to DCRNN are shown for **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE. Error metrics are expressed as relative reductions, while NSE is expressed as the absolute change \(\Delta\mathrm{NSE}\); positive values favor the variant. Points are horizontally offset within each forecast day to keep the similar SAS-Norm and STaR-GNN estimates visually distinguishable; vertical bars are 95% confidence intervals from an ordered seven-window moving-block bootstrap. MAE is the sum of the ten DMA-level MAEs, while MAPE, RMSE and NSE are evaluated on aggregate demand.

## Main Figure 6

**Robustness across forecast origins and high-variability demand windows.** **a–b**, Per-origin reductions in MAE, MAPE and RMSE achieved by STaR-GNN relative to DCRNN and STGCN for the 24 h and 168 h tasks. Faint points are the 46 common forecast origins; filled symbols and horizontal bars are the means and 95% confidence intervals from an ordered seven-origin moving-block bootstrap. **c**, Corresponding absolute NSE improvements. **d**, Win counts in the horizon-specific highest quartile of observed normalized mean absolute ramp (n = 12 for each task). The difficulty index is calculated independently of model errors by normalizing each DMA's mean absolute hourly demand change by its mean demand and taking the median across the ten DMAs. MAE is summed over DMA-level MAEs; MAPE, RMSE and NSE are evaluated on aggregate demand. Color denotes win rate, and cells show the number of wins. Positive effects consistently favor STaR-GNN.

## Main Figure 7

**Week-ahead demand dynamics from population-level error structure to a representative forecast.** **a**, Mean aggregate-demand absolute error over the 24 h daily cycle for the 168 h task, obtained by folding the seven forecast days across all common test origins; shaded bands are ordered seven-origin moving-block bootstrap 95% confidence intervals. **b**, Observed and predicted aggregate demand over a representative 168 h forecast selected using a pre-specified median-total-MAE rule. **c**, Corresponding hourly absolute errors. Vertical separators indicate consecutive 24 h forecast days. Demand and error are expressed in L s⁻¹.
