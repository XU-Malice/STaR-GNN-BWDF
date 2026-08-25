# Journal of Hydrology submission figure captions

## Main Figure 1

**Overall forecasting performance across prediction horizons and model families.** **a**, Relative reductions in MAE, MAPE and RMSE achieved by STaR-GNN against GRU, LSTM, MSNet, three MSCMNet variants, DCRNN and STGCN for the 24 h and 168 h tasks. **b**, Corresponding absolute improvements in NSE, defined as \(\Delta\mathrm{NSE}=\mathrm{NSE}_{\mathrm{STaR-GNN}}-\mathrm{NSE}_{\mathrm{baseline}}\). Error-metric reductions are calculated as \((E_{baseline}-E_{STaR-GNN})/E_{baseline}\times100\%\); positive values consistently favor STaR-GNN. Values for GRU, LSTM, MSNet and the MSCMNet variants are taken from Que et al. (2024), whereas DCRNN, STGCN and STaR-GNN are evaluated using the present study's pipeline.

## Main Figure 2

**DMA-level performance breadth across model families and forecasting horizons.** Distributions of within-DMA ranks across ten DMAs and four evaluation metrics are shown for the **a**, 24 h and **b**, 168 h tasks. Each observation is one DMA–metric combination, and lower ranks indicate better performance; boxes show the median and interquartile range, whiskers span the observed range, and overlaid points show all 40 combinations. **c**, Number of metrics for which STaR-GNN ranks first in each DMA and horizon. Models are ranked separately within every horizon–DMA–metric combination to avoid mixing heterogeneous demand scales or metric units; tied values receive the same minimum rank. Complete absolute values are reported in Supplementary Table S1.

## Main Figure 3

**Component contributions and lead-time stability during 168 h forecasting.** Day-wise paired improvements of DCRNN + SAS-Norm, DCRNN + FA-DPR and STaR-GNN relative to DCRNN are shown for **a**, MAE; **b**, MAPE; **c**, RMSE; and **d**, NSE. Error metrics are expressed as relative reductions, while NSE is expressed as the absolute change \(\Delta\mathrm{NSE}\); positive values favor the variant. Points are horizontally offset within each forecast day to keep the similar SAS-Norm and STaR-GNN estimates visually distinguishable; vertical bars are 95% confidence intervals from an ordered seven-window moving-block bootstrap. MAE is the sum of the ten DMA-level MAEs, while MAPE, RMSE and NSE are evaluated on aggregate demand.

## Main Figure 4

**Week-ahead demand dynamics from population-level error structure to a representative forecast.** **a**, Mean aggregate-demand absolute error over the 24 h daily cycle for the 168 h task, obtained by folding the seven forecast days across all common test windows; shaded bands are ordered moving-block 95% confidence intervals. **b**, Observed and predicted aggregate demand over a representative 168 h forecast selected using a pre-specified median-total-MAE rule. **c**, Corresponding hourly absolute errors. Vertical separators indicate consecutive 24 h forecast days. Demand and error are expressed in L s⁻¹.
