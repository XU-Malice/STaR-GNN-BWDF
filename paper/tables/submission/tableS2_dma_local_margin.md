**Table S2. DMA-level margins of STaR-GNN relative to the locally strongest competing model.**

| Horizon | DMA | MAE competitor | ΔMAE (%) | MAPE competitor | ΔMAPE (%) | RMSE competitor | ΔRMSE (%) | NSE competitor | ΔNSE |
|---|---|---|---:|---|---:|---|---:|---|---:|
| 24 h | A | GRU | -9.2 | GRU | -5.2 | MSCMNet-W | -40.3 | MSCMNet-W | -0.073 |
| 24 h | B | STGCN | +15.9 | STGCN | +17.0 | STGCN | +11.8 | STGCN | +0.043 |
| 24 h | C | MSCMNet-WM | +33.8 | MSCMNet-WM | +40.0 | MSCMNet-WM | +31.4 | MSCMNet-WM | +0.087 |
| 24 h | D | DCRNN | +1.7 | DCRNN | +1.4 | DCRNN | +3.2 | DCRNN | +0.010 |
| 24 h | E | DCRNN | +22.3 | DCRNN | +21.6 | DCRNN | +25.2 | DCRNN | +0.010 |
| 24 h | F | DCRNN | +12.6 | DCRNN | +11.4 | DCRNN | +11.4 | DCRNN | +0.091 |
| 24 h | G | DCRNN | +19.2 | DCRNN | +20.3 | MSCMNet-W | +16.2 | DCRNN | +0.035 |
| 24 h | H | GRU | +17.8 | MSCMNet-WM | +22.0 | GRU | +12.8 | STGCN | +0.027 |
| 24 h | I | STGCN | +19.9 | STGCN | +19.3 | STGCN | +16.3 | STGCN | +0.100 |
| 24 h | J | DCRNN | +3.4 | DCRNN | +0.7 | DCRNN | +4.3 | DCRNN | +0.014 |
| 168 h | A | MSCMNet-W | -17.3 | MSCMNet-W | -6.4 | MSCMNet-W | -40.4 | MSCMNet-W | -0.159 |
| 168 h | B | MSNet | +11.9 | MSNet | +10.3 | MSCMNet-WM | +5.2 | MSCMNet-WM | +0.063 |
| 168 h | C | MSCMNet-WM | +12.3 | MSCMNet-WM | +23.6 | MSCMNet-WM | +11.4 | MSCMNet-WM | +0.031 |
| 168 h | D | LSTM | +1.7 | LSTM | +0.4 | STGCN | +1.1 | STGCN | +0.004 |
| 168 h | E | MSCMNet-M | -14.9 | MSNet | -10.3 | MSNet | -9.9 | MSNet | -0.005 |
| 168 h | F | DCRNN | +9.5 | DCRNN | +12.0 | DCRNN | +9.1 | DCRNN | +0.086 |
| 168 h | G | MSCMNet-M | -8.0 | MSCMNet-M | -3.3 | MSCMNet-M | -13.7 | MSCMNet-M | -0.023 |
| 168 h | H | GRU | +7.3 | MSCMNet-WM | +12.3 | GRU | +12.1 | GRU | +0.026 |
| 168 h | I | MSCMNet-WM | +3.2 | MSCMNet-WM | +0.5 | MSCMNet-WM | +3.3 | MSCMNet-WM | -0.046 |
| 168 h | J | STGCN | +18.6 | MSNet | +13.0 | MSNet | +12.4 | DCRNN | +0.133 |

**Note.** For each horizon, DMA and metric, the comparator is selected independently as the best-performing non-STaR-GNN model among the eight alternatives. For MAE, MAPE and RMSE, \(\Delta E=(E_{competitor}-E_{STaR-GNN})/E_{competitor}\times100\%\); for NSE, \(\Delta\mathrm{NSE}=\mathrm{NSE}_{STaR-GNN}-\mathrm{NSE}_{competitor}\). Positive values favor STaR-GNN; negative values identify local losses. All values are retained, including cases in which STaR-GNN is not best. Source provenance for the underlying absolute metrics is provided in Supplementary Table S1.
