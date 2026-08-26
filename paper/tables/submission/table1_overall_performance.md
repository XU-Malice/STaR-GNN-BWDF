**Table 1. Overall forecasting performance of the comparison models for the 24 h and 168 h prediction horizons.**

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | GRU | 16.314 | 3.100 | 10.194 | 0.916 |
| 24 h | LSTM | 17.698 | 2.900 | 9.711 | 0.920 |
| 24 h | MSNet | 15.537 | 3.200 | 9.526 | 0.929 |
| 24 h | MSCMNet-WM | 14.790 | 2.700 | 7.924 | 0.957 |
| 24 h | MSCMNet-M | 14.912 | 2.800 | 8.111 | 0.954 |
| 24 h | MSCMNet-W | 14.471 | 2.600 | 7.586 | 0.959 |
| 24 h | DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| 24 h | STGCN | 12.358 | 2.425 | 7.905 | 0.961 |
| 24 h | STaR-GNN | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | GRU | 18.305 | 3.100 | 11.353 | 0.918 |
| 168 h | LSTM | 18.678 | 2.900 | 11.031 | 0.922 |
| 168 h | MSNet | 15.908 | 3.200 | 9.698 | 0.930 |
| 168 h | MSCMNet-WM | 15.290 | 2.700 | 8.097 | 0.957 |
| 168 h | MSCMNet-M | 15.405 | 2.800 | 8.395 | 0.953 |
| 168 h | MSCMNet-W | 14.950 | 2.600 | 7.756 | 0.960 |
| 168 h | DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| 168 h | STGCN | 14.569 | 3.576 | 10.306 | 0.933 |
| 168 h | STaR-GNN | **12.234** | **2.014** | **6.161** | **0.976** |

**Note.** MAE is the sum of the ten DMA-level MAEs; MAPE, RMSE and NSE are evaluated on aggregate demand. Values for GRU, LSTM, MSNet and the MSCMNet variants were reported by Que et al. (2024). DCRNN, STGCN and STaR-GNN were evaluated using the present study's pipeline. Cross-source comparisons are used for performance positioning, whereas paired inference is restricted to the latter three models. All manuscript values use uniform three-decimal display precision; the source CSV retains full precision.
