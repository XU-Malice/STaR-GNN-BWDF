**Table 2. Factorial ablation of SAS-Norm and FA-DPR, with absolute performance and improvements relative to DCRNN.**

| Horizon | Model | MAE ↓ | MAE reduction (%) | MAPE (%) ↓ | MAPE reduction (%) | RMSE ↓ | RMSE reduction (%) | NSE ↑ | NSE gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 h | DCRNN | 11.917 | 0.0 | 2.213 | 0.0 | 6.848 | 0.0 | 0.970 | 0.000 |
| 24 h | DCRNN + SAS-Norm | 10.468 | 12.2 | 2.010 | 9.2 | 6.134 | 10.4 | 0.976 | 0.006 |
| 24 h | DCRNN + FA-DPR | 11.238 | 5.7 | 1.945 | 12.1 | 6.079 | 11.2 | 0.977 | 0.007 |
| 24 h | STaR-GNN | **9.424** | **20.9** | **1.805** | **18.4** | **5.535** | **19.2** | **0.981** | **0.011** |
| 168 h | DCRNN | 16.801 | 0.0 | 3.248 | 0.0 | 9.817 | 0.0 | 0.940 | 0.000 |
| 168 h | DCRNN + SAS-Norm | **12.208** | **27.3** | 2.102 | 35.3 | 6.468 | 34.1 | 0.974 | 0.034 |
| 168 h | DCRNN + FA-DPR | 14.086 | 16.2 | 3.278 | -0.9 | 9.332 | 4.9 | 0.945 | 0.005 |
| 168 h | STaR-GNN | 12.234 | 27.2 | **2.014** | **38.0** | **6.161** | **37.2** | **0.976** | **0.036** |

**Note.** MAE is the sum of the ten DMA-level MAEs. MAPE, RMSE and NSE are evaluated on aggregate demand. Error reductions are calculated relative to DCRNN, and positive values indicate lower errors. NSE gain is the absolute difference from DCRNN. STGCN is an independent graph baseline and is not part of the factorial design.
