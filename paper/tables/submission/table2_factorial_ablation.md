**Table 2. Factorial ablation of SAS-Norm and FA-DPR for the 24 h and 168 h prediction horizons.**

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| 24 h | DCRNN + SAS-Norm | 10.468 | 2.010 | 6.134 | 0.976 |
| 24 h | DCRNN + FA-DPR | 11.238 | 1.945 | 6.079 | 0.977 |
| 24 h | STaR-GNN | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| 168 h | DCRNN + SAS-Norm | **12.208** | 2.102 | 6.468 | 0.974 |
| 168 h | DCRNN + FA-DPR | 14.086 | 3.278 | 9.332 | 0.945 |
| 168 h | STaR-GNN | 12.234 | **2.014** | **6.161** | **0.976** |

**Note.** MAE is the sum of the ten DMA-level MAEs; MAPE, RMSE and NSE are evaluated on aggregate demand. STGCN is an independent graph baseline and is excluded from the factorial ablation. At 168 h, SAS-Norm-only has the marginally lower MAE (12.208 vs. 12.234), whereas the full STaR-GNN is best in MAPE, RMSE and NSE. The corresponding paired moving-block analysis is reported with Main Fig. 4.
