**Table S3. Forecast-origin robustness of STaR-GNN relative to the same-protocol graph baselines.**

| Horizon | Baseline | Metric | Mean paired effect | 95% block CI | Wins / 46 | High-variability mean | Wins / 12 | Spearman ρ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 24 h | DCRNN | MAE | +19.32% | [+16.66%, +21.58%] | 45/46 | +20.74% | 12/12 | +0.190 |
| 24 h | DCRNN | MAPE | +10.79% | [-5.34%, +24.50%] | 35/46 | +12.37% | 10/12 | -0.013 |
| 24 h | DCRNN | RMSE | +10.10% | [-6.33%, +24.03%] | 32/46 | +9.53% | 9/12 | +0.032 |
| 24 h | DCRNN | NSE | +0.0099 | [+0.0017, +0.0176] | 32/46 | +0.0090 | 9/12 | +0.016 |
| 24 h | STGCN | MAE | +22.63% | [+20.59%, +24.97%] | 45/46 | +17.47% | 12/12 | -0.368 |
| 24 h | STGCN | MAPE | +17.25% | [+8.30%, +25.22%] | 33/46 | +2.85% | 7/12 | -0.305 |
| 24 h | STGCN | RMSE | +17.14% | [+9.07%, +24.36%] | 34/46 | +0.73% | 8/12 | -0.357 |
| 24 h | STGCN | NSE | +0.0199 | [+0.0141, +0.0253] | 34/46 | +0.0010 | 8/12 | -0.344 |
| 168 h | DCRNN | MAE | +26.68% | [+24.08%, +29.25%] | 46/46 | +24.31% | 12/12 | -0.403 |
| 168 h | DCRNN | MAPE | +35.05% | [+28.11%, +40.99%] | 45/46 | +33.39% | 11/12 | -0.270 |
| 168 h | DCRNN | RMSE | +32.96% | [+25.29%, +39.79%] | 45/46 | +30.45% | 11/12 | -0.327 |
| 168 h | DCRNN | NSE | +0.0371 | [+0.0256, +0.0493] | 45/46 | +0.0327 | 11/12 | -0.332 |
| 168 h | STGCN | MAE | +15.06% | [+9.79%, +19.83%] | 40/46 | +4.83% | 7/12 | -0.571 |
| 168 h | STGCN | MAPE | +33.35% | [+12.30%, +52.16%] | 36/46 | +26.36% | 9/12 | -0.370 |
| 168 h | STGCN | RMSE | +30.02% | [+11.60%, +46.67%] | 35/46 | +20.71% | 9/12 | -0.415 |
| 168 h | STGCN | NSE | +0.0432 | [+0.0175, +0.0686] | 35/46 | +0.0239 | 9/12 | -0.421 |

**Note.** Error-metric effects are per-origin relative reductions; NSE effects are absolute differences. Positive values favor STaR-GNN. Confidence intervals are from an ordered seven-origin moving-block bootstrap (50,000 iterations); adjacent 168 h windows overlap and are not treated as independent. Demand difficulty is defined from observations only: for each origin, each DMA's mean absolute hourly ramp is normalized by its mean demand, and the median over ten DMAs is used. The highest horizon-specific quartile contains 12 origins. Spearman ρ describes the association between this difficulty index and paired improvement and is not a causal estimate.
