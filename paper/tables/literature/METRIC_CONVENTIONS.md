# Metric conventions

The manuscript-facing overall comparison and factorial ablation use the same publisher-compatible convention aligned with Que et al. (2024) Supplementary Tables S1-1--S1-8:

- total MAE = sum of DMA A--J MAEs;
- total MAPE/RMSE/NSE = metrics on the hourly aggregate-demand series.

## Files

- `table_literature_comparison_common46.*`: nine-model overall comparison (GRU, LSTM, MSNet, MSCMNet_WM, MSCMNet_M, MSCMNet_W, DCRNN, STGCN, STaR-GNN).
- `table_ablation_common46.*`: **four-model factorial ablation only** (DCRNN, DCRNN + SAS-Norm, DCRNN + FA-DPR, STaR-GNN). STGCN is an independent graph baseline and is not included.
- `table_star_gnn_dma_common46.*`: STaR-GNN DMA A--J metrics with no cross-DMA aggregation.
- `table_internal_common46.*`: aggregate-demand operational diagnostics retained for reproducibility only; its MAE is not used for manuscript cross-model MAE comparisons.
- `table_comparison_common46.*`: backward-compatible alias of the literature-comparison table.

Frozen STaR-GNN publisher-compatible total MAE is `9.424199` (24 h) and `12.233590` (168 h).

For 168 h, SAS-Norm-only has publisher-compatible MAE `12.207835`, only `0.025755` (about `0.21%`) below STaR-GNN. The aggregate-demand ordering is the opposite (`4.919812` for STaR-GNN versus `5.122511` for SAS-Norm-only). Because 168 h forecast origins are one day apart and strongly overlap, the final manuscript audit uses a seven-origin moving-block bootstrap for the paired Full-minus-SAS difference; its confidence interval includes zero. This small point-estimate difference is therefore reported transparently but is not treated as a stable performance gap.

The 30/32 factorial cell audit has two transparent exceptions: FA-DPR 168 h MAPE is slightly worse than DCRNN, and STaR-GNN 168 h publisher-compatible MAE is slightly higher than SAS-Norm-only.
