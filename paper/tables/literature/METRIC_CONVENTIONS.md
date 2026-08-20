# Metric conventions

The manuscript-facing overall comparison and ablation tables use one publisher-compatible convention aligned with Que et al. (2024) Supplementary Tables S1-1--S1-8:

- total MAE = sum of DMA A--J MAEs;
- total MAPE/RMSE/NSE = metrics on the hourly aggregate-demand series.

Files:

- `table_literature_comparison_common46.*`: nine-model overall comparison (GRU, LSTM, MSNet, MSCMNet_WM, MSCMNet_M, MSCMNet_W, DCRNN, STGCN, STaR-GNN).
- `table_ablation_common46.*`: publisher-compatible graph-model/module comparison (STGCN, DCRNN, DCRNN + SAS-Norm, DCRNN + FA-DPR, STaR-GNN).
- `table_star_gnn_dma_common46.*`: STaR-GNN DMA A--J MAE/MAPE/RMSE/NSE for 24 h and 168 h; no cross-DMA aggregation is applied.
- `table_internal_common46.*`: pure aggregate-demand operational diagnostic retained for reproducibility only; it is not the manuscript-facing cross-model MAE table.
- `table_comparison_common46.*`: backward-compatible alias of the literature-comparison table.

Frozen STaR-GNN publisher-compatible total MAE is 9.424199 (24 h) and 12.233590 (168 h). The publisher-compatible ablation audit is 30/32 because (i) FA-DPR 168 h MAPE is slightly worse than DCRNN, and (ii) STaR-GNN 168 h sum-of-DMA MAE is slightly higher than SAS-Norm-only (12.233590 vs 12.207835).
