# Metric conventions

Two MAE definitions are intentionally retained and must not be mixed.

- `table_internal_common46.*`: all four metrics are calculated on the hourly aggregate-demand series after summing DMA A--J. This table is used for the unified in-repository comparison among STGCN, DCRNN and STaR-GNN.
- `table_literature_comparison_common46.*`: follows Que et al. (2024) supplementary Tables S1-1--S1-8. Total MAE is the sum of DMA-level MAEs; MAPE, RMSE and NSE are calculated on the hourly aggregate-demand series. This table is used only for direct comparison with the reported GRU/LSTM/MSNet/MSCMNet results.
- `table_comparison_common46.*` is retained as a backward-compatible alias of the literature-comparison table.

For the frozen common-46 results, STaR-GNN aggregate-demand MAE is 4.360841 (24 h) and 4.919812 (168 h), whereas the publisher-compatible sum-of-DMA MAE is 9.424199 and 12.233590, respectively.
