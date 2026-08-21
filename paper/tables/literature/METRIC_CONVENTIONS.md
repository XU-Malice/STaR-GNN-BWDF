# Metric conventions

Manuscript-facing overall comparison and factorial ablation use the same
publisher-compatible convention as the Que et al. (2024) supplementary tables.

- Total MAE: sum of DMA A--J MAEs.
- Total MAPE/RMSE/NSE: metric on the hourly aggregate-demand series.
- `table_literature_comparison_common46.*`: nine-model overall comparison; STGCN is an independent graph baseline.
- `table_ablation_common46.*`: exactly four factorial variants: DCRNN, DCRNN + SAS-Norm, DCRNN + FA-DPR, STaR-GNN.
- `table_star_gnn_dma_common46.*`: DMA A--J metrics without cross-DMA aggregation.
- `table_internal_common46.*`: aggregate-demand diagnostics only; do not mix its MAE with publisher-compatible MAE.

The 168 h publisher-compatible MAE of SAS-Norm-only (12.207835) and STaR-GNN
(12.233590) differs by only 0.025755 (about 0.21%). This small point-estimate
difference is reported transparently and is not used to claim that the full
model dominates SAS-Norm on every metric.
