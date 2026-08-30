**Table S4. Aggregate-demand error summary underlying Main Fig. 7.**

| Evaluation scope | Model | Aggregate-demand MAE (L s\(^{-1}\)) |
|---|---|---:|
| 46 common forecast origins, 168 h per origin | DCRNN | 7.735 |
| 46 common forecast origins, 168 h per origin | STGCN | 8.574 |
| 46 common forecast origins, 168 h per origin | STaR-GNN | **4.920** |

**Note.** Values are calculated from the same forecast origins and raw outputs used in Main Figs. 6 and 7. Aggregate demand is the sum of the ten DMA predictions at each hour, and MAE is averaged over all forecast hours and common origins. The representative trajectory in Main Fig. 7 is selected before visual inspection as the STaR-GNN origin whose 168 h aggregate-demand MAE is closest to the median across the 46 common origins.
