# STaR-GNN for Multi-DMA Water-Demand Forecasting

Official research code for leakage-safe 24 h and 168 h water-demand
forecasting over ten district metered areas (DMAs). The repository contains
the complete data protocol, training-only Pearson graph construction, DCRNN
and STGCN baselines, the STaR-GNN factorial variants, frozen common-46 test
artifacts, and scripts that reproduce the paper tables from raw BWDF data.

The public method name is **STaR-GNN**. The internal key `star_dcrnn` is kept
for checkpoint compatibility because the graph-recurrent backbone uses DCGRU.

## Main result

The paper configuration was selected by validation performance from three
predefined candidates and then evaluated once on the common-46 test protocol:

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

| Horizon | Variant | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | DCRNN | 5.356315 | 2.212928 | 6.848257 | 0.970419 |
| 24 h | DCRNN + State | 4.895320 | 2.010448 | 6.133886 | 0.976269 |
| 24 h | DCRNN + FA-DPR | 4.739336 | 1.944550 | 6.079036 | 0.976691 |
| 24 h | **STaR-GNN** | **4.360841** | **1.804574** | **5.534656** | **0.980679** |
| 168 h | DCRNN | 7.734838 | 3.248413 | 9.817428 | 0.939504 |
| 168 h | DCRNN + State | 5.122511 | 2.102380 | 6.468312 | 0.973739 |
| 168 h | DCRNN + FA-DPR | 7.578056 | 3.277716 | 9.332415 | 0.945334 |
| 168 h | **STaR-GNN** | **4.919812** | **2.013774** | **6.160881** | **0.976176** |

These are aggregate-demand metrics from 46 official test origins. Validation
selected the configuration (28/32 predefined relations); the test set was
used only for final transparent reporting (31/32 relations). See
[`docs/RESULTS_AND_ARTIFACTS_CN.md`](docs/RESULTS_AND_ARTIFACTS_CN.md).

## Repository layout

```text
configs/data/          data period, split, features, preprocessing
configs/graph/         training-only Pearson graph protocol
configs/model/         model-only architecture settings
configs/train/         development training settings
configs/paper/         immutable paper reproduction settings
src/dma_wdf/           data, graph, models, training, evaluation
scripts/reproduce/     freeze, verify, smoke, and full reproduction
tests/                 leakage, shape, gradient, protocol regression tests
docs/                  method, experiment, and reproduction documentation
results/paper/         generated frozen release and paper tables
```

## Installation

The recorded environment is Python 3.11.15, PyTorch 2.9.1+cu128, CUDA 12.8,
NumPy 2.4.6, and pandas 3.0.5.

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .
```

For an existing compatible environment:

```bash
python -m pip install -e ".[dev,plots,model]"
python scripts/reproduce/check_environment.py
```

## Quick verification

This checks the environment, source syntax, data protocol, graph identity,
model shapes, leakage guards, and regression tests without retraining the
paper models:

```bash
bash scripts/reproduce/smoke_test.sh
```

The GitHub Release contains checkpoints, frozen predictions, metrics, and
checksums, but intentionally does not redistribute licensed BWDF raw/processed
data. On a pristine server, the command below automatically runs the pinned
data pipeline and training-only Pearson graph builder when those artifacts are
missing; existing artifacts are reused and no model is retrained.

Verify the frozen paper artifacts, their hashes, and reported relations:

```bash
bash scripts/reproduce/verify_pretrained.sh
```

Re-evaluate every frozen DCRNN, STGCN, and STaR-GNN checkpoint on common-46:

```bash
bash scripts/reproduce/verify_pretrained.sh --re-evaluate --device cpu
```

For the author-side pre-release audit, including atomic recovery from an
interrupted artifact import, run:

```bash
bash scripts/reproduce/validate_everything.sh \
  /path/to/DMA-WDF \
  --device cuda:0
```

This checks all 10 checkpoints and frozen predictions, source and artifact
hashes, the complete test suite, the common-46 protocol, 10 fresh inference
runs, and all aggregate/DMA/day-wise/Pearson paper artifacts. It does not
retrain the models; use the from-scratch entry point below for that purpose.
Fresh CUDA/cuDNN inference is audited with absolute and relative tolerances of
`5e-4` (0.05%). The 40 metric comparisons are saved to CSV; checkpoint hashes,
protocol fields, and sample indices remain exact-match checks.
The final audit also validates DMA/day-wise/Pearson table cardinalities, nonempty
figures, and legacy HPO/SGDR code isolation without requiring `rg`.

## Reproduce from raw data

The complete pipeline trains every model before loading any official test
target:

```bash
bash scripts/reproduce/reproduce_all.sh \
  --device auto \
  --seeds 0
```

The stages are data preparation, training-only graph construction, STGCN
training, the four-cell DCRNN/STaR-GNN factorial experiment, frozen checkpoint
evaluation, and paper-table generation. Existing nonempty output directories
are never overwritten.

The complete command-to-function walkthrough is in
[`docs/FULL_PIPELINE_CN.md`](docs/FULL_PIPELINE_CN.md). Method details are in
[`docs/METHOD_CN.md`](docs/METHOD_CN.md), while result provenance and paper
artifacts are documented in
[`docs/RESULTS_AND_ARTIFACTS_CN.md`](docs/RESULTS_AND_ARTIFACTS_CN.md).

Before publishing, the author-side clean-room entry point creates a separate
source copy and a brand-new Conda prefix, rebuilds data and the training-only
Pearson graph, trains all 10 runs, and audits 40 common-46 metrics against the
frozen release:

```bash
bash scripts/reproduce/validate_clean_room.sh \
  --workspace /path/to/new-clean-room \
  --frozen-release results/paper/frozen_v1 \
  --device cuda:0 \
  --evaluation-device cuda:0
```

## Data and evaluation protocol

- Dataset period: 2021-01-01 to 2023-03-05, hourly.
- Official training period: through 2022-12-15.
- Official test period: from 2022-12-16.
- Input history: 672 h; horizons: 24 h and 168 h.
- Static graph: positive training-period Pearson correlations only, zero
  diagonal, random-walk normalization, shared by both horizons.
- Primary evaluation: `common_46`, with MAE, MAPE, RMSE, and NSE.
- Test-time teacher forcing and future demand access: disabled.

See [`data/README.md`](data/README.md) for obtaining BWDF. Raw data are not
redistributed by this repository.

## Models

- **DCRNN / Base**: the common DCGRU backbone with both proposed components
  disabled (`variant=backbone`). It is reported once in the paper.
- **STGCN**: independently trained cross-model baseline.
- **State**: DMA-wise daily state/shape transformation and restoration.
- **FA-DPR**: forecast-aligned daily pattern retrieval.
- **Full**: State + FA-DPR, the complete STaR-GNN.

The frozen release contains one DCRNN identity only: `star_gnn/Base`
(`variant=backbone`). No second `baselines/dcrnn` checkpoint is shipped.
See [`docs/RESULTS_AND_ARTIFACTS_CN.md`](docs/RESULTS_AND_ARTIFACTS_CN.md).

## Tests

```bash
python -m pytest tests -q
```

## References

- [DCRNN, ICLR 2018](https://openreview.net/forum?id=SJiHXGWAZ)
- [STGCN, IJCAI 2018](https://www.ijcai.org/proceedings/2018/505)
- [MSCMNet, Water Research X 2024](https://doi.org/10.1016/j.wroa.2024.100269)
- [BWDF / wf4bwdf](https://github.com/WaterFutures/wf4bwdf)

## License and citation

See `LICENSE` and `CITATION.cff`. Update the preferred paper citation after
the manuscript receives its final bibliographic record.
