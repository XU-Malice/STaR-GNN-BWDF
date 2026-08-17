"""Model-neutral aliases for the shared BWDF forecasting tensors.

The implementation originated with the DCRNN baseline, so the established
classes keep their historical names for checkpoint compatibility.  Both
DCRNN and STGCN intentionally consume the exact same arrays, scalers, split
logic, and evaluation-origin definitions through the aliases below.
"""

from dma_wdf.data.dcrnn_dataset import (
    DCRNNTestData as ForecastTestData,
    DCRNNTrainingData as ForecastTrainingData,
    DCRNNWindowSubset as ForecastWindowSubset,
    ZScoreScaler,
    build_evaluation_protocol_indices,
    encode_temporal_features,
    load_sample_index,
    prepare_dcrnn_test_data as prepare_forecast_test_data,
    prepare_dcrnn_training_data as prepare_forecast_training_data,
    split_development_index,
)

__all__ = [
    "ForecastTestData",
    "ForecastTrainingData",
    "ForecastWindowSubset",
    "ZScoreScaler",
    "build_evaluation_protocol_indices",
    "encode_temporal_features",
    "load_sample_index",
    "prepare_forecast_test_data",
    "prepare_forecast_training_data",
    "split_development_index",
]
