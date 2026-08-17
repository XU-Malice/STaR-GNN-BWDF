"""数据处理模块 — 从原始数据到模型就绪张量的完整管道。

子模块
------
- ``interpolation`` : 时间加权线性插值
- ``outlier_detection`` : 基于 IQR 的异常值检测与处理
- ``sliding_window`` : 滑动窗口切片与样本构建
- ``temporal_features`` : 时间特征工程（小时、星期、节假日等）
- ``weather_features`` : 气象特征重命名与处理
- ``metrics`` : 预测评估指标（MAE/MAPE/RMSE/NSE）
- ``loader`` : 从 wf4bwdf 包加载原始数据
- ``pipeline`` : 全流程数据构建编排
- ``builders`` : 模型就绪张量构建（MSNet / Share-STGNN / GRU-LSTM）
"""

from dma_wdf.data.interpolation import interpolate_time, interpolate_series
from dma_wdf.data.outlier_detection import (
    clip_outliers,
    compute_thresholds,
    detect_outliers,
    interpolate_outliers,
    preprocess_demand,
    preprocess_series,
)
from dma_wdf.data.sliding_window import (
    build_branch_array,
    build_sample_index,
    build_windows,
    combine_past,
    daily_starts,
    make_eval_index,
    make_train_starts,
    slice_hours,
    slice_window,
    target_24h,
    target_168h,
)
from dma_wdf.data.temporal_features import boolish_to_int, build_temporal_features
from dma_wdf.data.weather_features import PAPER_WEATHER_MAPPING, rename_weather
from dma_wdf.data.metrics import (
    compute_metrics,
    mae,
    mape,
    nse,
    rmse,
)
from dma_wdf.data.loader import load_raw_dataset, read_parquet_artifacts, select_period

__all__ = [
    # interpolation
    "interpolate_time",
    "interpolate_series",
    # outlier_detection
    "clip_outliers",
    "compute_thresholds",
    "detect_outliers",
    "interpolate_outliers",
    "preprocess_demand",
    "preprocess_series",
    # sliding_window
    "build_branch_array",
    "build_sample_index",
    "build_windows",
    "combine_past",
    "daily_starts",
    "make_eval_index",
    "make_train_starts",
    "slice_hours",
    "slice_window",
    "target_24h",
    "target_168h",
    # temporal_features
    "boolish_to_int",
    "build_temporal_features",
    # weather_features
    "PAPER_WEATHER_MAPPING",
    "rename_weather",
    # metrics
    "compute_metrics",
    "mae",
    "mape",
    "nse",
    "rmse",
    # loader
    "load_raw_dataset",
    "read_parquet_artifacts",
    "select_period",
]
