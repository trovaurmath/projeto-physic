from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import pandas as pd


class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    IMU_ACCEL = "imu_accel"
    IMU_GYRO = "imu_gyro"
    STRAIN_GAUGE = "strain_gauge"
    MAGNETOMETER = "magnetometer"
    GENERIC = "generic"


@dataclass(slots=True, frozen=True)
class SensorChannel:
    name: str
    sensor_type: SensorType = SensorType.GENERIC
    unit: str | None = None
    sample_rate_hz: float | None = None


@dataclass(slots=True)
class PreprocessConfig:
    timestamp_col: str = "timestamp"
    sensor_columns: Sequence[str] = field(default_factory=list)
    enforce_monotonic_time: bool = True
    drop_na_timestamps: bool = True
    duplicate_strategy: str = "mean"
    resample_rule: str | None = None
    interpolation_method: str = "time"
    interpolation_limit: int | None = None
    outlier_method: str = "mad"
    outlier_threshold: float = 3.5
    lowpass_cutoff_hz: float | None = None
    lowpass_order: int = 3


@dataclass(slots=True)
class FeatureConfig:
    rolling_windows: Sequence[str] = field(default_factory=lambda: ("5s", "30s"))
    include_frequency_features: bool = True
    frequency_top_n: int = 3


@dataclass(slots=True)
class AnomalyConfig:
    univariate_threshold: float = 3.5
    multivariate_quantile: float = 0.995


@dataclass(slots=True)
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)


@dataclass(slots=True)
class AnalysisResult:
    cleaned_data: pd.DataFrame
    feature_matrix: pd.DataFrame
    anomaly_flags: pd.DataFrame
    diagnostics: dict[str, Any]
