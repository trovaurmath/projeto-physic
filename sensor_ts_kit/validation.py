from __future__ import annotations

from typing import Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Raised when a sensor dataframe does not satisfy required constraints."""


def prepare_time_index(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    drop_na_timestamps: bool = True,
    duplicate_strategy: str = "mean",
) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        prepared = df.copy()
    else:
        if timestamp_col not in df.columns:
            raise DataValidationError(
                f"Timestamp column '{timestamp_col}' not found and dataframe index is not DatetimeIndex."
            )
        prepared = df.copy()
        prepared[timestamp_col] = pd.to_datetime(prepared[timestamp_col], errors="coerce")
        if drop_na_timestamps:
            prepared = prepared.dropna(subset=[timestamp_col])
        prepared = prepared.set_index(timestamp_col)

    prepared = prepared.sort_index()

    if prepared.index.has_duplicates:
        if duplicate_strategy == "mean":
            prepared = prepared.groupby(level=0).mean(numeric_only=True)
        elif duplicate_strategy == "first":
            prepared = prepared[~prepared.index.duplicated(keep="first")]
        elif duplicate_strategy == "last":
            prepared = prepared[~prepared.index.duplicated(keep="last")]
        else:
            raise DataValidationError(
                f"Invalid duplicate_strategy='{duplicate_strategy}'. Use: mean, first, last."
            )

    if not isinstance(prepared.index, pd.DatetimeIndex):
        raise DataValidationError("Prepared dataframe index must be a DatetimeIndex.")
    return prepared


def validate_sensor_dataframe(
    df: pd.DataFrame,
    sensor_columns: Iterable[str],
    enforce_monotonic: bool = True,
) -> None:
    if df.empty:
        raise DataValidationError("Input dataframe is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("Dataframe index must be DatetimeIndex.")
    if enforce_monotonic and not df.index.is_monotonic_increasing:
        raise DataValidationError("Timestamp index must be monotonic increasing.")

    columns = list(sensor_columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required sensor columns: {missing}")

    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"Sensor columns must be numeric: {non_numeric}")


def infer_sensor_columns(df: pd.DataFrame, provided: Iterable[str] | None = None) -> list[str]:
    if provided:
        return list(provided)
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def estimate_sample_rate_hz(index: pd.DatetimeIndex) -> float | None:
    if len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return None
    median_delta = float(deltas.median())
    if median_delta <= 0:
        return None
    return 1.0 / median_delta
