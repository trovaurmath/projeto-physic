from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def interpolate_missing(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "time",
    limit: int | None = None,
) -> pd.DataFrame:
    filled = df.copy()
    filled[columns] = filled[columns].interpolate(method=method, limit=limit, limit_direction="both")
    return filled


def robust_score(series: pd.Series, method: str = "mad") -> pd.Series:
    s = series.astype(float)
    if method == "zscore":
        std = float(s.std(ddof=0))
        if np.isclose(std, 0.0):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if np.isclose(iqr, 0.0):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.median()) / iqr

    median = s.median()
    mad = np.median(np.abs(s - median))
    if np.isclose(mad, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return 0.6745 * (s - median) / mad


def replace_outliers_with_nan(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "mad",
    threshold: float = 3.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        score = robust_score(cleaned[col], method=method).abs()
        col_mask = score > threshold
        cleaned.loc[col_mask, col] = np.nan
        mask[col] = col_mask
    return cleaned, mask


def lowpass_filter(
    df: pd.DataFrame,
    columns: list[str],
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int = 3,
) -> pd.DataFrame:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive.")

    nyquist = 0.5 * sample_rate_hz
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1:
        raise ValueError(
            "Cutoff frequency must be lower than Nyquist frequency (sample_rate_hz / 2)."
        )

    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    filtered = df.copy()
    min_len = max(3 * max(len(a), len(b)), 10)

    for col in columns:
        series = filtered[col].astype(float)
        if series.notna().sum() < min_len:
            continue
        interp = series.interpolate(method="linear", limit_direction="both")
        filtered[col] = filtfilt(b, a, interp.values)

    return filtered
