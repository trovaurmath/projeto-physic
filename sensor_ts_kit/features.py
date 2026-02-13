from __future__ import annotations

import numpy as np
import pandas as pd


def add_vector_magnitude(df: pd.DataFrame, axis_groups: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    enriched = df.copy()
    for output_col, (x_col, y_col, z_col) in axis_groups.items():
        enriched[output_col] = np.sqrt(
            enriched[x_col].astype(float) ** 2
            + enriched[y_col].astype(float) ** 2
            + enriched[z_col].astype(float) ** 2
        )
    return enriched


def _rolling_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)
    if np.isnan(y).any():
        return np.nan
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if np.isclose(denom, 0.0):
        return np.nan
    return float(np.dot(y - y.mean(), x_centered) / denom)


def build_rolling_feature_matrix(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for col in columns:
        for window in windows:
            roll = df[col].rolling(window=window, min_periods=1)
            part = pd.DataFrame(
                {
                    f"{col}_mean_{window}": roll.mean(),
                    f"{col}_std_{window}": roll.std(),
                    f"{col}_min_{window}": roll.min(),
                    f"{col}_max_{window}": roll.max(),
                    f"{col}_slope_{window}": roll.apply(_rolling_slope, raw=True),
                },
                index=df.index,
            )
            parts.append(part)
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def compute_frequency_summary(
    df: pd.DataFrame,
    columns: list[str],
    sample_rate_hz: float | None,
    top_n: int = 3,
) -> dict[str, dict[str, float | list[float]]]:
    if sample_rate_hz is None or sample_rate_hz <= 0:
        return {}

    summary: dict[str, dict[str, float | list[float]]] = {}
    for col in columns:
        signal = df[col].dropna().astype(float).values
        if len(signal) < 4:
            continue
        centered = signal - np.mean(signal)
        fft_vals = np.fft.rfft(centered)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
        if len(power) <= 1:
            continue

        positive_power = power[1:]
        positive_freqs = freqs[1:]
        top_idx = np.argsort(positive_power)[-top_n:][::-1]
        norm_power = positive_power / (np.sum(positive_power) + 1e-12)
        spectral_entropy = -np.sum(norm_power * np.log(norm_power + 1e-12))

        summary[col] = {
            "dominant_frequencies_hz": positive_freqs[top_idx].round(6).tolist(),
            "dominant_powers": positive_power[top_idx].round(6).tolist(),
            "spectral_entropy": float(np.round(spectral_entropy, 6)),
        }
    return summary
