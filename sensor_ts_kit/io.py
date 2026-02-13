from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import AnalysisResult


def load_sensor_csv(
    path: str | Path,
    timestamp_col: str = "timestamp",
    sensor_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found.")

    if sensor_columns is None:
        sensor_cols = [c for c in df.columns if c != timestamp_col]
    else:
        sensor_cols = list(sensor_columns)
        missing = [c for c in sensor_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing sensor columns: {missing}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[sensor_cols]


def save_analysis(result: AnalysisResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.cleaned_data.to_csv(output / "cleaned_data.csv")
    result.feature_matrix.to_csv(output / "feature_matrix.csv")
    result.anomaly_flags.to_csv(output / "anomaly_flags.csv")
    with (output / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(result.diagnostics, f, ensure_ascii=False, indent=2, default=str)
