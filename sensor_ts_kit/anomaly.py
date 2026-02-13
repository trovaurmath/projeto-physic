from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import robust_score


def detect_univariate_anomalies(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "mad",
    threshold: float = 3.5,
) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    for col in columns:
        score = robust_score(df[col], method=method).abs()
        flags[f"{col}_anomaly"] = (score > threshold).fillna(False)
    return flags


def detect_multivariate_anomalies(
    df: pd.DataFrame,
    columns: list[str],
    quantile: float = 0.995,
) -> pd.DataFrame:
    subset = df[columns].astype(float)
    med = subset.median()
    mad = (subset - med).abs().median()
    mad = mad.replace(0.0, np.nan)

    robust_z = 0.6745 * (subset - med) / mad
    score = np.sqrt((robust_z.fillna(0.0) ** 2).sum(axis=1))

    threshold = float(score.quantile(quantile)) if len(score) else np.inf
    out = pd.DataFrame(index=df.index)
    out["multivariate_score"] = score
    out["multivariate_anomaly"] = (score > threshold).fillna(False)
    out.attrs["threshold"] = threshold
    return out
