import numpy as np
import pandas as pd

from sensor_ts_kit.cleaning import interpolate_missing, replace_outliers_with_nan


def test_replace_outliers_and_interpolate():
    idx = pd.date_range("2026-01-01", periods=5, freq="s")
    df = pd.DataFrame({"temp": [20.0, 21.0, 500.0, 23.0, 24.0]}, index=idx)

    clean, mask = replace_outliers_with_nan(df, columns=["temp"], method="mad", threshold=3.5)
    assert mask["temp"].sum() == 1
    assert np.isnan(clean.loc[idx[2], "temp"])

    recovered = interpolate_missing(clean, columns=["temp"], method="time")
    assert not np.isnan(recovered.loc[idx[2], "temp"])
