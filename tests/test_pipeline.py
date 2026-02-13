import numpy as np
import pandas as pd

from sensor_ts_kit.models import FeatureConfig, PipelineConfig, PreprocessConfig
from sensor_ts_kit.pipeline import SensorAnalysisPipeline


def test_pipeline_end_to_end():
    idx = pd.date_range("2026-01-01", periods=20, freq="s")
    df = pd.DataFrame(
        {
            "temp": np.linspace(20, 30, 20),
            "pressure": np.linspace(1000, 1005, 20),
            "strain": np.linspace(0.1, 0.3, 20),
        },
        index=idx,
    )
    df.loc[idx[8], "temp"] = 200.0
    df.loc[idx[10], "pressure"] = np.nan

    config = PipelineConfig(
        preprocess=PreprocessConfig(
            sensor_columns=["temp", "pressure", "strain"],
            outlier_method="mad",
            outlier_threshold=3.0,
            interpolation_method="time",
            lowpass_cutoff_hz=None,
        ),
        features=FeatureConfig(rolling_windows=["3s", "10s"]),
    )

    result = SensorAnalysisPipeline(config).run(df)
    assert not result.cleaned_data.empty
    assert not result.feature_matrix.empty
    assert "temp_anomaly" in result.anomaly_flags.columns
    assert "multivariate_anomaly" in result.anomaly_flags.columns
