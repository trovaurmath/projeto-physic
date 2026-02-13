import pandas as pd

from sensor_ts_kit.validation import prepare_time_index


def test_prepare_time_index_with_duplicates_mean():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:01",
            ],
            "temp": [10.0, 14.0, 20.0],
        }
    )

    out = prepare_time_index(df, timestamp_col="timestamp", duplicate_strategy="mean")
    assert len(out) == 2
    assert out.iloc[0]["temp"] == 12.0
