from __future__ import annotations

import pandas as pd

from .anomaly import detect_multivariate_anomalies, detect_univariate_anomalies
from .cleaning import interpolate_missing, lowpass_filter, replace_outliers_with_nan
from .features import build_rolling_feature_matrix, compute_frequency_summary
from .models import AnalysisResult, PipelineConfig
from .validation import estimate_sample_rate_hz, infer_sensor_columns, prepare_time_index, validate_sensor_dataframe


class SensorAnalysisPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, raw_df: pd.DataFrame) -> AnalysisResult:
        preprocess_cfg = self.config.preprocess
        feature_cfg = self.config.features
        anomaly_cfg = self.config.anomaly

        prepared = prepare_time_index(
            raw_df,
            timestamp_col=preprocess_cfg.timestamp_col,
            drop_na_timestamps=preprocess_cfg.drop_na_timestamps,
            duplicate_strategy=preprocess_cfg.duplicate_strategy,
        )

        sensor_columns = infer_sensor_columns(prepared, preprocess_cfg.sensor_columns)
        validate_sensor_dataframe(
            prepared,
            sensor_columns=sensor_columns,
            enforce_monotonic=preprocess_cfg.enforce_monotonic_time,
        )

        working = prepared[sensor_columns].astype(float).copy()
        rows_before = len(working)

        if preprocess_cfg.resample_rule:
            working = working.resample(preprocess_cfg.resample_rule).mean()

        working = interpolate_missing(
            working,
            columns=sensor_columns,
            method=preprocess_cfg.interpolation_method,
            limit=preprocess_cfg.interpolation_limit,
        )

        working, outlier_mask = replace_outliers_with_nan(
            working,
            columns=sensor_columns,
            method=preprocess_cfg.outlier_method,
            threshold=preprocess_cfg.outlier_threshold,
        )

        working = interpolate_missing(
            working,
            columns=sensor_columns,
            method=preprocess_cfg.interpolation_method,
            limit=preprocess_cfg.interpolation_limit,
        )

        sample_rate_hz = estimate_sample_rate_hz(working.index)
        if preprocess_cfg.lowpass_cutoff_hz and sample_rate_hz:
            working = lowpass_filter(
                working,
                columns=sensor_columns,
                sample_rate_hz=sample_rate_hz,
                cutoff_hz=preprocess_cfg.lowpass_cutoff_hz,
                order=preprocess_cfg.lowpass_order,
            )

        feature_matrix = build_rolling_feature_matrix(
            working,
            columns=sensor_columns,
            windows=list(feature_cfg.rolling_windows),
        )

        univariate_flags = detect_univariate_anomalies(
            working,
            columns=sensor_columns,
            method=preprocess_cfg.outlier_method,
            threshold=anomaly_cfg.univariate_threshold,
        )
        multivariate = detect_multivariate_anomalies(
            working,
            columns=sensor_columns,
            quantile=anomaly_cfg.multivariate_quantile,
        )
        anomaly_flags = univariate_flags.join(multivariate)

        frequency_summary = {}
        if feature_cfg.include_frequency_features:
            frequency_summary = compute_frequency_summary(
                working,
                columns=sensor_columns,
                sample_rate_hz=sample_rate_hz,
                top_n=feature_cfg.frequency_top_n,
            )

        diagnostics = {
            "rows_before_processing": rows_before,
            "rows_after_processing": len(working),
            "sample_rate_hz_estimate": sample_rate_hz,
            "missing_after_cleaning": working.isna().sum().to_dict(),
            "outliers_replaced": outlier_mask.sum().to_dict(),
            "multivariate_threshold": multivariate.attrs.get("threshold"),
            "frequency_summary": frequency_summary,
        }

        return AnalysisResult(
            cleaned_data=working,
            feature_matrix=feature_matrix,
            anomaly_flags=anomaly_flags,
            diagnostics=diagnostics,
        )
