from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_sensor_csv, save_analysis
from .models import AnomalyConfig, FeatureConfig, PipelineConfig, PreprocessConfig
from .pipeline import SensorAnalysisPipeline


def _parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensor-ts-kit",
        description="Pipeline profissional para análise de séries temporais de sensores.",
    )
    parser.add_argument("input_csv", type=Path, help="Arquivo CSV de entrada.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_analysis"),
        help="Pasta para salvar resultados.",
    )
    parser.add_argument("--timestamp-col", default="timestamp", help="Nome da coluna de timestamp.")
    parser.add_argument(
        "--sensor-cols",
        default="",
        help="Lista de colunas de sensores separadas por virgula. Ex: temp,pressure,acc_x",
    )
    parser.add_argument("--resample-rule", default=None, help="Regra de resample Pandas. Ex: 1S, 100ms")
    parser.add_argument(
        "--rolling-windows",
        default="5s,30s",
        help="Janelas de rolling features separadas por virgula. Ex: 5s,30s,120s",
    )
    parser.add_argument(
        "--outlier-method",
        default="mad",
        choices=["mad", "zscore", "iqr"],
        help="Método de detecção de outliers.",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.5,
        help="Limite para detecção de outliers.",
    )
    parser.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        default=None,
        help="Cutoff (Hz) para filtro passa-baixa (opcional).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sensor_columns = _parse_csv_list(args.sensor_cols)
    rolling_windows = _parse_csv_list(args.rolling_windows)

    preprocess = PreprocessConfig(
        timestamp_col=args.timestamp_col,
        sensor_columns=sensor_columns,
        resample_rule=args.resample_rule,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
    )
    feature = FeatureConfig(rolling_windows=rolling_windows)
    anomaly = AnomalyConfig(univariate_threshold=args.outlier_threshold)
    config = PipelineConfig(preprocess=preprocess, features=feature, anomaly=anomaly)

    raw = load_sensor_csv(
        args.input_csv,
        timestamp_col=args.timestamp_col,
        sensor_columns=sensor_columns if sensor_columns else None,
    )
    result = SensorAnalysisPipeline(config).run(raw)
    save_analysis(result, args.output_dir)

    print(f"Análise finalizada. Arquivos salvos em: {args.output_dir.resolve()}")
    print("Diagnóstico resumido:")
    for key, value in result.diagnostics.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
