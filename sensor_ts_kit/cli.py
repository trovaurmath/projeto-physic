from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .io import load_sensor_csv, save_analysis
from .models import AnomalyConfig, FeatureConfig, PipelineConfig, PreprocessConfig
from .pipeline import SensorAnalysisPipeline
from .simulation import SimulationConfig, animate_sensor_simulation, generate_synthetic_sensor_data


def _parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensor-ts-kit",
        description="Kit profissional para analise de series temporais de sensores.",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Executa analise em um CSV real.")
    analyze.add_argument("input_csv", type=Path, help="Arquivo CSV de entrada.")
    analyze.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_analysis"),
        help="Pasta para salvar resultados.",
    )
    analyze.add_argument("--timestamp-col", default="timestamp", help="Nome da coluna de timestamp.")
    analyze.add_argument(
        "--sensor-cols",
        default="",
        help="Lista de colunas de sensores separadas por virgula. Ex: temp,pressure,acc_x",
    )
    analyze.add_argument("--resample-rule", default=None, help="Regra de resample Pandas. Ex: 1S, 100ms")
    analyze.add_argument(
        "--rolling-windows",
        default="5s,30s",
        help="Janelas de rolling features separadas por virgula. Ex: 5s,30s,120s",
    )
    analyze.add_argument(
        "--outlier-method",
        default="mad",
        choices=["mad", "zscore", "iqr"],
        help="Metodo de deteccao de outliers.",
    )
    analyze.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.5,
        help="Limite para deteccao de outliers.",
    )
    analyze.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        default=None,
        help="Cutoff (Hz) para filtro passa-baixa (opcional).",
    )

    simulate = subparsers.add_parser("simulate", help="Gera dados sinteticos e animacao.")
    simulate.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_output"),
        help="Pasta de saida da simulacao.",
    )
    simulate.add_argument(
        "--duration-seconds",
        type=float,
        default=30.0,
        help="Duracao total da simulacao em segundos.",
    )
    simulate.add_argument(
        "--sample-rate-hz",
        type=float,
        default=50.0,
        help="Taxa de amostragem dos sinais sinteticos.",
    )
    simulate.add_argument(
        "--anomaly-probability",
        type=float,
        default=0.01,
        help="Probabilidade de injetar anomalia por amostra.",
    )
    simulate.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente do gerador pseudo-aleatorio.",
    )
    simulate.add_argument(
        "--animation-file",
        default="sensor_simulation.gif",
        help="Arquivo de animacao de saida (.gif ou .mp4).",
    )
    simulate.add_argument(
        "--animation-fps",
        type=int,
        default=20,
        help="FPS da animacao.",
    )
    simulate.add_argument(
        "--window-seconds",
        type=float,
        default=8.0,
        help="Janela deslizante exibida na animacao.",
    )
    simulate.add_argument(
        "--channels",
        default="temperature_c,pressure_kpa,imu_acc_x,strain_ue,mag_x",
        help="Canais para plot animado, separados por virgula.",
    )
    simulate.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Nao executa o pipeline de analise nos dados sinteticos.",
    )
    simulate.add_argument(
        "--rolling-windows",
        default="5s,30s",
        help="Janelas para features, caso analise seja executada.",
    )
    simulate.add_argument(
        "--resample-rule",
        default=None,
        help="Regra de resample no pipeline, caso analise seja executada.",
    )
    simulate.add_argument(
        "--outlier-method",
        default="mad",
        choices=["mad", "zscore", "iqr"],
        help="Metodo de deteccao de outliers para o pipeline.",
    )
    simulate.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.5,
        help="Limite de outlier para o pipeline.",
    )
    simulate.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        default=None,
        help="Cutoff de passa-baixa para o pipeline.",
    )
    return parser


def _build_pipeline_from_args(args: argparse.Namespace) -> PipelineConfig:
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
    return PipelineConfig(preprocess=preprocess, features=feature, anomaly=anomaly)


def _run_analyze(args: argparse.Namespace) -> None:
    config = _build_pipeline_from_args(args)
    sensor_columns = _parse_csv_list(args.sensor_cols)

    raw = load_sensor_csv(
        args.input_csv,
        timestamp_col=args.timestamp_col,
        sensor_columns=sensor_columns if sensor_columns else None,
    )
    result = SensorAnalysisPipeline(config).run(raw)
    save_analysis(result, args.output_dir)

    print(f"Analise finalizada. Arquivos salvos em: {args.output_dir.resolve()}")
    print("Diagnostico resumido:")
    for key, value in result.diagnostics.items():
        print(f"- {key}: {value}")


def _run_simulate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_config = SimulationConfig(
        duration_seconds=args.duration_seconds,
        sample_rate_hz=args.sample_rate_hz,
        anomaly_probability=args.anomaly_probability,
        seed=args.seed,
    )
    data, anomaly_flags = generate_synthetic_sensor_data(sim_config)

    csv_path = output_dir / "simulated_sensor_data.csv"
    data_to_save = data.copy()
    data_to_save["simulated_anomaly"] = anomaly_flags.astype(int)
    data_to_save.index.name = "timestamp"
    data_to_save.to_csv(csv_path)

    channels = _parse_csv_list(args.channels)
    animation_path = output_dir / args.animation_file
    animation_path = animate_sensor_simulation(
        data,
        anomaly_flags=anomaly_flags,
        output_path=animation_path,
        channels=channels,
        fps=args.animation_fps,
        window_seconds=args.window_seconds,
    )

    print(f"Dados sinteticos salvos em: {csv_path.resolve()}")
    print(f"Animacao salva em: {animation_path.resolve()}")

    if args.skip_analysis:
        return

    rolling_windows = _parse_csv_list(args.rolling_windows)
    preprocess = PreprocessConfig(
        sensor_columns=list(data.columns),
        resample_rule=args.resample_rule,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
    )
    feature = FeatureConfig(rolling_windows=rolling_windows)
    anomaly = AnomalyConfig(univariate_threshold=args.outlier_threshold)
    result = SensorAnalysisPipeline(PipelineConfig(preprocess=preprocess, features=feature, anomaly=anomaly)).run(data)

    analysis_dir = output_dir / "analysis"
    save_analysis(result, analysis_dir)
    print(f"Analise da simulacao salva em: {analysis_dir.resolve()}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args_list = list(argv if argv is not None else sys.argv[1:])

    if args_list and args_list[0] not in {"analyze", "simulate", "-h", "--help"}:
        args_list = ["analyze", *args_list]
    if not args_list:
        parser.print_help()
        return

    args = parser.parse_args(args_list)
    if args.command == "simulate":
        _run_simulate(args)
    else:
        _run_analyze(args)


if __name__ == "__main__":
    main()
