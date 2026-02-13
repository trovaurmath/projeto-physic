# Kit Profissional de Analise de Series Temporais de Sensores

Kit em Python para dados de sensores como temperatura, pressão, IMU, strain gauge e magnetômetro, com pipeline limpo para:

- ingestão de CSV com timestamp,
- validação estrutural da série temporal,
- limpeza de dados (faltantes, outliers, filtro passa-baixa),
- engenharia de features temporal e frequencial,
- detecção de anomalias univariadas e multivariadas.

## Instalação

```bash
pip install -e .
```

## Uso rápido (CLI)

```bash
sensor-ts-kit dados_sensores.csv \
  --timestamp-col timestamp \
  --sensor-cols temp,pressao,acc_x,acc_y,acc_z,strain,mag_x,mag_y,mag_z \
  --resample-rule 100ms \
  --rolling-windows 2s,10s,60s \
  --outlier-method mad \
  --outlier-threshold 3.5 \
  --lowpass-cutoff-hz 8 \
  --output-dir saida_analise
```

Saídas:

- `saida_analise/cleaned_data.csv`
- `saida_analise/feature_matrix.csv`
- `saida_analise/anomaly_flags.csv`
- `saida_analise/diagnostics.json`

## Uso programático

```python
from sensor_ts_kit.io import load_sensor_csv
from sensor_ts_kit.models import PipelineConfig, PreprocessConfig, FeatureConfig, AnomalyConfig
from sensor_ts_kit.pipeline import SensorAnalysisPipeline

raw = load_sensor_csv("dados_sensores.csv", timestamp_col="timestamp")

config = PipelineConfig(
    preprocess=PreprocessConfig(
        timestamp_col="timestamp",
        sensor_columns=["temp", "pressure", "acc_x", "acc_y", "acc_z", "strain"],
        resample_rule="100ms",
        outlier_method="mad",
        outlier_threshold=3.5,
        lowpass_cutoff_hz=8.0,
    ),
    features=FeatureConfig(rolling_windows=["2s", "10s", "60s"]),
    anomaly=AnomalyConfig(univariate_threshold=3.5, multivariate_quantile=0.995),
)

pipeline = SensorAnalysisPipeline(config)
result = pipeline.run(raw)

print(result.cleaned_data.head())
print(result.feature_matrix.head())
print(result.anomaly_flags.tail())
print(result.diagnostics)
```

## Arquitetura

- `sensor_ts_kit/io.py`: leitura e persistência de resultados.
- `sensor_ts_kit/validation.py`: normalização e validação temporal.
- `sensor_ts_kit/cleaning.py`: interpolação, remoção de outliers e filtro digital.
- `sensor_ts_kit/features.py`: features rolling e resumo espectral FFT.
- `sensor_ts_kit/anomaly.py`: anomalias univariadas e multivariadas robustas.
- `sensor_ts_kit/pipeline.py`: orquestração completa com configuração central.
