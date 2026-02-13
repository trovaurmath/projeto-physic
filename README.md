# Kit Profissional de Analise de Series Temporais de Sensores

Kit em Python para dados de sensores como temperatura, pressao, IMU, strain gauge e magnetometro.

Recursos:
- ingestao de CSV com timestamp,
- validacao estrutural da serie temporal,
- limpeza de dados (faltantes, outliers, filtro passa-baixa),
- engenharia de features temporal e frequencial,
- deteccao de anomalias univariadas e multivariadas,
- simulacao sintetica com animacao GIF/MP4.

## Instalacao

```bash
pip install -e .
```

## CLI

### 1) Analisar CSV real

Compatibilidade direta (comando antigo):

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

Forma explicita:

```bash
sensor-ts-kit analyze dados_sensores.csv --output-dir saida_analise
```

Saidas:
- `saida_analise/cleaned_data.csv`
- `saida_analise/feature_matrix.csv`
- `saida_analise/anomaly_flags.csv`
- `saida_analise/diagnostics.json`

### 2) Simular e gerar animacao

```bash
sensor-ts-kit simulate \
  --duration-seconds 40 \
  --sample-rate-hz 50 \
  --anomaly-probability 0.015 \
  --animation-file sensor_simulation.gif \
  --animation-fps 20 \
  --window-seconds 8 \
  --channels temperature_c,pressure_kpa,imu_acc_x,strain_ue,mag_x \
  --output-dir simulation_output
```

Saidas:
- `simulation_output/simulated_sensor_data.csv`
- `simulation_output/sensor_simulation.gif` (ou `.mp4`)
- `simulation_output/analysis/*` (se nao usar `--skip-analysis`)

## Uso programatico (pipeline)

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
print(result.diagnostics)
```

## Uso programatico (simulacao)

```python
from sensor_ts_kit.simulation import SimulationConfig, generate_synthetic_sensor_data, animate_sensor_simulation

data, anomalies = generate_synthetic_sensor_data(
    SimulationConfig(duration_seconds=20, sample_rate_hz=40, anomaly_probability=0.02, seed=42)
)

animate_sensor_simulation(
    data,
    anomaly_flags=anomalies,
    output_path="sensor_simulation.gif",
    channels=["temperature_c", "pressure_kpa", "imu_acc_x", "strain_ue", "mag_x"],
)
```

## Arquitetura

- `sensor_ts_kit/io.py`: leitura e persistencia de resultados.
- `sensor_ts_kit/validation.py`: normalizacao e validacao temporal.
- `sensor_ts_kit/cleaning.py`: interpolacao, remocao de outliers e filtro digital.
- `sensor_ts_kit/features.py`: features rolling e resumo espectral FFT.
- `sensor_ts_kit/anomaly.py`: anomalias univariadas e multivariadas robustas.
- `sensor_ts_kit/pipeline.py`: orquestracao completa com configuracao central.
- `sensor_ts_kit/simulation.py`: geracao sintetica e animacao dos sinais.
