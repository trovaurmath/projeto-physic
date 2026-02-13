from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SimulationConfig:
    duration_seconds: float = 30.0
    sample_rate_hz: float = 50.0
    anomaly_probability: float = 0.01
    seed: int | None = 42


def generate_synthetic_sensor_data(
    config: SimulationConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    cfg = config or SimulationConfig()
    if cfg.duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if cfg.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if not (0.0 <= cfg.anomaly_probability <= 1.0):
        raise ValueError("anomaly_probability must be between 0 and 1.")

    rng = np.random.default_rng(cfg.seed)
    num_samples = max(int(cfg.duration_seconds * cfg.sample_rate_hz), 10)
    t = np.arange(num_samples) / cfg.sample_rate_hz

    temperature_c = 24.0 + 0.8 * np.sin(2 * np.pi * 0.03 * t) + rng.normal(0.0, 0.08, size=num_samples)
    pressure_kpa = 101.3 + 0.25 * np.sin(2 * np.pi * 0.06 * t + 1.2) + rng.normal(0.0, 0.03, size=num_samples)

    imu_acc_x = 0.3 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(0.0, 0.02, size=num_samples)
    imu_acc_y = 0.3 * np.cos(2 * np.pi * 1.2 * t) + rng.normal(0.0, 0.02, size=num_samples)
    imu_acc_z = 9.81 + 0.05 * np.sin(2 * np.pi * 0.8 * t + 0.2) + rng.normal(0.0, 0.02, size=num_samples)

    imu_gyro_x = 0.4 * np.sin(2 * np.pi * 0.9 * t) + rng.normal(0.0, 0.01, size=num_samples)
    imu_gyro_y = 0.35 * np.cos(2 * np.pi * 1.1 * t + 0.3) + rng.normal(0.0, 0.01, size=num_samples)
    imu_gyro_z = 0.2 * np.sin(2 * np.pi * 0.7 * t + 0.5) + rng.normal(0.0, 0.01, size=num_samples)

    strain_ue = 180 + 8 * np.sin(2 * np.pi * 0.04 * t) + 0.6 * t + rng.normal(0.0, 0.5, size=num_samples)
    mag_x = 30 + 0.6 * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0.0, 0.08, size=num_samples)
    mag_y = -12 + 0.5 * np.cos(2 * np.pi * 0.2 * t + 0.5) + rng.normal(0.0, 0.08, size=num_samples)
    mag_z = 44 + 0.7 * np.sin(2 * np.pi * 0.15 * t + 0.8) + rng.normal(0.0, 0.08, size=num_samples)

    data = pd.DataFrame(
        {
            "temperature_c": temperature_c,
            "pressure_kpa": pressure_kpa,
            "imu_acc_x": imu_acc_x,
            "imu_acc_y": imu_acc_y,
            "imu_acc_z": imu_acc_z,
            "imu_gyro_x": imu_gyro_x,
            "imu_gyro_y": imu_gyro_y,
            "imu_gyro_z": imu_gyro_z,
            "strain_ue": strain_ue,
            "mag_x": mag_x,
            "mag_y": mag_y,
            "mag_z": mag_z,
        }
    )
    data.index = pd.date_range("2026-01-01", periods=num_samples, freq=pd.to_timedelta(1 / cfg.sample_rate_hz, unit="s"))

    anomaly_mask = pd.Series(rng.random(num_samples) < cfg.anomaly_probability, index=data.index, name="simulated_anomaly")
    anomaly_indices = np.where(anomaly_mask.values)[0]

    for idx in anomaly_indices:
        span = int(rng.integers(1, 5))
        end = min(idx + span, num_samples)
        target_cols = rng.choice(data.columns, size=int(rng.integers(1, 4)), replace=False)
        for col in target_cols:
            amplitude = float(rng.uniform(3.0, 8.0) * data[col].std())
            sign = -1.0 if rng.random() < 0.5 else 1.0
            data.iloc[idx:end, data.columns.get_loc(col)] += sign * amplitude

    return data, anomaly_mask


def animate_sensor_simulation(
    data: pd.DataFrame,
    anomaly_flags: pd.Series | None = None,
    output_path: str | Path = "sensor_simulation.gif",
    channels: list[str] | None = None,
    fps: int = 20,
    window_seconds: float = 8.0,
    max_frames: int = 600,
    dpi: int = 110,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate animation.") from exc

    if data.empty:
        raise ValueError("data cannot be empty.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    selected = channels or ["temperature_c", "pressure_kpa", "imu_acc_x", "strain_ue", "mag_x"]
    missing = [col for col in selected if col not in data.columns]
    if missing:
        raise KeyError(f"Missing channels in data: {missing}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    working = data[selected].copy()
    if isinstance(working.index, pd.DatetimeIndex):
        x_seconds = (working.index - working.index[0]).total_seconds().to_numpy()
        dt = np.median(np.diff(x_seconds)) if len(x_seconds) > 1 else 1.0
    else:
        x_seconds = np.arange(len(working), dtype=float)
        dt = 1.0

    window_samples = max(int(window_seconds / max(dt, 1e-6)), 1)
    frame_step = max(int(np.ceil(len(working) / max_frames)), 1)
    frame_indices = np.arange(0, len(working), frame_step)
    if frame_indices[-1] != len(working) - 1:
        frame_indices = np.append(frame_indices, len(working) - 1)

    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 2.2 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]

    lines = []
    scatters = []
    y_arrays = [working[col].astype(float).to_numpy() for col in selected]
    flags = anomaly_flags.reindex(working.index).fillna(False).to_numpy() if anomaly_flags is not None else None

    for axis, col, y in zip(axes, selected, y_arrays):
        axis.set_title(col)
        axis.grid(True, alpha=0.25)
        line, = axis.plot([], [], lw=1.7, color="#005f73")
        scatter = axis.scatter([], [], s=24, color="#bb3e03")
        lines.append(line)
        scatters.append(scatter)

        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        margin = (y_max - y_min) * 0.15 if y_max > y_min else 1.0
        axis.set_ylim(y_min - margin, y_max + margin)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()

    def update(frame_i: int):
        current_idx = int(frame_indices[frame_i])
        start = max(0, current_idx - window_samples)
        x = x_seconds[start : current_idx + 1]

        for line, scatter, y, axis in zip(lines, scatters, y_arrays, axes):
            segment = y[start : current_idx + 1]
            line.set_data(x, segment)
            x_end = x_seconds[current_idx]
            x_start = max(0.0, x_end - window_seconds)
            axis.set_xlim(x_start, x_end + dt)

            if flags is not None:
                local_flags = flags[start : current_idx + 1]
                flagged_x = x[local_flags]
                flagged_y = segment[local_flags]
                if len(flagged_x) > 0:
                    scatter.set_offsets(np.column_stack([flagged_x, flagged_y]))
                else:
                    scatter.set_offsets(np.empty((0, 2)))
            else:
                scatter.set_offsets(np.empty((0, 2)))

        return [*lines, *scatters]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    if out_path.suffix.lower() == ".mp4":
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)
    animation.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path
