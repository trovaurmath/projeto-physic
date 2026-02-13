from sensor_ts_kit.simulation import SimulationConfig, generate_synthetic_sensor_data


def test_generate_synthetic_sensor_data_schema():
    data, anomaly = generate_synthetic_sensor_data(
        SimulationConfig(duration_seconds=2.0, sample_rate_hz=10.0, anomaly_probability=0.2, seed=7)
    )

    expected_cols = {
        "temperature_c",
        "pressure_kpa",
        "imu_acc_x",
        "imu_acc_y",
        "imu_acc_z",
        "imu_gyro_x",
        "imu_gyro_y",
        "imu_gyro_z",
        "strain_ue",
        "mag_x",
        "mag_y",
        "mag_z",
    }
    assert set(data.columns) == expected_cols
    assert len(data) == 20
    assert anomaly.dtype.name == "bool"
    assert anomaly.index.equals(data.index)
