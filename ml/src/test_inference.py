from inference import predict_anomaly


sample = {
    "magnitude_range": 0.10,
    "magnitude_max": 1.08,
    "magnitude_min": 0.98,
    "dynamic_mean": 0.008,
    "dynamic_std": 0.005,
    "dynamic_max": 0.05,
    "movement_change_mean": 0.002,
    "movement_change_std": 0.003,
    "movement_change_max": 0.04,
    "movement_energy": 0.0003,
    "strong_movement_ratio": 0.004,
    "x_std": 0.01,
    "y_std": 0.015,
    "z_std": 0.014
}


result = predict_anomaly(sample)

print("\nPrediction:")
print(result)