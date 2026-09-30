from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports, reportUnusedImport]


class PredictionRequest(BaseModel):
    magnitude_range: float
    magnitude_max: float
    magnitude_min: float

    dynamic_mean: float
    dynamic_std: float
    dynamic_max: float

    movement_change_mean: float
    movement_change_std: float
    movement_change_max: float

    movement_energy: float
    strong_movement_ratio: float

    x_std: float
    y_std: float
    z_std: float


class PredictionResponse(BaseModel):
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    prediction: str