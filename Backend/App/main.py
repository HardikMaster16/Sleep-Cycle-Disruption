from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]

from .schemas import PredictionRequest, PredictionResponse
from .services.ml_service import predict

app = FastAPI(
    title="Sleep Cycle Disruption API",
    description=(
        "Backend API for detecting anomalous movement "
        "patterns during sleep."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Sleep Cycle Disruption API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ml_service": "available",
    }


@app.post("/predict", response_model=PredictionResponse)
def prediction(request: PredictionRequest):

    try:
        result = predict(
            request.model_dump()
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )