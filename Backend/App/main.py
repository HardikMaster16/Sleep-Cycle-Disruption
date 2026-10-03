from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictionRequest, PredictionResponse
from .services.ml_service import predict, predict_csv


app = FastAPI(
    title="Sleep Cycle Disruption API",
    description=(
        "Backend API for detecting anomalous movement "
        "patterns during sleep."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://sleep-cycle-disruption-1.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Basic routes
# --------------------------------------------------

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


# --------------------------------------------------
# CSV prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV accelerometer file.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty.",
            )

        predictions = predict_csv(
            BytesIO(contents)
        )

        return {
            "filename": file.filename,
            "predictions": predictions,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Movement analysis failed: {str(exc)}",
        )


# --------------------------------------------------
# Direct feature prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict/features",
    response_model=PredictionResponse,
)
def prediction_from_features(
    request: PredictionRequest,
):

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