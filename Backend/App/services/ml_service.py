from pathlib import Path
import sys

# --------------------------------------------------
# Locate project root
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

ML_SRC_DIR = PROJECT_DIR / "ML" / "src"

if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))


# --------------------------------------------------
# Import Member 1's inference code
# --------------------------------------------------

from inference import predict_anomaly


# --------------------------------------------------
# Public backend function
# --------------------------------------------------

def predict(features: dict) -> dict:
    """
    Send validated feature data to the ML inference layer.

    The backend does not perform preprocessing, scaling,
    EIF scoring, or threshold calculation itself.
    """

    result = predict_anomaly(features)

    if not result:
        raise RuntimeError("ML inference returned no prediction.")

    return result[0]