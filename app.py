"""
app.py - FastAPI application for the Software Bug Prediction service.

Architecture: XGBoost + LightGBM + RandomForest -> LogisticRegression
              meta-learner -> isotonic calibration -> threshold 0.475
"""

import time
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from inference import predict_single, get_feature_names, get_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Software Bug Prediction API",
    description=(
        "Predicts whether a software module is buggy using a stacking ensemble "
        "of XGBoost, LightGBM, and RandomForest with isotonic probability calibration."
    ),
    version="2.0.0",
)


@app.middleware("http")
async def log_response_time(request: Request, call_next):
    """Log response time in milliseconds for every request."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    logger.info("%s %s  %d  %.2f ms",
                request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


@app.get("/")
def home():
    return {
        "message":        "Software Bug Prediction API is Running",
        "model":          "Stacking Ensemble (XGBoost + LightGBM + RandomForest)",
        "model_features": len(get_feature_names()),
        "threshold":      get_threshold(),
    }


@app.get("/features")
def features():
    """Return the list of features the model expects."""
    return {"features": get_feature_names(), "count": len(get_feature_names())}


@app.post("/predict")
def predict(data: dict):
    """
    Predict whether a software module is buggy.

    Accepts a JSON object of CK / Halstead metric values.
    Missing features are padded to 0. Unknown keys are ignored.
    Non-numeric values for numeric fields are coerced to 0 with a warning
    (returns 200 with valid prediction, not 500).
    """
    # Sanitise input: coerce non-numeric values to 0 instead of crashing
    clean = {}
    bad_keys = []
    for k, v in data.items():
        if isinstance(v, (int, float)):
            clean[k] = v
        else:
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                clean[k] = 0.0
                bad_keys.append(k)

    try:
        result = predict_single(clean)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    if bad_keys:
        result["warnings"] = (
            f"Non-numeric values coerced to 0 for: {bad_keys}"
        )

    return result
