"""
api.py — Alternate FastAPI entry point (thin wrapper around inference module).
Kept for backward compatibility. Use app.py for the primary server.
"""

from fastapi import FastAPI, HTTPException
from inference import predict_single, get_feature_names, get_threshold

app = FastAPI(title="Software Bug Prediction API")


@app.get("/")
def home():
    return {
        "message":         "Software Bug Prediction API is running!",
        "total_features":  len(get_feature_names()),
        "applied_threshold": get_threshold(),
    }


@app.post("/predict")
def predict(data: dict):
    try:
        result = predict_single(data)
        # Map key names to match original api.py response shape
        return {
            "prediction":       result["prediction"],
            "status":           result["status"],
            "probability":      result["bug_probability"],
            "applied_threshold": result["threshold_used"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")
