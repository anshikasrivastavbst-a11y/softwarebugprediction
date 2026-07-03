"""
inference.py - Shared prediction logic for the stacking ensemble.

Pipeline:
    raw input
        -> preprocess_data
        -> RobustScaler  (fitted on all 38 features)
        -> feature selection to 37 features
        -> XGBoost + LightGBM + RandomForest  (base learners)
        -> LogisticRegression                 (meta-learner)
        -> IsotonicRegression                 (calibration)
        -> threshold -> final prediction
"""

import os
import numpy as np
import pandas as pd
import joblib

from preprocessing import preprocess_data

_BASE = os.path.dirname(os.path.abspath(__file__))

REQUIRED = ["model.pkl", "scaler.pkl", "threshold.pkl", "selected_features.pkl"]
for _f in REQUIRED:
    if not os.path.exists(os.path.join(_BASE, _f)):
        raise FileNotFoundError(f"{_f} not found - run train.py first.")

_ensemble          = joblib.load(os.path.join(_BASE, "model.pkl"))
_scaler            = joblib.load(os.path.join(_BASE, "scaler.pkl"))
_threshold         = joblib.load(os.path.join(_BASE, "threshold.pkl"))
_selected_features = joblib.load(os.path.join(_BASE, "selected_features.pkl"))

# All 38 features the scaler was fitted on (includes unified_loc)
_scaler_features = list(_scaler.feature_names_in_)

_xgb     = _ensemble["xgb"]
_lgbm    = _ensemble["lgbm"]
_rf      = _ensemble["rf"]
_meta_lr = _ensemble["meta_lr"]
_iso_cal = _ensemble["iso_cal"]

print(f"Ensemble loaded | scaler features: {len(_scaler_features)} | "
      f"model features: {len(_selected_features)} | threshold: {_threshold:.3f}")


def get_feature_names() -> list:
    return list(_selected_features)


def get_threshold() -> float:
    return float(_threshold)


def _prepare(raw_df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess -> scale (38 features) -> select (37 features).
    Scaler was fitted on 38 features before selection, so we scale
    all 38 first then keep only the 37 selected ones.
    """
    df = preprocess_data(raw_df, training=False)

    # Pad ALL 38 scaler features - missing ones become 0
    for col in _scaler_features:
        if col not in df.columns:
            df[col] = 0.0

    df = df[_scaler_features]

    # Replace inf / -inf and clip extremes to prevent scaler overflow
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df = df.clip(-1e9, 1e9)

    # Scale with all 38 features
    scaled = pd.DataFrame(
        _scaler.transform(df),
        columns=_scaler_features
    )

    # Return only the 37 selected features
    return scaled[_selected_features].values


def predict_dataframe(raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Run stacking ensemble on a DataFrame.

    - An empty dict {} creates a 1-row all-zero DataFrame -> valid prediction.
    - A truly empty DataFrame (0 rows, named columns) -> returns empty arrays.
    - A DataFrame with 0 rows AND 0 columns -> returns empty arrays.
    """
    # Truly empty: no rows
    if len(raw_df) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # A DataFrame built from [{}] has 1 row but 0 named columns.
    # _prepare will pad all features to 0, which is valid all-zero input.
    arr    = _prepare(raw_df)
    xgb_p  = _xgb.predict_proba(arr)[:, 1]
    lgbm_p = _lgbm.predict_proba(arr)[:, 1]
    rf_p   = _rf.predict_proba(arr)[:, 1]
    raw_p  = _meta_lr.predict_proba(np.column_stack([xgb_p, lgbm_p, rf_p]))[:, 1]
    probs  = _iso_cal.transform(raw_p)
    preds  = (probs >= _threshold).astype(int)
    return preds, probs


def predict_single(raw_dict: dict) -> dict:
    """Run stacking ensemble on a single feature dict."""
    df = pd.DataFrame([raw_dict])
    preds, probs = predict_dataframe(df)
    prob       = float(probs[0])
    prediction = int(preds[0])
    return {
        "prediction":      prediction,
        "status":          "Buggy" if prediction == 1 else "Not Buggy",
        "bug_probability": round(prob, 4),
        "threshold_used":  round(_threshold, 4),
    }
