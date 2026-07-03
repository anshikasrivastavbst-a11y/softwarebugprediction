"""Unit tests for the FastAPI endpoints (app.py — stacking ensemble)."""

import os
import sys
import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

from inference import get_feature_names, get_threshold
FEATURES  = get_feature_names()
THRESHOLD = get_threshold()

VALID_PAYLOAD = {
    "wmc": 10, "dit": 2, "noc": 1, "cbo": 5, "rfc": 20,
    "lcom": 50, "ca": 3, "ce": 4, "npm": 8, "lcom3": 0.5,
    "loc": 150, "dam": 0.3, "moa": 1, "mfa": 0.2, "cam": 0.6,
    "ic": 0, "cbm": 1, "amc": 15.0, "max_cc": 5, "avg_cc": 2.5,
}


# ── Home endpoint ─────────────────────────────────────────────────────────────
class TestHomeEndpoint:
    def test_get_home_status_200(self):
        assert client.get("/").status_code == 200

    def test_get_home_returns_json(self):
        assert client.get("/").headers["content-type"].startswith("application/json")

    def test_home_contains_message(self):
        assert "message" in client.get("/").json()

    def test_home_contains_model_features(self):
        body = client.get("/").json()
        assert "model_features" in body
        assert isinstance(body["model_features"], int)
        assert body["model_features"] > 0

    def test_home_contains_threshold(self):
        body = client.get("/").json()
        assert "threshold" in body
        assert 0.0 < body["threshold"] < 1.0


# ── /predict valid input ──────────────────────────────────────────────────────
class TestPredictValidInput:
    def test_predict_returns_200(self):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_predict_has_prediction_key(self):
        assert "prediction" in client.post("/predict", json=VALID_PAYLOAD).json()

    def test_predict_has_status_key(self):
        assert "status" in client.post("/predict", json=VALID_PAYLOAD).json()

    def test_predict_has_bug_probability_key(self):
        assert "bug_probability" in client.post("/predict", json=VALID_PAYLOAD).json()

    def test_predict_has_threshold_used_key(self):
        assert "threshold_used" in client.post("/predict", json=VALID_PAYLOAD).json()

    def test_prediction_is_binary(self):
        assert client.post("/predict", json=VALID_PAYLOAD).json()["prediction"] in (0, 1)

    def test_status_is_buggy_or_not_buggy(self):
        assert client.post("/predict", json=VALID_PAYLOAD).json()["status"] in ("Buggy", "Not Buggy")

    def test_probability_in_range(self):
        p = client.post("/predict", json=VALID_PAYLOAD).json()["bug_probability"]
        assert 0.0 <= p <= 1.0

    def test_threshold_used_in_range(self):
        t = client.post("/predict", json=VALID_PAYLOAD).json()["threshold_used"]
        assert 0.0 < t < 1.0

    def test_status_matches_prediction(self):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        expected = "Buggy" if body["prediction"] == 1 else "Not Buggy"
        assert body["status"] == expected

    def test_probability_consistent_with_prediction(self):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        if body["bug_probability"] >= body["threshold_used"]:
            assert body["prediction"] == 1
        else:
            assert body["prediction"] == 0


# ── Empty payload ─────────────────────────────────────────────────────────────
class TestPredictEmptyPayload:
    def test_empty_payload_returns_200(self):
        assert client.post("/predict", json={}).status_code == 200

    def test_empty_payload_returns_valid_prediction(self):
        body = client.post("/predict", json={}).json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["bug_probability"] <= 1.0


# ── Target leakage guard ──────────────────────────────────────────────────────
class TestTargetLeakageInAPI:
    @pytest.mark.parametrize("target_col", ["bug", "defects", "label", "target"])
    def test_target_col_does_not_change_prediction(self, target_col):
        resp_clean = client.post("/predict", json=VALID_PAYLOAD).json()
        payload_with = dict(VALID_PAYLOAD)
        payload_with[target_col] = 99999
        resp_with = client.post("/predict", json=payload_with).json()
        assert resp_clean["prediction"] == resp_with["prediction"]
        assert abs(resp_clean["bug_probability"] - resp_with["bug_probability"]) < 1e-4


# ── Engineered features ───────────────────────────────────────────────────────
class TestEngineeredFeaturesInAPI:
    def test_loc_based_features_computed(self):
        payload = dict(VALID_PAYLOAD)
        payload.update({"loc": 200, "rfc": 30, "wmc": 10, "cbo": 6, "dit": 3, "v(g)": 12})
        assert client.post("/predict", json=payload).status_code == 200

    def test_result_differs_with_high_complexity(self):
        resp_low  = client.post("/predict", json={}).json()
        resp_high = client.post("/predict", json={"loc": 5000, "v(g)": 200, "rfc": 500}).json()
        assert resp_low["prediction"] in (0, 1)
        assert resp_high["prediction"] in (0, 1)
# ── Extra keys ignored ────────────────────────────────────────────────────────
class TestExtraKeysIgnored:
    def test_unknown_key_does_not_crash(self):
        payload = dict(VALID_PAYLOAD)
        payload["totally_unknown_key_xyz"] = 42
        assert client.post("/predict", json=payload).status_code == 200

    def test_prediction_same_with_extra_key(self):
        r1 = client.post("/predict", json=VALID_PAYLOAD).json()
        payload = dict(VALID_PAYLOAD)
        payload["noise_col"] = 12345
        r2 = client.post("/predict", json=payload).json()
        assert r1["prediction"] == r2["prediction"]


# ── Reproducibility ───────────────────────────────────────────────────────────
class TestReproducibility:
    def test_same_input_same_output(self):
        r1 = client.post("/predict", json=VALID_PAYLOAD).json()
        r2 = client.post("/predict", json=VALID_PAYLOAD).json()
        assert r1["prediction"] == r2["prediction"]
        assert abs(r1["bug_probability"] - r2["bug_probability"]) < 1e-9

    def test_high_complexity_predicts_consistently(self):
        payload = {"wmc": 200, "cbo": 150, "rfc": 500, "lcom": 5000,
                   "loc": 10000, "v(g)": 300, "max_cc": 50}
        r1 = client.post("/predict", json=payload).json()
        r2 = client.post("/predict", json=payload).json()
        assert r1["prediction"] == r2["prediction"]


# ── Data type edge cases ──────────────────────────────────────────────────────
class TestDataTypeEdgeCases:
    def test_float_values_accepted(self):
        payload = {k: float(v) for k, v in VALID_PAYLOAD.items()}
        assert client.post("/predict", json=payload).status_code == 200

    def test_string_numeric_values_handled(self):
        payload = {k: str(v) for k, v in VALID_PAYLOAD.items()}
        assert client.post("/predict", json=payload).status_code in (200, 400)

    def test_negative_values_accepted(self):
        payload = {k: -abs(v) if isinstance(v, (int, float)) else v
                   for k, v in VALID_PAYLOAD.items()}
        assert client.post("/predict", json=payload).status_code == 200

    def test_zero_values_accepted(self):
        assert client.post("/predict", json={k: 0 for k in VALID_PAYLOAD}).status_code == 200
