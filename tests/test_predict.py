"""
Unit tests for the stacking ensemble prediction pipeline.
Uses inference.py as the single source of truth.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── Skip entire module if artifacts not yet generated ─────────────────────────
for _art in ["model.pkl", "scaler.pkl", "threshold.pkl", "selected_features.pkl"]:
    if not os.path.exists(_art):
        pytest.skip(f"{_art} missing — run train.py first", allow_module_level=True)

from inference import (
    predict_dataframe, predict_single,
    get_feature_names, get_threshold,
)

FEATURES  = get_feature_names()
THRESHOLD = get_threshold()
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Artifacts
# ─────────────────────────────────────────────────────────────────────────────
class TestArtifacts:
    def test_features_non_empty(self):
        assert len(FEATURES) > 0

    def test_features_are_strings(self):
        assert all(isinstance(f, str) for f in FEATURES)

    def test_threshold_in_range(self):
        assert 0.0 < THRESHOLD < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Output types
# ─────────────────────────────────────────────────────────────────────────────
class TestOutputTypes:
    def test_probability_is_float(self):
        result = predict_single({})
        assert isinstance(result["bug_probability"], float)

    def test_probability_in_range(self):
        result = predict_single({})
        assert 0.0 <= result["bug_probability"] <= 1.0

    def test_prediction_is_binary(self):
        result = predict_single({})
        assert result["prediction"] in (0, 1)

    def test_status_is_valid(self):
        result = predict_single({})
        assert result["status"] in ("Buggy", "Not Buggy")

    def test_status_matches_prediction(self):
        result = predict_single({})
        expected = "Buggy" if result["prediction"] == 1 else "Not Buggy"
        assert result["status"] == expected

    def test_batch_output_length(self):
        df = pd.DataFrame([{} for _ in range(5)])
        preds, probs = predict_dataframe(df)
        assert len(preds) == 5
        assert len(probs) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 3. Threshold behaviour
# ─────────────────────────────────────────────────────────────────────────────
class TestThreshold:
    def test_threshold_used_in_response(self):
        result = predict_single({})
        assert abs(result["threshold_used"] - THRESHOLD) < 1e-6

    def test_probability_consistent_with_prediction(self):
        result = predict_single({"wmc": 200, "cbo": 150, "loc": 5000})
        if result["bug_probability"] >= result["threshold_used"]:
            assert result["prediction"] == 1
        else:
            assert result["prediction"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Missing / extra features
# ─────────────────────────────────────────────────────────────────────────────
class TestRobustness:
    def test_empty_input_runs(self):
        result = predict_single({})
        assert result["prediction"] in (0, 1)

    def test_partial_features_runs(self):
        half = {f: 1.0 for f in FEATURES[:len(FEATURES)//2]}
        result = predict_single(half)
        assert result["prediction"] in (0, 1)

    def test_unknown_features_ignored(self):
        result = predict_single({"unknown_xyz": 999})
        assert result["prediction"] in (0, 1)

    def test_negative_values_handled(self):
        row = {f: -1.0 for f in FEATURES}
        result = predict_single(row)
        assert result["prediction"] in (0, 1)

    def test_zero_values_handled(self):
        row = {f: 0 for f in FEATURES}
        result = predict_single(row)
        assert result["prediction"] in (0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Target leakage guard
# ─────────────────────────────────────────────────────────────────────────────
class TestTargetLeakGuard:
    @pytest.mark.parametrize("target_col", ["bug", "defects", "label", "target"])
    def test_target_col_does_not_affect_prediction(self, target_col):
        row_clean = {f: 5.0 for f in FEATURES}
        row_leak  = dict(row_clean)
        row_leak[target_col] = 99999
        r1 = predict_single(row_clean)
        r2 = predict_single(row_leak)
        assert r1["prediction"] == r2["prediction"]
        assert abs(r1["bug_probability"] - r2["bug_probability"]) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
class TestReproducibility:
    def test_same_input_same_output(self):
        row = {f: 5.0 for f in FEATURES}
        r1 = predict_single(row)
        r2 = predict_single(row)
        assert r1["prediction"] == r2["prediction"]
        assert abs(r1["bug_probability"] - r2["bug_probability"]) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 7. Real dataset predictions
# ─────────────────────────────────────────────────────────────────────────────
class TestRealDataPrediction:
    def _run(self, fname):
        df = pd.read_csv(os.path.join(DATASET_DIR, fname))
        preds, probs = predict_dataframe(df)
        return preds, probs, len(df)

    def test_ant_predictions_valid(self):
        preds, probs, n = self._run("ant-1.7 (1).csv")
        assert len(preds) == n
        assert set(preds).issubset({0, 1})
        assert np.all((probs >= 0) & (probs <= 1))

    def test_kc1_predictions_valid(self):
        preds, probs, n = self._run("kc1.csv")
        assert len(preds) == n
        assert set(preds).issubset({0, 1})

    def test_pc1_predictions_valid(self):
        preds, probs, n = self._run("pc1.csv")
        assert len(preds) == n
        assert set(preds).issubset({0, 1})

    def test_both_classes_predicted(self):
        preds, _, _ = self._run("kc1.csv")
        assert 0 in preds and 1 in preds

    def test_target_col_stripped_in_prediction(self):
        df = pd.read_csv(os.path.join(DATASET_DIR, "ant-1.7 (1).csv"))
        preds, probs = predict_dataframe(df)
        assert len(preds) == len(df)
        assert set(preds).issubset({0, 1})

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["wmc", "loc"])
        result = predict_dataframe(df)
        assert len(result[0]) == 0
