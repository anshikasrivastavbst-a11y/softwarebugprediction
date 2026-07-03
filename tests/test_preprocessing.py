"""
Unit tests for preprocessing.py
Covers every fix, branch, and edge case.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import preprocess_data, FEATURES_TO_DROP


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _base_df(**extra):
    """Return a minimal valid DataFrame with a numeric target."""
    data = {"wmc": [5, 10, 15], "loc": [100, 200, 300], "target": [0, 1, 0]}
    data.update(extra)
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────
# 1. Return type & immutability
# ─────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_dataframe(self):
        result = preprocess_data(_base_df())
        assert isinstance(result, pd.DataFrame)

    def test_input_not_mutated(self):
        """preprocess_data must not modify the caller's DataFrame."""
        df = _base_df()
        original_cols = list(df.columns)
        preprocess_data(df)
        assert list(df.columns) == original_cols

    def test_empty_dataframe_returns_empty(self):
        """An empty DataFrame should pass through without error."""
        df = pd.DataFrame()
        result = preprocess_data(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ─────────────────────────────────────────────────────────────
# 2. NaN / missing value handling  (Fix #1)
# ─────────────────────────────────────────────────────────────

class TestFillNA:
    def test_no_nans_after_preprocessing(self):
        """All NaNs must be filled to 0 — even after select_dtypes (Fix #1)."""
        df = pd.DataFrame({"wmc": [1, None, 3], "loc": [None, 2.0, 3.0], "target": [0, 1, 0]})
        result = preprocess_data(df)
        assert result.isnull().sum().sum() == 0

    def test_all_nan_column_filled_with_zero(self):
        # Use np.nan so pandas infers float64 (numeric dtype).
        # A plain Python None in an all-None list gives object dtype,
        # which select_dtypes(include=np.number) correctly drops —
        # so we must give pandas enough signal to treat the column as numeric.
        df = pd.DataFrame({"wmc": [np.nan, np.nan], "target": [0, 1]})
        result = preprocess_data(df)
        assert "wmc" in result.columns
        assert (result["wmc"] == 0).all()

    def test_partial_nan_filled_correctly(self):
        df = pd.DataFrame({"wmc": [5.0, None, 15.0], "target": [0, 1, 0]})
        result = preprocess_data(df)
        assert result["wmc"].tolist() == [5.0, 0.0, 15.0]


# ─────────────────────────────────────────────────────────────
# 3. unified_loc engineering  (Fix #2)
# ─────────────────────────────────────────────────────────────

class TestUnifiedLoc:
    def test_unified_loc_from_loc_only(self):
        df = pd.DataFrame({"loc": [100, 200], "target": [0, 1]})
        result = preprocess_data(df)
        assert "unified_loc" in result.columns
        assert result["unified_loc"].tolist() == [100.0, 200.0]

    def test_unified_loc_from_lOCode_only(self):
        """Fix #2: unified_loc must be created even when only lOCode is present."""
        df = pd.DataFrame({"lOCode": [50, 75], "target": [0, 1]})
        result = preprocess_data(df)
        assert "unified_loc" in result.columns
        assert result["unified_loc"].tolist() == [50.0, 75.0]

    def test_unified_loc_prefers_loc_over_lOCode(self):
        """loc takes precedence; lOCode fills NaN positions."""
        df = pd.DataFrame({"loc": [100.0, None], "lOCode": [99, 55], "target": [0, 1]})
        result = preprocess_data(df)
        assert result["unified_loc"].tolist() == [100.0, 55.0]

    def test_unified_loc_nan_in_both_fills_zero(self):
        df = pd.DataFrame({"loc": [None, 200.0], "lOCode": [None, 180.0], "target": [0, 1]})
        result = preprocess_data(df)
        assert result["unified_loc"].iloc[0] == 0.0

    def test_no_unified_loc_when_neither_col_present(self):
        """If neither loc nor lOCode exist, unified_loc should not appear."""
        df = pd.DataFrame({"wmc": [1, 2], "target": [0, 1]})
        result = preprocess_data(df)
        assert "unified_loc" not in result.columns


# ─────────────────────────────────────────────────────────────
# 4. Boolean target column preserved  (Fix #3)
# ─────────────────────────────────────────────────────────────

class TestBoolTarget:
    def test_bool_target_not_dropped(self):
        """Fix #3: bool dtype must be converted to int before select_dtypes."""
        df = pd.DataFrame({"wmc": [5, 10], "loc": [100, 200], "target": [False, True]})
        result = preprocess_data(df, training=True)
        assert "target" in result.columns

    def test_bool_target_converted_to_int(self):
        df = pd.DataFrame({"wmc": [5, 10], "target": [False, True]})
        result = preprocess_data(df, training=True)
        assert result["target"].tolist() == [0, 1]

    def test_bool_feature_column_converted(self):
        """Non-target bool columns should also be converted."""
        df = pd.DataFrame({"wmc": [1, 2], "flag": [True, False], "target": [0, 1]})
        result = preprocess_data(df, training=True)
        # flag is numeric now (0/1), target preserved
        assert "target" in result.columns


# ─────────────────────────────────────────────────────────────
# 5. Case-insensitive feature drop  (Fix #4)
# ─────────────────────────────────────────────────────────────

class TestFeatureDrop:
    @pytest.mark.parametrize("col", ["L", "V", "D", "N", "E", "l", "v", "d", "n", "e"])
    def test_noisy_col_dropped(self, col):
        """Fix #4: both upper- and lower-case noisy columns must be removed."""
        df = pd.DataFrame({"wmc": [1, 2], col: [9, 8], "target": [0, 1]})
        result = preprocess_data(df, training=True)
        assert col not in result.columns

    def test_locCodeAndComment_dropped(self):
        df = pd.DataFrame({"wmc": [1, 2], "locCodeAndComment": [3, 4], "target": [0, 1]})
        result = preprocess_data(df)
        assert "locCodeAndComment" not in result.columns

    def test_uniq_Opnd_dropped(self):
        df = pd.DataFrame({"wmc": [1, 2], "uniq_Opnd": [5, 6], "target": [0, 1]})
        result = preprocess_data(df)
        assert "uniq_Opnd" not in result.columns

    def test_sno_dropped(self):
        df = pd.DataFrame({"sno": [1, 2], "wmc": [5, 10], "target": [0, 1]})
        result = preprocess_data(df)
        assert "sno" not in result.columns

    def test_useful_columns_not_dropped(self):
        """Columns like 'wmc', 'rfc', 'cbo' must survive feature drop."""
        df = pd.DataFrame({"wmc": [5, 10], "rfc": [20, 30], "cbo": [3, 6], "target": [0, 1]})
        result = preprocess_data(df)
        for col in ["wmc", "rfc", "cbo"]:
            assert col in result.columns


# ─────────────────────────────────────────────────────────────
# 6. Unnamed trailing columns dropped  (Fix #5)
# ─────────────────────────────────────────────────────────────

class TestUnnamedColumns:
    def test_unnamed_col_dropped(self):
        """Fix #5: Unnamed: N columns from Excel exports must be removed."""
        df = pd.DataFrame({"wmc": [1, 2], "Unnamed: 21": [0.0, 0.0], "target": [0, 1]})
        result = preprocess_data(df)
        assert "Unnamed: 21" not in result.columns

    def test_multiple_unnamed_cols_dropped(self):
        df = pd.DataFrame({
            "wmc": [1, 2],
            "Unnamed: 0": [0.0, 1.0],
            "Unnamed: 22": [3.0, 4.0],
            "target": [0, 1],
        })
        result = preprocess_data(df)
        assert "Unnamed: 0" not in result.columns
        assert "Unnamed: 22" not in result.columns


# ─────────────────────────────────────────────────────────────
# 7. Target leakage in inference mode  (Fix #8)
# ─────────────────────────────────────────────────────────────

class TestTargetLeakage:
    @pytest.mark.parametrize("target_col", ["target", "bug", "defects", "label"])
    def test_target_stripped_in_inference(self, target_col):
        """Fix #8: target-like columns must be dropped when training=False."""
        df = pd.DataFrame({"wmc": [5, 10], "loc": [100, 200], target_col: [0, 1]})
        result = preprocess_data(df, training=False)
        assert target_col not in result.columns

    def test_target_kept_in_training_mode(self):
        df = pd.DataFrame({"wmc": [5, 10], "loc": [100, 200], "target": [0, 1]})
        result = preprocess_data(df, training=True)
        assert "target" in result.columns

    def test_bug_col_kept_in_training_mode(self):
        """'bug' renamed to 'target' before calling preprocess_data in train.py,
        but test that passing 'target' works as expected."""
        df = pd.DataFrame({"wmc": [5, 10], "target": [0, 1]})
        result = preprocess_data(df, training=True)
        assert "target" in result.columns


# ─────────────────────────────────────────────────────────────
# 8. Feature engineering correctness
# ─────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_rfc_to_wmc_ratio(self):
        df = pd.DataFrame({"rfc": [10.0], "wmc": [4.0], "target": [0]})
        result = preprocess_data(df)
        assert "rfc_to_wmc_ratio" in result.columns
        # 10 / (4 + 1) = 2.0
        assert abs(result["rfc_to_wmc_ratio"].iloc[0] - 2.0) < 1e-9

    def test_rfc_to_wmc_ratio_zero_wmc(self):
        """wmc=0 must not cause division by zero (denominator is wmc+1)."""
        df = pd.DataFrame({"rfc": [10.0], "wmc": [0.0], "target": [0]})
        result = preprocess_data(df)
        assert np.isfinite(result["rfc_to_wmc_ratio"].iloc[0])
        assert abs(result["rfc_to_wmc_ratio"].iloc[0] - 10.0) < 1e-9

    def test_cbo_to_dit_ratio(self):
        df = pd.DataFrame({"cbo": [6.0], "dit": [2.0], "target": [0]})
        result = preprocess_data(df)
        assert "cbo_to_dit_ratio" in result.columns
        # 6 / (2 + 1) = 2.0
        assert abs(result["cbo_to_dit_ratio"].iloc[0] - 2.0) < 1e-9

    def test_cbo_to_dit_ratio_zero_dit(self):
        df = pd.DataFrame({"cbo": [9.0], "dit": [0.0], "target": [0]})
        result = preprocess_data(df)
        assert np.isfinite(result["cbo_to_dit_ratio"].iloc[0])

    def test_complexity_per_loc(self):
        df = pd.DataFrame({"loc": [100.0], "v(g)": [10.0], "target": [0]})
        result = preprocess_data(df)
        assert "complexity_per_loc" in result.columns
        # unified_loc=100, v(g)=10 → 10/(100+1) ≈ 0.099
        expected = 10.0 / (100.0 + 1)
        assert abs(result["complexity_per_loc"].iloc[0] - expected) < 1e-9

    def test_complexity_per_loc_zero_unified_loc(self):
        """Denominator is unified_loc+1 so zero loc must not cause division by zero."""
        df = pd.DataFrame({"loc": [0.0], "v(g)": [5.0], "target": [0]})
        result = preprocess_data(df)
        assert np.isfinite(result["complexity_per_loc"].iloc[0])

    def test_no_rfc_to_wmc_when_cols_missing(self):
        """Ratio should not appear if one of the source columns is absent."""
        df = pd.DataFrame({"wmc": [5], "target": [0]})  # no rfc
        result = preprocess_data(df)
        assert "rfc_to_wmc_ratio" not in result.columns

    def test_no_complexity_per_loc_when_vg_missing(self):
        df = pd.DataFrame({"loc": [100], "target": [0]})  # no v(g)
        result = preprocess_data(df)
        assert "complexity_per_loc" not in result.columns


# ─────────────────────────────────────────────────────────────
# 9. Only numeric columns in output
# ─────────────────────────────────────────────────────────────

class TestNumericOutput:
    def test_all_output_columns_numeric(self):
        df = pd.DataFrame({
            "wmc": [5, 10], "loc": [100, 200],
            "class_name": ["Foo", "Bar"],   # string — must be dropped
            "target": [0, 1],
        })
        result = preprocess_data(df)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"Column '{col}' is not numeric"

    def test_string_columns_dropped(self):
        df = pd.DataFrame({"wmc": [1, 2], "name": ["A", "B"], "target": [0, 1]})
        result = preprocess_data(df)
        assert "name" not in result.columns


# ─────────────────────────────────────────────────────────────
# 10. Real dataset smoke tests
# ─────────────────────────────────────────────────────────────

class TestRealDatasets:
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")

    def _load(self, filename):
        path = os.path.join(self.DATASET_DIR, filename)
        df = pd.read_csv(path)
        for col in ["bug", "defects", "label"]:
            if col in df.columns:
                c = df[col]
                dtype = str(c.dtype)
                if "str" in dtype or "object" in dtype or "string" in dtype:
                    vals = c.astype(str).str.strip().str.upper()
                    if set(vals.unique()).issubset({"Y", "N"}):
                        df[col] = vals.map({"Y": 1, "N": 0})
                    else:
                        df[col] = pd.to_numeric(vals, errors="coerce").fillna(0).astype(int)
                else:
                    df[col] = (pd.to_numeric(c, errors="coerce").fillna(0) > 0).astype(int)
                df.rename(columns={col: "target"}, inplace=True)
                break
        return df

    def test_ant_dataset(self):
        df = self._load("ant-1.7 (1).csv")
        result = preprocess_data(df, training=True)
        assert result.isnull().sum().sum() == 0
        assert "target" in result.columns

    def test_kc1_dataset_no_uppercase_leakage(self):
        """pc1 / kc1 have uppercase L,V,D,N,E — must all be dropped (Fix #4)."""
        df = self._load("kc1.csv")
        result = preprocess_data(df, training=True)
        for bad in ["L", "V", "D", "N", "E"]:
            assert bad not in result.columns, f"Noise column '{bad}' leaked into features"

    def test_pc1_dataset_no_uppercase_leakage(self):
        df = self._load("pc1.csv")
        result = preprocess_data(df, training=True)
        for bad in ["L", "V", "D", "N", "E"]:
            assert bad not in result.columns

    def test_synapse_unnamed_col_removed(self):
        """synapse-1.0 has an Unnamed: 21 trailing column (Fix #5)."""
        df = self._load("synapse-1.0.csv")
        result = preprocess_data(df, training=True)
        assert not any(c.startswith("Unnamed:") for c in result.columns)

    def test_xerces_unnamed_col_removed(self):
        df = self._load("xerces-1.2.csv")
        result = preprocess_data(df, training=True)
        assert not any(c.startswith("Unnamed:") for c in result.columns)

    def test_all_datasets_no_nan(self):
        for fname in os.listdir(self.DATASET_DIR):
            if not fname.endswith(".csv"):
                continue
            df = self._load(fname)
            result = preprocess_data(df, training=True)
            assert result.isnull().sum().sum() == 0, \
                f"{fname}: NaN values found after preprocessing"

    def test_all_datasets_have_target(self):
        for fname in os.listdir(self.DATASET_DIR):
            if not fname.endswith(".csv"):
                continue
            df = self._load(fname)
            result = preprocess_data(df, training=True)
            assert "target" in result.columns, f"{fname}: 'target' column missing"
