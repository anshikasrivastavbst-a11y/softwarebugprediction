import pandas as pd
import numpy as np

# Features to remove — listed in both lower and upper case to handle
# datasets like pc1.csv that use uppercase column names (L, V, D, N, E …)
FEATURES_TO_DROP = [
    # Halstead / noise — lowercase variants
    "D", "L", "l", "locCodeAndComment", "V", "N", "E",
    "uniq_Opnd", "iv(G)", "total_Op",
    # Uppercase variants present in pc1.csv and similar NASA datasets
    "d", "v", "n", "e",
]

# Columns that are never informative regardless of capitalisation
_UNNAMED_PATTERN = r"^Unnamed:"

# Target-like column names — strip them in inference mode so they
# cannot accidentally leak into features.
_TARGET_COLUMNS = {"target", "bug", "defects", "label"}


def preprocess_data(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Shared preprocessing for train.py, predict.py, FastAPI and Streamlit.

    Parameters
    ----------
    df       : raw input DataFrame
    training : when True the 'target' column is preserved; when False
               any target-like column is dropped before returning.

    Returns
    -------
    Cleaned DataFrame ready for scaling / prediction.
    """

    df = df.copy()

    # ------------------------------------------------------------------
    # 0. Drop unnamed / trailing columns produced by Excel exports
    # ------------------------------------------------------------------
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)

    # ------------------------------------------------------------------
    # 1. Convert boolean target columns to int BEFORE select_dtypes,
    #    because bool is NOT included in np.number and would be silently
    #    dropped together with the labels we need to keep.
    # ------------------------------------------------------------------
    for col in list(df.columns):
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    #    All engineered columns are created BEFORE the noisy-feature drop
    #    so that the source columns are still present when we need them.
    # ------------------------------------------------------------------

    # unified_loc — combine "loc" and "lOCode" into one column.
    # Create this regardless of which source column is present so that
    # downstream complexity_per_loc can always rely on it.
    if "loc" in df.columns and "lOCode" in df.columns:
        df["unified_loc"] = df["loc"].fillna(df["lOCode"]).fillna(0)
    elif "loc" in df.columns:
        df["unified_loc"] = df["loc"].fillna(0)
    elif "lOCode" in df.columns:
        df["unified_loc"] = df["lOCode"].fillna(0)
    # else: neither present — unified_loc won't be created here; the model
    # will fill it with 0 later via the missing-column padding in predict.py

    if {"rfc", "wmc"}.issubset(df.columns):
        df["rfc_to_wmc_ratio"] = df["rfc"] / (df["wmc"] + 1)

    if {"cbo", "dit"}.issubset(df.columns):
        df["cbo_to_dit_ratio"] = df["cbo"] / (df["dit"] + 1)

    # complexity_per_loc depends on unified_loc — only compute when available
    if {"v(g)", "unified_loc"}.issubset(df.columns):
        df["complexity_per_loc"] = df["v(g)"] / (df["unified_loc"] + 1)

    # ── Additional CK interaction features ────────────────────────────
    # Grounded in Chidamber & Kemerer metric semantics:

    # lcom_to_wmc: cohesion deficit per method — high → god class risk
    if {"lcom", "wmc"}.issubset(df.columns):
        df["lcom_to_wmc"] = df["lcom"] / (df["wmc"] + 1)

    # fan_in_out: total coupling (afferent + efferent)
    if {"ca", "ce"}.issubset(df.columns):
        df["fan_in_out"] = df["ca"] + df["ce"]

    # inheritance_x_methods: deep + wide hierarchy → high change-impact
    if {"dit", "wmc"}.issubset(df.columns):
        df["dit_x_wmc"] = df["dit"] * df["wmc"]

    # coupling_cohesion: classes with high coupling AND low cohesion are
    # most prone to defects (Briand et al.)
    if {"cbo", "lcom"}.issubset(df.columns):
        df["coupling_cohesion"] = df["cbo"] * np.log1p(df["lcom"])

    # ------------------------------------------------------------------
    # 3. Drop housekeeping columns
    # ------------------------------------------------------------------
    df.drop(columns=["sno"], errors="ignore", inplace=True)

    # ------------------------------------------------------------------
    # 4. Drop noisy / redundant features
    #    Use case-insensitive matching to catch both "L" and "l", etc.
    # ------------------------------------------------------------------
    lower_to_drop = {c.lower() for c in FEATURES_TO_DROP}
    cols_to_drop = [c for c in df.columns if c.lower() in lower_to_drop]
    df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    # ------------------------------------------------------------------
    # 5. In inference mode, remove any accidental target columns so they
    #    cannot be treated as input features by the model.
    # ------------------------------------------------------------------
    if not training:
        leak_cols = [c for c in df.columns if c.lower() in _TARGET_COLUMNS]
        df.drop(columns=leak_cols, errors="ignore", inplace=True)

    # ------------------------------------------------------------------
    # 6. Keep only numeric columns
    # ------------------------------------------------------------------
    df = df.select_dtypes(include=np.number)

    # ------------------------------------------------------------------
    # 7. Fill remaining NaN values — use assignment, not inplace=True,
    #    because select_dtypes may return a copy in pandas 3.x and
    #    inplace on a copy silently fails.
    # ------------------------------------------------------------------
    df = df.fillna(0)

    return df
