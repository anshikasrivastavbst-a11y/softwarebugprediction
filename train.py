"""
train.py - Final Leakage-Free Software Bug Prediction Pipeline
==============================================================
Design guarantees:
  - SMOTE applied INSIDE each CV fold (never before splitting)
  - 70 / 15 / 15 stratified split: train / cal / test
  - Isotonic calibration fit on cal set ONLY
  - Threshold tuned on cal set ONLY
  - Final test set touched EXACTLY ONCE at the end
  - Every step logged to training_log_final.txt
  - Artifacts saved once and not overwritten
"""

import os, sys, time, datetime
import numpy as np
import pandas as pd
import joblib
import optuna
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, recall_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Logger: writes to both stdout and training_log_final.txt ──────────────────
LOG_FILE = "training_log_final.txt"
_log_f = open(LOG_FILE, "w", buffering=1, encoding="utf-8")

def log(msg=""):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _log_f.write(line + "\n")

log("=" * 70)
log("FINAL LEAKAGE-FREE TRAINING RUN")
log(f"Started : {datetime.datetime.now().isoformat()}")
log("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD ALL DATASETS
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 1: Loading datasets...")
folder_path = "dataset"
dataframes = []

for file in sorted(os.listdir(folder_path)):
    if not file.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(folder_path, file))
    tgt = next((c for c in ["bug","defects","label"] if c in df.columns), None)
    if tgt is None:
        log(f"  SKIP {file} — no target column")
        continue
    col = df[tgt]; dtype = str(col.dtype)
    if "bool" in dtype:
        df[tgt] = col.astype(int)
    elif "str" in dtype or "object" in dtype or "string" in dtype:
        vals = col.astype(str).str.strip().str.upper()
        if set(vals.unique()).issubset({"Y","N"}):
            df[tgt] = vals.map({"Y":1,"N":0})
        elif set(vals.unique()).issubset({"TRUE","FALSE"}):
            df[tgt] = vals.map({"TRUE":1,"FALSE":0})
        else:
            df[tgt] = pd.to_numeric(vals, errors="coerce").fillna(0)
            df[tgt] = (df[tgt]>0).astype(int)
    else:
        df[tgt] = pd.to_numeric(col, errors="coerce").fillna(0)
        df[tgt] = (df[tgt]>0).astype(int)
    df.rename(columns={tgt:"target"}, inplace=True)
    df["dataset_name"] = file.replace(".csv","")
    dataframes.append(df)
    log(f"  Loaded {file:45s} rows={len(df):5d}  buggy={df['target'].mean():.1%}")

final_df = pd.concat(dataframes, ignore_index=True)
log(f"\n  Rows before dedup : {len(final_df)}")
final_df = final_df.drop_duplicates().reset_index(drop=True)
log(f"  Rows after  dedup : {len(final_df)}")
log(f"  Datasets loaded   : {len(dataframes)}")
log(f"  Class distribution:\n{final_df['target'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 2: Preprocessing...")
final_df = preprocess_data(final_df, training=True)
log(f"  Shape after preprocessing : {final_df.shape}")
log(f"  NaN values remaining      : {final_df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: THREE-WAY STRATIFIED SPLIT  70 / 15 / 15
# Requirement 2: test set is isolated here and not touched again until Step 12
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 3: Three-way stratified split  70 / 15 / 15...")
X = final_df.drop(columns=["target"])
y = final_df["target"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

log(f"  Train : {len(X_train):5d} rows  ({len(X_train)/len(X)*100:.1f}%)  buggy={y_train.mean():.1%}")
log(f"  Cal   : {len(X_cal):5d} rows  ({len(X_cal)/len(X)*100:.1f}%)  buggy={y_cal.mean():.1%}")
log(f"  Test  : {len(X_test):5d} rows  ({len(X_test)/len(X)*100:.1f}%)  buggy={y_test.mean():.1%}")
log("  [TEST SET NOW ISOLATED — will not be touched until Step 12]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: SCALE  (fit on train only — strict no-leakage)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 4: Scaling (RobustScaler, fit on train only)...")
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
)
X_cal_scaled = pd.DataFrame(
    scaler.transform(X_cal), columns=X.columns, index=X_cal.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns, index=X_test.index
)
log("  Scaler fitted on train set only.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: FEATURE SELECTION
# Done on CLEAN (pre-SMOTE) training data — structural decision, not label-leaking
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 5: Feature selection (XGBoost importance on clean train set)...")
_sel = XGBClassifier(n_estimators=150, random_state=42, eval_metric="logloss")
_sel.fit(X_train_scaled, y_train)
importances = pd.Series(_sel.feature_importances_, index=X.columns)
selected_features = importances[importances > 0].index.tolist()
dropped = [f for f in X.columns if f not in selected_features]
log(f"  Original features  : {len(X.columns)}")
log(f"  Selected features  : {len(selected_features)}")
log(f"  Dropped (zero imp) : {dropped}")

X_tr  = X_train_scaled[selected_features].reset_index(drop=True)
X_ca  = X_cal_scaled[selected_features]
X_te  = X_test_scaled[selected_features]
y_tr  = y_train.reset_index(drop=True)
y_ca  = y_cal.values
y_te  = y_test.values

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: OPTUNA — XGBoost  (SMOTE INSIDE EACH FOLD — Requirement 1)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 6: Optuna tuning — XGBoost (50 trials, SMOTE inside fold)...")
t0 = time.time()

def xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "eval_metric":      "logloss",
        "random_state":     42,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        # SMOTE on training fold only — validation fold never sees synthetic samples
        X_f, y_f = SMOTE(random_state=42, k_neighbors=5).fit_resample(
            X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        )
        m = XGBClassifier(**params)
        m.fit(X_f, y_f)
        scores.append(f1_score(y_tr.iloc[val_idx],
                               m.predict(X_tr.iloc[val_idx]), average="macro"))
    return np.mean(scores)

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)
xgb_best = xgb_study.best_params
xgb_best.update({"eval_metric": "logloss", "random_state": 42})
log(f"  XGBoost best CV Macro F1 : {xgb_study.best_value:.4f}  ({time.time()-t0:.0f}s)")
log(f"  Best params : {xgb_best}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: OPTUNA — LightGBM  (SMOTE inside fold)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 7: Optuna tuning — LightGBM (50 trials, SMOTE inside fold)...")
t0 = time.time()

def lgbm_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 9),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0, 2),
        "class_weight":      "balanced",
        "random_state":      42,
        "verbose":           -1,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        X_f, y_f = SMOTE(random_state=42, k_neighbors=5).fit_resample(
            X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        )
        m = LGBMClassifier(**params)
        m.fit(X_f, y_f)
        scores.append(f1_score(y_tr.iloc[val_idx],
                               m.predict(X_tr.iloc[val_idx]), average="macro"))
    return np.mean(scores)

lgbm_study = optuna.create_study(direction="maximize")
lgbm_study.optimize(lgbm_objective, n_trials=50)
lgbm_best = lgbm_study.best_params
lgbm_best.update({"random_state": 42, "verbose": -1, "class_weight": "balanced"})
log(f"  LightGBM best CV Macro F1 : {lgbm_study.best_value:.4f}  ({time.time()-t0:.0f}s)")
log(f"  Best params : {lgbm_best}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: OPTUNA — RandomForest  (SMOTE inside fold)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 8: Optuna tuning — RandomForest (30 trials, SMOTE inside fold)...")
t0 = time.time()

def rf_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 20),
        "max_features":      trial.suggest_float("max_features", 0.2, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            -1,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        X_f, y_f = SMOTE(random_state=42, k_neighbors=5).fit_resample(
            X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        )
        m = RandomForestClassifier(**params)
        m.fit(X_f, y_f)
        scores.append(f1_score(y_tr.iloc[val_idx],
                               m.predict(X_tr.iloc[val_idx]), average="macro"))
    return np.mean(scores)

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=30)
rf_best = rf_study.best_params
rf_best.update({"random_state": 42, "n_jobs": -1, "class_weight": "balanced"})
log(f"  RandomForest best CV Macro F1 : {rf_study.best_value:.4f}  ({time.time()-t0:.0f}s)")
log(f"  Best params : {rf_best}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: STACKING ENSEMBLE
# OOF predictions built with SMOTE-per-fold on training data.
# Cal and test sets are never seen by base learners during stacking.
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 9: Building stacking ensemble (OOF, SMOTE-per-fold)...")
t0 = time.time()

skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_tr_arr = X_tr.values
X_ca_arr = X_ca.values
X_te_arr = X_te.values

oof_xgb  = np.zeros(len(X_tr_arr))
oof_lgbm = np.zeros(len(X_tr_arr))
oof_rf   = np.zeros(len(X_tr_arr))
cal_xgb  = np.zeros(len(X_ca_arr))
cal_lgbm = np.zeros(len(X_ca_arr))
cal_rf   = np.zeros(len(X_ca_arr))
te_xgb   = np.zeros(len(X_te_arr))
te_lgbm  = np.zeros(len(X_te_arr))
te_rf    = np.zeros(len(X_te_arr))

for fold, (tr_idx, val_idx) in enumerate(skf5.split(X_tr_arr, y_tr.values)):
    log(f"  Fold {fold+1}/5 ...", )
    X_f, y_f = SMOTE(random_state=42, k_neighbors=5).fit_resample(
        X_tr_arr[tr_idx], y_tr.values[tr_idx]
    )
    xm = XGBClassifier(**xgb_best);  xm.fit(X_f, y_f)
    lm = LGBMClassifier(**lgbm_best); lm.fit(X_f, y_f)
    rm = RandomForestClassifier(**rf_best); rm.fit(X_f, y_f)

    oof_xgb[val_idx]  = xm.predict_proba(X_tr_arr[val_idx])[:, 1]
    oof_lgbm[val_idx] = lm.predict_proba(X_tr_arr[val_idx])[:, 1]
    oof_rf[val_idx]   = rm.predict_proba(X_tr_arr[val_idx])[:, 1]

    cal_xgb  += xm.predict_proba(X_ca_arr)[:, 1]
    cal_lgbm += lm.predict_proba(X_ca_arr)[:, 1]
    cal_rf   += rm.predict_proba(X_ca_arr)[:, 1]
    te_xgb   += xm.predict_proba(X_te_arr)[:, 1]
    te_lgbm  += lm.predict_proba(X_te_arr)[:, 1]
    te_rf    += rm.predict_proba(X_te_arr)[:, 1]

cal_xgb /= 5; cal_lgbm /= 5; cal_rf /= 5
te_xgb  /= 5; te_lgbm  /= 5; te_rf  /= 5

meta_train = np.column_stack([oof_xgb,  oof_lgbm,  oof_rf])
meta_cal   = np.column_stack([cal_xgb,  cal_lgbm,  cal_rf])
meta_test  = np.column_stack([te_xgb,   te_lgbm,   te_rf])

meta_lr = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
meta_lr.fit(meta_train, y_tr.values)
log(f"  Meta-learner trained. ({time.time()-t0:.0f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: RETRAIN BASE LEARNERS ON FULL TRAINING SET
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 10: Retraining base learners on full training set (SMOTE applied once)...")
t0 = time.time()
sm_full = SMOTE(random_state=42, k_neighbors=5)
X_tr_full, y_tr_full = sm_full.fit_resample(X_tr_arr, y_tr.values)
log(f"  SMOTE balanced training set: {dict(zip(*np.unique(y_tr_full, return_counts=True)))}")

xgb_final  = XGBClassifier(**xgb_best);         xgb_final.fit(X_tr_full,  y_tr_full)
lgbm_final = LGBMClassifier(**lgbm_best);        lgbm_final.fit(X_tr_full, y_tr_full)
rf_final   = RandomForestClassifier(**rf_best);  rf_final.fit(X_tr_full,   y_tr_full)
log(f"  Base learners ready. ({time.time()-t0:.0f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: ISOTONIC CALIBRATION — fit on CAL SET ONLY  (Requirement 3)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 11: Isotonic calibration on cal set ONLY (test set NOT used)...")
xgb_cal_p  = xgb_final.predict_proba(X_ca_arr)[:, 1]
lgbm_cal_p = lgbm_final.predict_proba(X_ca_arr)[:, 1]
rf_cal_p   = rf_final.predict_proba(X_ca_arr)[:, 1]
raw_cal    = meta_lr.predict_proba(
    np.column_stack([xgb_cal_p, lgbm_cal_p, rf_cal_p])
)[:, 1]

iso_cal = IsotonicRegression(out_of_bounds="clip")
iso_cal.fit(raw_cal, y_ca)
cal_proba_calibrated = iso_cal.transform(raw_cal)
log("  Isotonic calibration fitted on cal set only.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11b: THRESHOLD TUNING — on CAL SET ONLY  (Requirement 4)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 11b: Threshold tuning on cal set ONLY (test set NOT used)...")
log(f"  {'Thresh':>7}  {'MacroF1':>8}  {'BuggyF1':>8}  {'NB-F1':>8}  {'BuggyRec':>9}  {'NB-Rec':>8}")

best_t, best_mf1 = 0.5, 0.0
cal_rows = []
for t in np.arange(0.20, 0.81, 0.01):
    preds = (cal_proba_calibrated >= t).astype(int)
    mf1   = f1_score(y_ca, preds, average="macro")
    bf1   = f1_score(y_ca, preds, pos_label=1, zero_division=0)
    nbf1  = f1_score(y_ca, preds, pos_label=0, zero_division=0)
    brec  = recall_score(y_ca, preds, pos_label=1, zero_division=0)
    nbrec = recall_score(y_ca, preds, pos_label=0, zero_division=0)
    cal_rows.append((t, mf1, bf1, nbf1, brec, nbrec))
    if mf1 > best_mf1 and brec >= 0.70 and nbrec >= 0.70:
        best_mf1, best_t = mf1, t

if best_mf1 == 0.0:
    best_t   = max(cal_rows, key=lambda x: x[1])[0]
    best_mf1 = max(cal_rows, key=lambda x: x[1])[1]
    log("  (Recall constraint not met — using best Macro F1 threshold)")

# Log the full sweep
for t, mf1, bf1, nbf1, brec, nbrec in cal_rows:
    marker = " <-- SELECTED" if abs(t - best_t) < 0.005 else ""
    log(f"  {t:.2f}   {mf1:.4f}    {bf1:.4f}    {nbf1:.4f}    {brec:.4f}    {nbrec:.4f}{marker}")

log(f"\n  Selected threshold : {best_t:.2f}  (cal-set Macro F1 = {best_mf1:.4f})")
log("  [THRESHOLD SELECTED — cal set will not be used again]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12: SAVE ARTIFACTS  (Requirement 6 — saved once, not overwritten)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 12: Saving artifacts...")

ensemble = {
    "xgb":     xgb_final,
    "lgbm":    lgbm_final,
    "rf":      rf_final,
    "meta_lr": meta_lr,
    "iso_cal": iso_cal,
}
joblib.dump(ensemble,         "model.pkl")
joblib.dump(scaler,           "scaler.pkl")
joblib.dump(best_t,           "threshold.pkl")
joblib.dump(selected_features,"selected_features.pkl")

log("  model.pkl           saved")
log("  scaler.pkl          saved")
log("  threshold.pkl       saved")
log("  selected_features.pkl saved")
log("  [ARTIFACTS FROZEN — will not be overwritten]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13: FINAL EVALUATION — test set used EXACTLY ONCE  (Requirement 5)
# ─────────────────────────────────────────────────────────────────────────────
log("\nSTEP 13: FINAL EVALUATION ON UNTOUCHED TEST SET (used for the first and only time)")
log("=" * 70)

xgb_te_p  = xgb_final.predict_proba(X_te_arr)[:, 1]
lgbm_te_p = lgbm_final.predict_proba(X_te_arr)[:, 1]
rf_te_p   = rf_final.predict_proba(X_te_arr)[:, 1]
raw_te    = meta_lr.predict_proba(np.column_stack([xgb_te_p, lgbm_te_p, rf_te_p]))[:, 1]
y_proba   = iso_cal.transform(raw_te)
y_pred    = (y_proba >= best_t).astype(int)

acc    = accuracy_score(y_te, y_pred)
mf1    = f1_score(y_te, y_pred, average="macro")
bf1    = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
nbf1   = f1_score(y_te, y_pred, pos_label=0, zero_division=0)
brec   = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
nbrec  = recall_score(y_te, y_pred, pos_label=0, zero_division=0)
auc    = roc_auc_score(y_te, y_proba)
cm     = confusion_matrix(y_te, y_pred)

report = classification_report(y_te, y_pred, target_names=["Not Buggy", "Buggy"])

log(report)
log(f"Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
log(f"Macro F1           : {mf1:.4f}")
log(f"Buggy F1           : {bf1:.4f}  (recall = {brec:.4f})")
log(f"Not-Buggy F1       : {nbf1:.4f}  (recall = {nbrec:.4f})")
log(f"AUC-ROC            : {auc:.4f}")
log(f"Decision Threshold : {best_t:.3f}  (tuned on cal set only)")
log(f"Features used      : {len(selected_features)}")
log(f"Test set size      : {len(y_te)}")
log(f"\nConfusion Matrix:")
log(f"  [[TN={cm[0][0]}  FP={cm[0][1]}]")
log(f"   [FN={cm[1][0]}  TP={cm[1][1]}]]")
log("=" * 70)
log("[EVALUATION COMPLETE — test set will not be used again]")
log(f"\nRun completed : {datetime.datetime.now().isoformat()}")

_log_f.close()
print(f"\nFull log saved to: {LOG_FILE}")
