"""
experiment.py  —  5-experiment comparison on untouched test set (evaluated ONCE at end).
All experiments share the same 70/15/15 stratified split and preprocessing.
Results logged to experiment_log.txt and experiment_results.csv.
"""
import os, sys, time, datetime, warnings
import numpy as np, pandas as pd
import joblib, optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing import preprocess_data
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, StratifiedGroupKFold
)
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    recall_score, classification_report, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

LOG_FILE = "experiment_log.txt"
_lf = open(LOG_FILE, "w", buffering=1, encoding="utf-8")

def log(msg=""):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line); _lf.write(line + "\n")

log("=" * 72)
log("EXPERIMENT RUN — 5 improvements, one final test-set evaluation")
log(f"Started: {datetime.datetime.now().isoformat()}")
log("=" * 72)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
log("\n── LOADING DATA ──")
folder = "dataset"
dfs, groups_raw = [], []

for fname in sorted(os.listdir(folder)):
    if not fname.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(folder, fname))
    tgt = next((c for c in ["bug","defects","label"] if c in df.columns), None)
    if tgt is None: continue
    col = df[tgt]; dtype = str(col.dtype)
    if "bool" in dtype:
        df[tgt] = col.astype(int)
    elif "str" in dtype or "object" in dtype or "string" in dtype:
        vals = col.astype(str).str.strip().str.upper()
        if set(vals.unique()).issubset({"Y","N"}):   df[tgt] = vals.map({"Y":1,"N":0})
        elif set(vals.unique()).issubset({"TRUE","FALSE"}): df[tgt] = vals.map({"TRUE":1,"FALSE":0})
        else:
            df[tgt] = pd.to_numeric(vals, errors="coerce").fillna(0)
            df[tgt] = (df[tgt]>0).astype(int)
    else:
        df[tgt] = pd.to_numeric(col, errors="coerce").fillna(0)
        df[tgt] = (df[tgt]>0).astype(int)
    df.rename(columns={tgt:"target"}, inplace=True)
    df["dataset_name"] = fname.replace(".csv","")
    dfs.append(df)
    log(f"  {fname:45s} rows={len(df):5d}  buggy={df['target'].mean():.1%}")

final_df = pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
final_df = preprocess_data(final_df, training=True)
# Keep dataset_name for GroupKFold before it gets dropped as non-numeric
groups_series = final_df.pop("dataset_name") if "dataset_name" in final_df.columns else None

log(f"  Total after dedup+preprocess: {len(final_df)}  features={final_df.shape[1]-1}")

# ── 70 / 15 / 15 SPLIT (done once, shared by all experiments) ────────────────
log("\n── THREE-WAY STRATIFIED SPLIT  70/15/15 ──")
X = final_df.drop(columns=["target"])
y = final_df["target"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Groups aligned to training set (for Exp 2)
if groups_series is not None:
    groups_train = groups_series.iloc[X_train.index].reset_index(drop=True)
else:
    groups_train = None

log(f"  Train={len(X_train)}  Cal={len(X_cal)}  Test={len(X_test)}")
log(f"  Buggy%  train={y_train.mean():.1%}  cal={y_cal.mean():.1%}  test={y_test.mean():.1%}")
log("  [TEST SET LOCKED until final evaluation]")

# ── SCALE (fit on train only) ─────────────────────────────────────────────────
scaler = RobustScaler()
X_tr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns).reset_index(drop=True)
X_ca_s = pd.DataFrame(scaler.transform(X_cal),   columns=X.columns)
X_te_s = pd.DataFrame(scaler.transform(X_test),  columns=X.columns)

# ── FEATURE SELECTION (permutation importance on clean train, no SMOTE) ───────
log("\n── FEATURE SELECTION (permutation importance, no SMOTE) ──")
_base = XGBClassifier(n_estimators=150, random_state=42, eval_metric="logloss")
_base.fit(X_tr_s, y_train)
perm = permutation_importance(_base, X_tr_s, y_train,
                               n_repeats=5, random_state=42, scoring="f1_macro")
imp_mean = pd.Series(perm.importances_mean, index=X.columns)

FULL_FEATS = X.columns.tolist()                           # all features
SEL_FEATS  = imp_mean[imp_mean > 0].index.tolist()        # permutation > 0
TOP_FEATS  = imp_mean.nlargest(25).index.tolist()         # top-25

log(f"  Full features      : {len(FULL_FEATS)}")
log(f"  Permutation sel.   : {len(SEL_FEATS)}")
log(f"  Top-25 by perm imp : {len(TOP_FEATS)}")

y_tr  = y_train.reset_index(drop=True)
y_ca  = y_cal.values
y_te  = y_test.values

# ── SHARED HELPERS ────────────────────────────────────────────────────────────
results_table = []

def cal_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "MacroF1":  f1_score(y_true, y_pred, average="macro"),
        "BuggyF1":  f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "NB_F1":    f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "AUC_ROC":  roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Threshold": threshold,
    }

def best_threshold(y_true, y_prob):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        preds = (y_prob >= t).astype(int)
        mf1   = f1_score(y_true, preds, average="macro")
        br    = recall_score(y_true, preds, pos_label=1, zero_division=0)
        nbr   = recall_score(y_true, preds, pos_label=0, zero_division=0)
        if mf1 > best_f and br >= 0.68 and nbr >= 0.68:
            best_f, best_t = mf1, t
    if best_f == 0.0:
        best_t = max(np.arange(0.20,0.81,0.01),
                     key=lambda t: f1_score(y_true,(y_prob>=t).astype(int),average="macro"))
    return best_t

def stack_predict(models, meta_lr, iso_cal, X_arr):
    probs = [m.predict_proba(X_arr)[:,1] for m in models]
    raw = meta_lr.predict_proba(np.column_stack(probs))[:,1]
    return iso_cal.transform(raw)

def tune_xgb(X_arr, y_arr, cv_splitter, n_trials=50):
    def obj(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators",100,500),
            "max_depth":        trial.suggest_int("max_depth",3,9),
            "learning_rate":    trial.suggest_float("learning_rate",0.005,0.2,log=True),
            "subsample":        trial.suggest_float("subsample",0.5,1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.4,1.0),
            "min_child_weight": trial.suggest_int("min_child_weight",1,15),
            "gamma":            trial.suggest_float("gamma",0,5),
            "eval_metric":"logloss","random_state":42,
        }
        return _cv_score(XGBClassifier(**p), cv_splitter, X_arr, y_arr)
    s = optuna.create_study(direction="maximize",
          pruner=optuna.pruners.MedianPruner(n_startup_trials=10,n_warmup_steps=3))
    s.optimize(obj, n_trials=n_trials)
    r = s.best_params; r.update({"eval_metric":"logloss","random_state":42})
    return r, s.best_value

def tune_lgbm(X_arr, y_arr, cv_splitter, n_trials=50):
    def obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators",100,500),
            "max_depth":         trial.suggest_int("max_depth",3,9),
            "learning_rate":     trial.suggest_float("learning_rate",0.005,0.2,log=True),
            "subsample":         trial.suggest_float("subsample",0.5,1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",0.4,1.0),
            "min_child_samples": trial.suggest_int("min_child_samples",5,60),
            "reg_alpha":         trial.suggest_float("reg_alpha",0,2),
            "reg_lambda":        trial.suggest_float("reg_lambda",0,2),
            "random_state":42,"verbose":-1,
        }
        return _cv_score(LGBMClassifier(**p), cv_splitter, X_arr, y_arr)
    s = optuna.create_study(direction="maximize",
          pruner=optuna.pruners.MedianPruner(n_startup_trials=10,n_warmup_steps=3))
    s.optimize(obj, n_trials=n_trials)
    r = s.best_params; r.update({"random_state":42,"verbose":-1})
    return r, s.best_value

def tune_rf(X_arr, y_arr, cv_splitter, n_trials=30):
    def obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators",100,500),
            "max_depth":         trial.suggest_int("max_depth",3,20),
            "max_features":      trial.suggest_float("max_features",0.2,1.0),
            "min_samples_split": trial.suggest_int("min_samples_split",2,20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",1,10),
            "random_state":42,"n_jobs":-1,
        }
        return _cv_score(RandomForestClassifier(**p), cv_splitter, X_arr, y_arr)
    s = optuna.create_study(direction="maximize",
          pruner=optuna.pruners.MedianPruner(n_startup_trials=8,n_warmup_steps=3))
    s.optimize(obj, n_trials=n_trials)
    r = s.best_params; r.update({"random_state":42,"n_jobs":-1})
    return r, s.best_value

# _cv_score is used by all tune_* helpers — defined after helpers so it can reference them
def _cv_score_smote(model_class, params, cv_splitter, X_arr, y_arr, groups=None):
    """CV with SMOTE inside fold."""
    scores = []
    split_args = (X_arr, y_arr) if groups is None else (X_arr, y_arr, groups)
    for tr, val in cv_splitter.split(*split_args):
        X_f, y_f = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_arr[tr], y_arr[tr])
        m = model_class(**params)
        m.fit(X_f, y_f)
        scores.append(f1_score(y_arr[val], m.predict(X_arr[val]), average="macro"))
    return np.mean(scores)

def _cv_score_weighted(model_class, params, cv_splitter, X_arr, y_arr, groups=None):
    """CV with class_weight='balanced' (no SMOTE)."""
    p2 = dict(params)
    if "n_estimators" in p2:  # tree model
        p2["class_weight"] = "balanced"
    scores = []
    split_args = (X_arr, y_arr) if groups is None else (X_arr, y_arr, groups)
    for tr, val in cv_splitter.split(*split_args):
        m = model_class(**p2)
        m.fit(X_arr[tr], y_arr[tr])
        scores.append(f1_score(y_arr[val], m.predict(X_arr[val]), average="macro"))
    return np.mean(scores)

# Default _cv_score used by tune_* (will be monkey-patched per experiment)
_cv_score = None

def build_stacking(X_arr, y_arr, X_ca_arr, xgb_p, lgbm_p, rf_p, skf, use_smote=True):
    """Build stacking ensemble, return (base_models, meta_lr, iso_cal, cal_proba)."""
    n = len(X_arr)
    oof_x = np.zeros(n); oof_l = np.zeros(n); oof_r = np.zeros(n)
    ca_x = np.zeros(len(X_ca_arr)); ca_l = np.zeros(len(X_ca_arr)); ca_r = np.zeros(len(X_ca_arr))

    for tr, val in skf.split(X_arr, y_arr):
        if use_smote:
            Xf, yf = SMOTE(random_state=42,k_neighbors=5).fit_resample(X_arr[tr],y_arr[tr])
        else:
            Xf, yf = X_arr[tr], y_arr[tr]
        xm = XGBClassifier(**xgb_p);  xm.fit(Xf, yf)
        lm = LGBMClassifier(**lgbm_p); lm.fit(Xf, yf)
        rm = RandomForestClassifier(**rf_p); rm.fit(Xf, yf)
        oof_x[val] = xm.predict_proba(X_arr[val])[:,1]
        oof_l[val] = lm.predict_proba(X_arr[val])[:,1]
        oof_r[val] = rm.predict_proba(X_arr[val])[:,1]
        ca_x += xm.predict_proba(X_ca_arr)[:,1]
        ca_l += lm.predict_proba(X_ca_arr)[:,1]
        ca_r += rm.predict_proba(X_ca_arr)[:,1]

    ca_x/=5; ca_l/=5; ca_r/=5
    meta_tr = np.column_stack([oof_x,oof_l,oof_r])
    meta_ca = np.column_stack([ca_x,ca_l,ca_r])

    meta_lr = LogisticRegression(C=1.0,random_state=42,max_iter=1000)
    meta_lr.fit(meta_tr, y_arr)
    raw_ca = meta_lr.predict_proba(meta_ca)[:,1]
    iso    = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_ca, y_ca)
    cal_ca = iso.transform(raw_ca)
    t      = best_threshold(y_ca, cal_ca)

    # Final base models on full training set
    if use_smote:
        Xf_full, yf_full = SMOTE(random_state=42,k_neighbors=5).fit_resample(X_arr, y_arr)
    else:
        Xf_full, yf_full = X_arr, y_arr
    xm_f = XGBClassifier(**xgb_p);   xm_f.fit(Xf_full, yf_full)
    lm_f = LGBMClassifier(**lgbm_p);  lm_f.fit(Xf_full, yf_full)
    rm_f = RandomForestClassifier(**rf_p); rm_f.fit(Xf_full, yf_full)

    return [xm_f, lm_f, rm_f], meta_lr, iso, t

def eval_on_cal(models, meta_lr, iso, threshold, X_ca_arr, y_ca, label):
    probs = stack_predict(models, meta_lr, iso, X_ca_arr)
    m = cal_metrics(y_ca, probs, threshold)
    m["Experiment"] = label
    log(f"  {label:50s}  MacroF1={m['MacroF1']:.4f}  AUC={m['AUC_ROC']:.4f}  "
        f"BuggyF1={m['BuggyF1']:.4f}  NB_F1={m['NB_F1']:.4f}  thresh={threshold:.2f}")
    results_table.append(m)
    return m
