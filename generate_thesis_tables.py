"""
generate_thesis_tables.py
=========================
Reads outputs from results/ and generates thesis-ready tables in CSV format.
Ready to copy into Microsoft Word as formatted tables.
"""

import os
import pandas as pd
import numpy as np
import joblib

RESULTS = "results"
TABLES  = os.path.join(RESULTS, "thesis_tables")
os.makedirs(TABLES, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Overall Performance Table
# ─────────────────────────────────────────────────────────────────────────────
metrics_df = pd.read_csv(f"{RESULTS}/overall_metrics.csv")

t1 = metrics_df.copy()
t1.columns = ["Metric", "Value"]
t1["Value"] = t1["Value"].apply(
    lambda x: f"{float(x):.4f}" if str(x).replace(".","").isdigit() else x
)
t1.to_csv(f"{TABLES}/table1_overall_performance.csv", index=False)
print("[1] Overall Performance Table saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-Class Performance Table
# ─────────────────────────────────────────────────────────────────────────────
cr = pd.read_csv(f"{RESULTS}/classification_report.csv", index_col=0)

# Keep only the class rows
class_rows = cr.loc[["Not Buggy", "Buggy"]].copy()
class_rows = class_rows[["precision","recall","f1-score","support"]]
class_rows.columns = ["Precision","Recall","F1-Score","Support"]
class_rows["Support"] = class_rows["Support"].astype(int)
class_rows = class_rows.round(4)
class_rows.index.name = "Class"
class_rows.reset_index().to_csv(f"{TABLES}/table2_per_class_performance.csv", index=False)
print("[2] Per-Class Performance Table saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix Table
# ─────────────────────────────────────────────────────────────────────────────
cm_df = pd.read_csv(f"{RESULTS}/confusion_matrix.csv", index_col=0)
cm_df.index.name   = "Actual \\ Predicted"
cm_df.reset_index().to_csv(f"{TABLES}/table3_confusion_matrix.csv", index=False)
print("[3] Confusion Matrix Table saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Importance Table (top 20 by mean importance)
# ─────────────────────────────────────────────────────────────────────────────
combined = pd.read_csv(f"{RESULTS}/feature_importance_combined.csv")
top20 = combined.head(20).copy()
top20["Rank"] = range(1, 21)
top20 = top20[["Rank","feature","xgb_imp","lgbm_imp","rf_imp","mean_importance"]]
top20.columns = ["Rank","Feature","XGBoost","LightGBM","RandomForest","Mean Importance"]
top20 = top20.round({"XGBoost":5,"LightGBM":5,"RandomForest":5,"Mean Importance":5})
top20.to_csv(f"{TABLES}/table4_feature_importance.csv", index=False)
print("[4] Feature Importance Table saved (top 20).")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Error Analysis Summary
# ─────────────────────────────────────────────────────────────────────────────
tp = len(pd.read_csv(f"{RESULTS}/true_positives.csv"))
tn = len(pd.read_csv(f"{RESULTS}/true_negatives.csv"))
fp = len(pd.read_csv(f"{RESULTS}/false_positives.csv"))
fn = len(pd.read_csv(f"{RESULTS}/false_negatives.csv"))
total = tp + tn + fp + fn

error_df = pd.DataFrame({
    "Category":   ["True Positive (TP)","True Negative (TN)",
                   "False Positive (FP)","False Negative (FN)","Total Samples"],
    "Count":      [tp, tn, fp, fn, total],
    "Percentage": [f"{tp/total*100:.2f}%", f"{tn/total*100:.2f}%",
                   f"{fp/total*100:.2f}%", f"{fn/total*100:.2f}%", "100.00%"],
    "Description":["Buggy correctly identified","Not Buggy correctly identified",
                   "Not Buggy wrongly flagged as Buggy","Buggy module missed",""],
})
error_df.to_csv(f"{TABLES}/table5_error_analysis.csv", index=False)
print("[5] Error Analysis Summary saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Dataset Statistics Summary
# ─────────────────────────────────────────────────────────────────────────────
folder = "dataset"
rows = []
for fname in sorted(os.listdir(folder)):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(folder, fname))
    tgt = None
    for c in ["bug","defects","label"]:
        if c in df.columns:
            tgt = c; break
    if tgt is None:
        continue
    col = df[tgt]; dtype = str(col.dtype)
    if "str" in dtype or "object" in dtype or "string" in dtype:
        vals = col.astype(str).str.strip().str.upper()
        if set(vals.unique()).issubset({"Y","N"}):
            binary = vals.map({"Y":1,"N":0})
        else:
            binary = pd.to_numeric(vals, errors="coerce").fillna(0)
            binary = (binary>0).astype(int)
    else:
        binary = (pd.to_numeric(col, errors="coerce").fillna(0)>0).astype(int)
    buggy = int(binary.sum())
    total_r = len(df)
    rows.append({
        "Dataset":       fname.replace(".csv",""),
        "Total Samples": total_r,
        "Buggy":         buggy,
        "Not Buggy":     total_r - buggy,
        "Buggy %":       f"{buggy/total_r*100:.1f}%",
        "Features":      len(df.columns)-1,
        "Source":        "PROMISE/NASA",
    })
dataset_df = pd.DataFrame(rows)
dataset_df.to_csv(f"{TABLES}/table6_dataset_statistics.csv", index=False)
print("[6] Dataset Statistics Summary saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Model Configuration Summary
# ─────────────────────────────────────────────────────────────────────────────
config_df = pd.DataFrame({
    "Component": [
        "Base Learner 1","Base Learner 2","Base Learner 3",
        "Meta-Learner","Calibration","Feature Selection",
        "Class Balancing","Train/Test Split","Scaler",
        "Optimization","CV Strategy","Total Features","Datasets",
    ],
    "Configuration": [
        "XGBoost Classifier","LightGBM Classifier","Random Forest Classifier",
        "Logistic Regression","Isotonic Regression","XGBoost Feature Importance (importance > 0)",
        "SMOTE (k_neighbors=5, 50/50 balance)","80/20 Stratified","RobustScaler",
        "Optuna (TPE Sampler, Macro F1 objective)","3-Fold Stratified CV (Optuna trials)",
        "63 (after selection from 64)","21 PROMISE/NASA datasets",
    ],
})
config_df.to_csv(f"{TABLES}/table7_model_configuration.csv", index=False)
print("[7] Model Configuration Summary saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Hyperparameter Summary
# ─────────────────────────────────────────────────────────────────────────────
ensemble = joblib.load("model.pkl")
xgb_m  = ensemble["xgb"]
lgbm_m = ensemble["lgbm"]
rf_m   = ensemble["rf"]

def get_params(model, prefix):
    p = model.get_params()
    rows_list = []
    key_params = {
        "n_estimators","max_depth","learning_rate","subsample",
        "colsample_bytree","min_child_weight","gamma","reg_alpha",
        "reg_lambda","num_leaves","min_child_samples","n_jobs",
        "max_features","min_samples_split","random_state",
        "scale_pos_weight","class_weight",
    }
    for k,v in p.items():
        if k in key_params and v is not None:
            rows_list.append({"Model":prefix,"Parameter":k,"Value":str(v)})
    return rows_list

hyp_rows = get_params(xgb_m,"XGBoost") + get_params(lgbm_m,"LightGBM") + get_params(rf_m,"RandomForest")
hyp_df = pd.DataFrame(hyp_rows)
hyp_df.to_csv(f"{TABLES}/table8_hyperparameters.csv", index=False)
print("[8] Hyperparameter Summary saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Threshold and Calibration Summary
# ─────────────────────────────────────────────────────────────────────────────
threshold = float(joblib.load("threshold.pkl"))
with open(f"{RESULTS}/roc_auc.txt") as f:
    auc_line = f.read().strip()
with open(f"{RESULTS}/pr_auc.txt") as f:
    pr_line = f.read().strip()

thresh_df = pd.DataFrame({
    "Parameter":   ["Decision Threshold","Calibration Method",
                    "Threshold Selection Criterion",
                    "ROC-AUC","PR-AUC (Average Precision)"],
    "Value":       [threshold,"Isotonic Regression",
                    "Maximum Macro F1 on test set (known limitation: see thesis limitations section)",
                    auc_line.split(":")[1].strip(),
                    pr_line.split(":")[1].strip()],
})
thresh_df.to_csv(f"{TABLES}/table9_threshold_calibration.csv", index=False)
print("[9] Threshold and Calibration Summary saved.")

print(f"\n=== Script 2 complete. Thesis tables in {TABLES}/ ===")
print("    9 tables generated — ready for Word/LaTeX.")
