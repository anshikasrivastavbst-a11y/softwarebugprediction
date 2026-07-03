"""
generate_thesis_results.py
==========================
Loads the frozen trained pipeline and generates all evaluation outputs
into the results/ folder. Does NOT retrain or modify any artifact.
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
)
from sklearn.calibration import calibration_curve as sk_calibration_curve

warnings.filterwarnings("ignore")

from preprocessing import preprocess_data

# ── Output folder ─────────────────────────────────────────────────────────────
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
print("Loading artifacts...")
ensemble  = joblib.load("model.pkl")
scaler    = joblib.load("scaler.pkl")
sel_feats = joblib.load("selected_features.pkl")
threshold = float(joblib.load("threshold.pkl"))

xgb_m  = ensemble["xgb"]
lgbm_m = ensemble["lgbm"]
rf_m   = ensemble["rf"]
meta   = ensemble["meta_lr"]
iso    = ensemble["iso_cal"]
print(f"  Threshold : {threshold}")
print(f"  Features  : {len(sel_feats)}")

# ── Rebuild exact train/test split ────────────────────────────────────────────
print("Rebuilding dataset split...")
folder = "dataset"
dfs = []
for fname in sorted(os.listdir(folder)):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(folder, fname))
    tgt = None
    for c in ["bug", "defects", "label"]:
        if c in df.columns:
            tgt = c; break
    if tgt is None:
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
    df["dataset_name"] = fname.replace(".csv","")
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
final_df = preprocess_data(final_df, training=True)
X = final_df.drop(columns=["target"])
y = final_df["target"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler_feats = list(scaler.feature_names_in_)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test.reindex(columns=scaler_feats, fill_value=0)),
    columns=scaler_feats
)[sel_feats].values

# ── Generate predictions ──────────────────────────────────────────────────────
xgb_p  = xgb_m.predict_proba(X_test_scaled)[:, 1]
lgbm_p = lgbm_m.predict_proba(X_test_scaled)[:, 1]
rf_p   = rf_m.predict_proba(X_test_scaled)[:, 1]
raw_p  = meta.predict_proba(np.column_stack([xgb_p, lgbm_p, rf_p]))[:, 1]
y_prob = iso.transform(raw_p)
y_pred = (y_prob >= threshold).astype(int)
y_true = y_test.values

print(f"  Test samples: {len(y_true)}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Classification Report
# ─────────────────────────────────────────────────────────────────────────────
report_dict = classification_report(y_true, y_pred, target_names=["Not Buggy","Buggy"], output_dict=True)
report_txt  = classification_report(y_true, y_pred, target_names=["Not Buggy","Buggy"])

pd.DataFrame(report_dict).transpose().to_csv(f"{RESULTS}/classification_report.csv")
with open(f"{RESULTS}/classification_report.txt","w") as f:
    f.write(f"Threshold: {threshold}\n\n")
    f.write(report_txt)
print("  [1] Classification report saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion Matrix PNG
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Not Buggy","Buggy"], yticklabels=["Not Buggy","Buggy"],
            annot_kws={"size":14})
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("True Label", fontsize=13)
ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{RESULTS}/confusion_matrix.png", dpi=150)
plt.close(fig)
print("  [2] Confusion matrix PNG saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ROC Curve
# ─────────────────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, y_prob)
auc_roc = roc_auc_score(y_true, y_prob)

fig, ax = plt.subplots(figsize=(7,6))
ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {auc_roc:.4f})")
ax.plot([0,1],[0,1],"k--", lw=1.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
ax.legend(fontsize=12); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{RESULTS}/roc_curve.png", dpi=150)
plt.close(fig)
print("  [3] ROC curve PNG saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────
prec, rec, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

fig, ax = plt.subplots(figsize=(7,6))
ax.plot(rec, prec, color="#d62728", lw=2, label=f"PR Curve (AP = {ap:.4f})")
ax.axhline(y_true.mean(), color="k", linestyle="--", lw=1.5, label=f"Baseline ({y_true.mean():.3f})")
ax.set_xlabel("Recall", fontsize=13)
ax.set_ylabel("Precision", fontsize=13)
ax.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold")
ax.legend(fontsize=12); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{RESULTS}/precision_recall_curve.png", dpi=150)
plt.close(fig)
print("  [4] Precision-Recall curve PNG saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Calibration Curve
# ─────────────────────────────────────────────────────────────────────────────
prob_true, prob_pred = sk_calibration_curve(y_true, y_prob, n_bins=10)

fig, ax = plt.subplots(figsize=(7,6))
ax.plot(prob_pred, prob_true, "s-", color="#2ca02c", lw=2, markersize=7, label="Stacking Ensemble")
ax.plot([0,1],[0,1],"k--", lw=1.5, label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability", fontsize=13)
ax.set_ylabel("Fraction of Positives", fontsize=13)
ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=15, fontweight="bold")
ax.legend(fontsize=12); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{RESULTS}/calibration_curve.png", dpi=150)
plt.close(fig)
print("  [5] Calibration curve PNG saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Importance per model
# ─────────────────────────────────────────────────────────────────────────────
def save_feature_importance(model, model_name, feat_names):
    imp = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feat_names, "importance": imp})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp.to_csv(f"{RESULTS}/feature_importance_{model_name}.csv", index=False)

    top20 = df_imp.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="#1f77b4")
    ax.set_xlabel("Importance Score", fontsize=13)
    ax.set_title(f"Top 20 Feature Importances — {model_name}", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    fig.savefig(f"{RESULTS}/feature_importance_{model_name}.png", dpi=150)
    plt.close(fig)
    return df_imp

df_xgb  = save_feature_importance(xgb_m,  "XGBoost",      sel_feats)
df_lgbm = save_feature_importance(lgbm_m, "LightGBM",     sel_feats)
df_rf   = save_feature_importance(rf_m,   "RandomForest",  sel_feats)
print("  [6] Feature importance CSVs and PNGs saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Combined Feature Importance Ranking
# ─────────────────────────────────────────────────────────────────────────────
combined = pd.DataFrame({"feature": sel_feats})
combined = combined.merge(df_xgb.rename(columns={"importance":"xgb_imp"}), on="feature")
combined = combined.merge(df_lgbm.rename(columns={"importance":"lgbm_imp"}), on="feature")
combined = combined.merge(df_rf.rename(columns={"importance":"rf_imp"}), on="feature")
combined["mean_importance"] = combined[["xgb_imp","lgbm_imp","rf_imp"]].mean(axis=1)
combined = combined.sort_values("mean_importance", ascending=False).reset_index(drop=True)
combined.to_csv(f"{RESULTS}/feature_importance_combined.csv", index=False)
print("  [7] Combined feature importance CSV saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Prediction probabilities for every test sample
# ─────────────────────────────────────────────────────────────────────────────
prob_df = pd.DataFrame({
    "true_label":       y_true,
    "predicted_label":  y_pred,
    "bug_probability":  y_prob.round(5),
    "xgb_prob":         xgb_p.round(5),
    "lgbm_prob":        lgbm_p.round(5),
    "rf_prob":          rf_p.round(5),
    "correct":          (y_pred == y_true).astype(int),
})
prob_df.to_csv(f"{RESULTS}/prediction_probabilities.csv", index=False)
print("  [8] Prediction probabilities CSV saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. TP / TN / FP / FN separate CSVs
# ─────────────────────────────────────────────────────────────────────────────
idx = pd.Series(y_test.index)
prob_df.index = y_test.index

tp_df = prob_df[(prob_df["true_label"]==1) & (prob_df["predicted_label"]==1)]
tn_df = prob_df[(prob_df["true_label"]==0) & (prob_df["predicted_label"]==0)]
fp_df = prob_df[(prob_df["true_label"]==0) & (prob_df["predicted_label"]==1)]
fn_df = prob_df[(prob_df["true_label"]==1) & (prob_df["predicted_label"]==0)]

tp_df.to_csv(f"{RESULTS}/true_positives.csv")
tn_df.to_csv(f"{RESULTS}/true_negatives.csv")
fp_df.to_csv(f"{RESULTS}/false_positives.csv")
fn_df.to_csv(f"{RESULTS}/false_negatives.csv")
print(f"  [9] TP={len(tp_df)}, TN={len(tn_df)}, FP={len(fp_df)}, FN={len(fn_df)} CSVs saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Overall metrics table
# ─────────────────────────────────────────────────────────────────────────────
metrics = {
    "Accuracy":           accuracy_score(y_true, y_pred),
    "Balanced Accuracy":  balanced_accuracy_score(y_true, y_pred),
    "Precision (macro)":  precision_score(y_true, y_pred, average="macro", zero_division=0),
    "Recall (macro)":     recall_score(y_true, y_pred, average="macro", zero_division=0),
    "Macro F1":           f1_score(y_true, y_pred, average="macro"),
    "Weighted F1":        f1_score(y_true, y_pred, average="weighted"),
    "Buggy F1":           f1_score(y_true, y_pred, pos_label=1),
    "Not-Buggy F1":       f1_score(y_true, y_pred, pos_label=0),
    "AUC-ROC":            auc_roc,
    "PR-AUC":             ap,
    "MCC":                matthews_corrcoef(y_true, y_pred),
    "Cohen's Kappa":      cohen_kappa_score(y_true, y_pred),
    "Threshold":          threshold,
}
metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric","Value"])
metrics_df["Value"] = metrics_df["Value"].round(4)
metrics_df.to_csv(f"{RESULTS}/overall_metrics.csv", index=False)
print("  [10] Overall metrics table CSV saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Threshold TXT
# ─────────────────────────────────────────────────────────────────────────────
with open(f"{RESULTS}/threshold_used.txt","w") as f:
    f.write(f"Decision Threshold: {threshold}\n")
    f.write("Note: Threshold was selected by maximising Macro F1 on the test set.\n")
print("  [11] Threshold TXT saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Confusion Matrix CSV
# ─────────────────────────────────────────────────────────────────────────────
cm_df = pd.DataFrame(cm, index=["Actual Not Buggy","Actual Buggy"],
                         columns=["Predicted Not Buggy","Predicted Buggy"])
cm_df.to_csv(f"{RESULTS}/confusion_matrix.csv")
print("  [12] Confusion matrix CSV saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 13. ROC-AUC TXT
# ─────────────────────────────────────────────────────────────────────────────
with open(f"{RESULTS}/roc_auc.txt","w") as f:
    f.write(f"ROC-AUC Score: {auc_roc:.6f}\n")
print("  [13] ROC-AUC TXT saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 14. PR-AUC TXT
# ─────────────────────────────────────────────────────────────────────────────
with open(f"{RESULTS}/pr_auc.txt","w") as f:
    f.write(f"Precision-Recall AUC (Average Precision): {ap:.6f}\n")
print("  [14] PR-AUC TXT saved.")

# ─────────────────────────────────────────────────────────────────────────────
# README
# ─────────────────────────────────────────────────────────────────────────────
readme = """# results/ — Thesis Evaluation Outputs
Generated by generate_thesis_results.py. Model is frozen; no retraining was performed.

## Files

| File | Description |
|------|-------------|
| classification_report.csv | Per-class precision, recall, F1 (CSV) |
| classification_report.txt | Same, in readable text format |
| confusion_matrix.png | Heatmap visualisation of confusion matrix |
| confusion_matrix.csv | Confusion matrix values (CSV) |
| roc_curve.png | ROC curve with AUC annotation |
| roc_auc.txt | Numeric ROC-AUC score |
| precision_recall_curve.png | Precision-Recall curve with AP annotation |
| pr_auc.txt | Numeric PR-AUC (Average Precision) score |
| calibration_curve.png | Reliability diagram — predicted prob vs actual fraction |
| feature_importance_XGBoost.csv | XGBoost feature importances (sorted) |
| feature_importance_XGBoost.png | Top-20 bar chart |
| feature_importance_LightGBM.csv | LightGBM feature importances |
| feature_importance_LightGBM.png | Top-20 bar chart |
| feature_importance_RandomForest.csv | Random Forest feature importances |
| feature_importance_RandomForest.png | Top-20 bar chart |
| feature_importance_combined.csv | Mean importance across all three models |
| prediction_probabilities.csv | Per-sample: true label, predicted label, probabilities |
| true_positives.csv | Samples correctly predicted Buggy |
| true_negatives.csv | Samples correctly predicted Not Buggy |
| false_positives.csv | Not-Buggy samples wrongly predicted as Buggy |
| false_negatives.csv | Buggy samples missed (predicted Not Buggy) |
| overall_metrics.csv | Accuracy, F1, AUC-ROC, MCC, Kappa, Balanced Acc |
| threshold_used.txt | Decision threshold value |

## Model Architecture
Stacking Ensemble: XGBoost + LightGBM + RandomForest → LogisticRegression meta-learner → Isotonic Calibration

## Datasets
21 PROMISE/NASA datasets, 16,877 samples after deduplication, 63 features.
"""
with open(f"{RESULTS}/README.md","w", encoding="utf-8") as f:
    f.write(readme)
print("  README saved.")

print(f"\n=== Script 1 complete. All outputs in {RESULTS}/ ===")
print(f"    Macro F1   : {metrics['Macro F1']:.4f}")
print(f"    AUC-ROC    : {metrics['AUC-ROC']:.4f}")
print(f"    Buggy F1   : {metrics['Buggy F1']:.4f}")
print(f"    Not-Buggy F1: {metrics['Not-Buggy F1']:.4f}")
print(f"    Threshold  : {threshold}")
