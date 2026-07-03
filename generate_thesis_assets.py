"""
generate_thesis_assets.py
=========================
Publication-quality figures at 300 DPI for Master's thesis.
Large fonts, clean layout, proper axis labels and legends.
Saved in thesis_assets/
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve as sk_cal_curve

warnings.filterwarnings("ignore")
from preprocessing import preprocess_data

ASSETS = "thesis_assets"
os.makedirs(ASSETS, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        13,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
PALETTE = {"buggy":"#d62728","not_buggy":"#1f77b4","neutral":"#2ca02c","highlight":"#ff7f0e"}

# ── Load artifacts ────────────────────────────────────────────────────────────
ensemble  = joblib.load("model.pkl")
scaler    = joblib.load("scaler.pkl")
sel_feats = joblib.load("selected_features.pkl")
threshold = float(joblib.load("threshold.pkl"))

xgb_m  = ensemble["xgb"]
lgbm_m = ensemble["lgbm"]
rf_m   = ensemble["rf"]
meta   = ensemble["meta_lr"]
iso    = ensemble["iso_cal"]

# ── Rebuild split ─────────────────────────────────────────────────────────────
folder = "dataset"
dfs = []
for fname in sorted(os.listdir(folder)):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(folder, fname))
    tgt = None
    for c in ["bug","defects","label"]:
        if c in df.columns: tgt=c; break
    if tgt is None: continue
    col=df[tgt]; dtype=str(col.dtype)
    if "bool" in dtype:
        df[tgt]=col.astype(int)
    elif "str" in dtype or "object" in dtype or "string" in dtype:
        vals=col.astype(str).str.strip().str.upper()
        if set(vals.unique()).issubset({"Y","N"}): df[tgt]=vals.map({"Y":1,"N":0})
        elif set(vals.unique()).issubset({"TRUE","FALSE"}): df[tgt]=vals.map({"TRUE":1,"FALSE":0})
        else:
            df[tgt]=pd.to_numeric(vals,errors="coerce").fillna(0)
            df[tgt]=(df[tgt]>0).astype(int)
    else:
        df[tgt]=pd.to_numeric(col,errors="coerce").fillna(0)
        df[tgt]=(df[tgt]>0).astype(int)
    df.rename(columns={tgt:"target"},inplace=True)
    df["dataset_name"]=fname.replace(".csv","")
    dfs.append(df)

final_df = pd.concat(dfs,ignore_index=True).drop_duplicates().reset_index(drop=True)
final_df = preprocess_data(final_df, training=True)
X = final_df.drop(columns=["target"])
y = final_df["target"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler_feats = list(scaler.feature_names_in_)
X_te = pd.DataFrame(
    scaler.transform(X_test.reindex(columns=scaler_feats, fill_value=0)),
    columns=scaler_feats)[sel_feats].values

xgb_p  = xgb_m.predict_proba(X_te)[:, 1]
lgbm_p = lgbm_m.predict_proba(X_te)[:, 1]
rf_p   = rf_m.predict_proba(X_te)[:, 1]
raw_p  = meta.predict_proba(np.column_stack([xgb_p, lgbm_p, rf_p]))[:, 1]
y_prob = iso.transform(raw_p)
y_pred = (y_prob >= threshold).astype(int)
y_true = y_test.values

# ─────────────────────────────────────────────────────────────────────────────
# 1. ROC Curve
# ─────────────────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=PALETTE["buggy"], lw=2.5,
        label=f"Stacking Ensemble (AUC = {auc:.4f})")
ax.plot([0,1],[0,1], "k--", lw=1.5, alpha=0.6, label="Random Classifier (AUC = 0.50)")
ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["buggy"])
ax.set_xlabel("False Positive Rate", labelpad=8)
ax.set_ylabel("True Positive Rate", labelpad=8)
ax.set_title("Receiver Operating Characteristic (ROC) Curve", pad=12)
ax.legend(loc="lower right")
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(f"{ASSETS}/roc_curve_300dpi.png")
plt.close(fig)
print("[1] ROC Curve saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────
prec, rec, thresholds_pr = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
baseline = y_true.mean()

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec, prec, color=PALETTE["neutral"], lw=2.5,
        label=f"Stacking Ensemble (AP = {ap:.4f})")
ax.axhline(baseline, color="k", linestyle="--", lw=1.5, alpha=0.6,
           label=f"No-Skill Baseline ({baseline:.3f})")
ax.fill_between(rec, prec, baseline, alpha=0.08, color=PALETTE["neutral"])
ax.set_xlabel("Recall", labelpad=8)
ax.set_ylabel("Precision", labelpad=8)
ax.set_title("Precision-Recall Curve", pad=12)
ax.legend(loc="upper right")
ax.set_xlim([0, 1.01]); ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(f"{ASSETS}/precision_recall_curve_300dpi.png")
plt.close(fig)
print("[2] Precision-Recall Curve saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

labels = np.array([[f"{v}\n({p:.1f}%)" for v,p in zip(row_v, row_p)]
                   for row_v, row_p in zip(cm, cm_pct)])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax,
            xticklabels=["Not Buggy","Buggy"],
            yticklabels=["Not Buggy","Buggy"],
            annot_kws={"size":13, "weight":"bold"},
            linewidths=0.5, linecolor="gray")
ax.set_xlabel("Predicted Label", labelpad=10)
ax.set_ylabel("True Label", labelpad=10)
ax.set_title("Confusion Matrix", pad=12)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
fig.savefig(f"{ASSETS}/confusion_matrix_300dpi.png")
plt.close(fig)
print("[3] Confusion Matrix saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibration Curve
# ─────────────────────────────────────────────────────────────────────────────
prob_true, prob_pred = sk_cal_curve(y_true, y_prob, n_bins=10)

fig, axes = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios":[3,1]})

ax = axes[0]
ax.plot(prob_pred, prob_true, "s-", color=PALETTE["neutral"], lw=2.5,
        markersize=8, label="Stacking Ensemble (isotonic calibrated)")
ax.plot([0,1],[0,1],"k--",lw=1.5,alpha=0.6,label="Perfect Calibration")
ax.set_ylabel("Fraction of Positives", labelpad=8)
ax.set_title("Calibration Curve (Reliability Diagram)", pad=12)
ax.legend(loc="upper left"); ax.grid(True, alpha=0.25)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])

axes[1].hist(y_prob, range=(0,1), bins=20, color=PALETTE["highlight"], alpha=0.7, edgecolor="k", lw=0.5)
axes[1].set_xlabel("Mean Predicted Probability", labelpad=8)
axes[1].set_ylabel("Count")
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(f"{ASSETS}/calibration_curve_300dpi.png")
plt.close(fig)
print("[4] Calibration Curve saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature Importance Bar Chart (top 20 mean importance)
# ─────────────────────────────────────────────────────────────────────────────
combined = pd.read_csv("results/feature_importance_combined.csv").head(20)
colors = [PALETTE["buggy"] if i < 5 else PALETTE["not_buggy"] if i < 10 else "#aec7e8"
          for i in range(len(combined))]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(combined["feature"][::-1], combined["mean_importance"][::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Mean Feature Importance (XGBoost + LightGBM + RandomForest)", labelpad=8)
ax.set_title("Top 20 Features by Mean Ensemble Importance", pad=12)
ax.grid(True, axis="x", alpha=0.25)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.0002, bar.get_y() + bar.get_height()/2,
            f"{w:.4f}", va="center", ha="left", fontsize=9)
plt.tight_layout()
fig.savefig(f"{ASSETS}/feature_importance_top20_300dpi.png")
plt.close(fig)
print("[5] Feature Importance Bar Chart saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Class Distribution Chart (full dataset + train/test breakdown)
# ─────────────────────────────────────────────────────────────────────────────
X_train_all = final_df.drop(columns=["target"])
y_all = final_df["target"]
_, _, y_train_split, y_test_split = train_test_split(
    X_train_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

splits = {
    "Full Dataset": y_all,
    "Train Set (80%)": y_train_split,
    "Test Set (20%)":  y_test_split,
}

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, (name, s) in zip(axes, splits.items()):
    counts = s.value_counts().sort_index()
    labels_pie = ["Not Buggy","Buggy"]
    colors_pie  = [PALETTE["not_buggy"], PALETTE["buggy"]]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels_pie, colors=colors_pie,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize":12}, pctdistance=0.75,
    )
    for at in autotexts: at.set_fontsize(11); at.set_fontweight("bold")
    ax.set_title(f"{name}\n(n={len(s):,})", fontsize=13, fontweight="bold", pad=8)
plt.suptitle("Class Distribution Across Dataset Splits", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{ASSETS}/class_distribution_300dpi.png", bbox_inches="tight")
plt.close(fig)
print("[6] Class Distribution Chart saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Prediction Probability Distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
bins = np.linspace(0, 1, 41)

ax.hist(y_prob[y_true==0], bins=bins, alpha=0.65, color=PALETTE["not_buggy"],
        label="Not Buggy (Actual)", edgecolor="white", lw=0.4)
ax.hist(y_prob[y_true==1], bins=bins, alpha=0.65, color=PALETTE["buggy"],
        label="Buggy (Actual)", edgecolor="white", lw=0.4)
ax.axvline(threshold, color="black", linestyle="--", lw=2,
           label=f"Decision Threshold ({threshold:.3f})")
ax.set_xlabel("Predicted Bug Probability", labelpad=8)
ax.set_ylabel("Number of Samples", labelpad=8)
ax.set_title("Distribution of Predicted Bug Probabilities by True Class", pad=12)
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(f"{ASSETS}/probability_distribution_300dpi.png")
plt.close(fig)
print("[7] Prediction Probability Distribution saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Feature Correlation Heatmap (top 20-25 features by importance)
# ─────────────────────────────────────────────────────────────────────────────
combined_all = pd.read_csv("results/feature_importance_combined.csv")
top_feats = combined_all.head(22)["feature"].tolist()

X_test_df = pd.DataFrame(X_te, columns=sel_feats)
corr_df = X_test_df[top_feats].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
sns.heatmap(
    corr_df, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1,
    ax=ax, mask=mask,
    annot_kws={"size": 7.5},
    linewidths=0.3, linecolor="lightgray",
    square=True,
)
ax.set_title("Feature Correlation Heatmap (Top 22 Features by Importance)", pad=14)
ax.tick_params(axis="x", rotation=45, labelsize=10)
ax.tick_params(axis="y", rotation=0,  labelsize=10)
plt.tight_layout()
fig.savefig(f"{ASSETS}/feature_correlation_heatmap_300dpi.png", bbox_inches="tight")
plt.close(fig)
print("[8] Feature Correlation Heatmap saved (top 22 features).")

print(f"\n=== Script 3 complete. Publication-quality figures in {ASSETS}/ ===")
print("    All figures saved at 300 DPI, ready for thesis submission.")
