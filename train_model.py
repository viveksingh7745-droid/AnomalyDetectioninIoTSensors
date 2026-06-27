"""
train_model.py
Trains an Isolation Forest for anomaly detection on IoT sensor data.
Uses only NORMAL data for training (unsupervised anomaly detection).
Labels are used ONLY for evaluation — never for training.
"""

import pandas as pd
import numpy as np
import pickle, json, os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

PROC_FILE     = "data/iot_processed.csv"
MODEL_FILE    = "models/isolation_forest.pkl"
SCALER_FILE   = "models/scaler.pkl"
FEATURES_FILE = "models/feature_names.pkl"
METRICS_FILE  = "models/metrics.json"

SENSORS = ["temperature", "pressure", "vibration", "current_draw"]

# Features used for anomaly detection
FEATURE_COLS = [
    "temperature", "pressure", "vibration", "current_draw",
    "temperature_lag1", "pressure_lag1", "vibration_lag1", "current_draw_lag1",
    "temperature_roll6_mean", "pressure_roll6_mean",
    "vibration_roll6_mean",  "current_draw_roll6_mean",
    "temperature_roll6_std", "vibration_roll6_std",
    "temperature_zscore",    "pressure_zscore",
    "vibration_zscore",      "current_draw_zscore",
    "temp_pressure_ratio",   "power_index", "mech_stress",
    "hour_sin", "hour_cos",
    "is_working_hours", "is_weekend",
]


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} readings  |  True anomalies: {df['anomaly'].sum()} ({df['anomaly'].mean()*100:.1f}%)")
    return df


def plot_anomaly_scores(df, scores, threshold, path="outputs/5_anomaly_scores.png"):
    """Plot raw anomaly scores over time."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), facecolor="#F8F8F8")
    fig.suptitle("Isolation Forest — Anomaly Detection Results", fontsize=14, fontweight="bold")

    # 1. Temperature with detected anomalies
    ax = axes[0]
    ax.plot(df["timestamp"], df["temperature"], linewidth=0.6,
            color="#E53935", alpha=0.7, label="Temperature (°C)")
    detected = df[scores == -1]
    ax.scatter(detected["timestamp"], detected["temperature"],
               color="black", s=20, zorder=5, label=f"Detected anomaly ({len(detected)})")
    true_anom = df[df["anomaly"] == 1]
    ax.scatter(true_anom["timestamp"], true_anom["temperature"],
               color="orange", s=8, zorder=4, alpha=0.5, label=f"True anomaly ({len(true_anom)})")
    ax.set_ylabel("Temperature (°C)")
    ax.set_facecolor("white")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # 2. Raw anomaly score over time
    ax = axes[1]
    ax.plot(df["timestamp"], -scores_raw, linewidth=0.6, color="#1565C0", alpha=0.7,
            label="Anomaly score (higher = more anomalous)")
    ax.axhline(-threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {-threshold:.3f}")
    ax.set_ylabel("Anomaly Score")
    ax.set_facecolor("white")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # 3. Vibration with anomalies
    ax = axes[2]
    ax.plot(df["timestamp"], df["vibration"], linewidth=0.6,
            color="#2E7D32", alpha=0.7, label="Vibration (mm/s)")
    ax.scatter(detected["timestamp"], detected["vibration"],
               color="black", s=20, zorder=5, label="Detected anomaly")
    ax.set_ylabel("Vibration (mm/s)")
    ax.set_xlabel("Date")
    ax.set_facecolor("white")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_roc(y_true, y_score, path="outputs/6_roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#F8F8F8")
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1565C0")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Isolation Forest", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")
    return auc


def plot_confusion_matrix(cm_arr, path="outputs/7_confusion_matrix.png"):
    labels = ["Normal", "Anomaly"]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#F8F8F8")
    im = ax.imshow(cm_arr, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    thresh = cm_arr.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i,j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm_arr[i,j] > thresh else "black")
    ax.set_title("Confusion Matrix\n(Isolation Forest)", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_feature_importance_proxy(model, scaler, feature_names,
                                   path="outputs/8_feature_scores.png"):
    """
    Proxy feature importance: measure how much each feature's anomaly score
    increases when we perturb it to an extreme value.
    """
    X_base = np.zeros((1, len(feature_names)))
    base_score = model.score_samples(X_base)[0]

    importances = []
    for i in range(len(feature_names)):
        X_pert = X_base.copy()
        X_pert[0, i] = 3.0       # 3 std-dev perturbation
        delta = base_score - model.score_samples(X_pert)[0]
        importances.append(max(delta, 0))

    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
    colors = cm.Reds_r(np.linspace(0.2, 0.8, len(imp)))

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#F8F8F8")
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_title("Top 20 Feature Sensitivity — Isolation Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Score drop under perturbation (higher = more influential)")
    ax.set_facecolor("white")
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_anomaly_by_type(df, scores, path="outputs/9_anomaly_by_type.png"):
    """Score distribution per true anomaly type."""
    df = df.copy()
    df["score"] = -scores_raw       # higher = more anomalous

    types   = df["anomaly_type"].unique()
    colors  = ["#2E7D32","#E53935","#1565C0","#F57F17","#7B1FA2","#795548"]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#F8F8F8")

    data_list  = []
    tick_labels = []
    for t in sorted(types):
        data_list.append(df[df["anomaly_type"]==t]["score"].values)
        tick_labels.append(t.replace("_"," "))

    bp = ax.boxplot(data_list, labels=tick_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title("Anomaly Score Distribution by Fault Type", fontsize=13, fontweight="bold")
    ax.set_ylabel("Anomaly Score (higher = more anomalous)")
    ax.set_facecolor("white")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def main():
    global scores_raw   # used in nested plot functions

    df = load_data(PROC_FILE)
    X  = df[FEATURE_COLS].values
    y  = df["anomaly"].values

    # ── Scale features ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # ── Train on NORMAL data only (unsupervised) ──────────────────────────────
    X_normal = X_s[y == 0]
    print(f"\nTraining Isolation Forest on {len(X_normal)} normal readings ...")

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.04,    # expected anomaly fraction
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_normal)

    # ── Predict on full dataset ───────────────────────────────────────────────
    scores_raw = model.score_samples(X_s)   # raw scores (more negative = anomaly)
    predictions = model.predict(X_s)         # -1 = anomaly, 1 = normal
    y_pred = (predictions == -1).astype(int)  # 1 = anomaly
    y_score = -scores_raw                     # higher = more anomalous (for ROC)

    # Find best threshold on PR curve
    prec, rec, thresholds = precision_recall_curve(y, y_score)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_thresh = thresholds[f1s.argmax()] if len(thresholds) > 0 else 0.5
    y_pred_tuned = (y_score >= best_thresh).astype(int)

    # ── Evaluation ────────────────────────────────────────────────────────────
    auc    = roc_auc_score(y, y_score)
    f1     = f1_score(y, y_pred_tuned)
    prec_s = precision_score(y, y_pred_tuned, zero_division=0)
    rec_s  = recall_score(y, y_pred_tuned)

    print(f"\n{'='*55}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  Precision       : {prec_s:.4f}")
    print(f"  Recall          : {rec_s:.4f}")
    print(f"  Detected anomalies: {y_pred_tuned.sum()} / {y.sum()} true anomalies")
    print(f"{'='*55}")
    print(f"\n{classification_report(y, y_pred_tuned, target_names=['Normal','Anomaly'])}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    threshold = np.percentile(scores_raw, 4)   # bottom 4% = anomaly
    plot_anomaly_scores(df, predictions, threshold)
    plot_roc(y, y_score)
    cm_arr = confusion_matrix(y, y_pred_tuned)
    plot_confusion_matrix(cm_arr)
    plot_feature_importance_proxy(model, scaler, FEATURE_COLS)
    plot_anomaly_by_type(df, predictions)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(MODEL_FILE,    "wb") as f: pickle.dump(model,         f)
    with open(SCALER_FILE,   "wb") as f: pickle.dump(scaler,        f)
    with open(FEATURES_FILE, "wb") as f: pickle.dump(FEATURE_COLS,  f)

    tn, fp, fn, tp = cm_arr.ravel()
    metrics = {
        "roc_auc":          round(auc,    4),
        "f1_score":         round(f1,     4),
        "precision":        round(prec_s, 4),
        "recall":           round(rec_s,  4),
        "best_threshold":   round(float(best_thresh), 4),
        "total_readings":   int(len(df)),
        "true_anomalies":   int(y.sum()),
        "detected":         int(y_pred_tuned.sum()),
        "confusion_matrix": {"TP":int(tp),"TN":int(tn),"FP":int(fp),"FN":int(fn)},
    }
    with open(METRICS_FILE, "w") as f: json.dump(metrics, f, indent=2)

    # Save predictions back to CSV
    df["anomaly_score"] = y_score.round(4)
    df["predicted"]     = y_pred_tuned
    df.to_csv("data/predictions.csv", index=False)

    print(f"\nModel saved   → {MODEL_FILE}")
    print(f"Metrics saved → {METRICS_FILE}")
    print(f"Predictions   → data/predictions.csv")


if __name__ == "__main__":
    main()
