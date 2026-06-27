"""
run_pipeline.py
Runs the complete IoT Anomaly Detection pipeline:
  1. Generate data  →  2. Preprocess + EDA  →  3. Train  →  4. Predict demo
"""

import subprocess, sys, os

STEPS = [
    ("generate_data.py", "Step 1/4 — Generating IoT sensor dataset"),
    ("preprocess.py",    "Step 2/4 — Preprocessing & EDA"),
    ("train_model.py",   "Step 3/4 — Training Isolation Forest"),
    ("predict.py",       "Step 4/4 — Demo anomaly detection"),
]

def banner(msg):
    print("\n" + "=" * 62)
    print(f"  {msg}")
    print("=" * 62)

def run(script, label):
    banner(label)
    r = subprocess.run([sys.executable, script], capture_output=False)
    if r.returncode != 0:
        print(f"\n❌  {script} failed."); sys.exit(1)

def main():
    banner("IoT Anomaly Detection — Pipeline Starting")
    for d in ["data","models","outputs"]: os.makedirs(d, exist_ok=True)
    for script, label in STEPS:
        run(script, label)

    banner("Pipeline complete! 🎉")
    print("\nOutputs:")
    print("  data/iot_sensors.csv           ← raw 30-day sensor readings (8640 rows)")
    print("  data/iot_processed.csv         ← engineered features (25+ features)")
    print("  data/predictions.csv           ← anomaly scores + predictions")
    print("  models/isolation_forest.pkl    ← trained Isolation Forest")
    print("  models/scaler.pkl              ← StandardScaler")
    print("  models/metrics.json            ← ROC-AUC, F1, precision, recall")
    print("  outputs/1_sensor_overview.png  ← 30-day time-series with anomalies")
    print("  outputs/2_distributions.png    ← normal vs anomaly distributions")
    print("  outputs/3_correlations.png     ← sensor correlation heatmap")
    print("  outputs/4_anomaly_types.png    ← fault type breakdown")
    print("  outputs/5_anomaly_scores.png   ← score timeline + detections")
    print("  outputs/6_roc_curve.png        ← ROC-AUC curve")
    print("  outputs/7_confusion_matrix.png ← TP/TN/FP/FN")
    print("  outputs/8_feature_scores.png   ← feature sensitivity")
    print("  outputs/9_anomaly_by_type.png  ← score by fault type")
    print("\nAdditional modes:")
    print("  python predict.py --simulate   ← live stream with injected faults")
    print("  python predict.py --custom     ← enter your own sensor readings")

if __name__ == "__main__":
    main()
