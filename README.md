# IoT Anomaly Detection — CodTech Internship Project

Detects equipment faults and sensor anomalies in real-time using
**Isolation Forest** on multi-sensor industrial IoT data.
No external dataset or internet connection required.

## Problem Statement
Industrial machines generate continuous sensor streams. Undetected
anomalies lead to unplanned downtime and costly failures. This system
learns what "normal" looks like and raises alerts when readings deviate.

## Sensors Monitored
| Sensor | Unit | Normal Range |
|---|---|---|
| Temperature | °C | 45 – 75 |
| Pressure | bar | 2.5 – 4.0 |
| Vibration | mm/s | 0.5 – 4.0 |
| Current Draw | A | 8 – 25 |

## Anomaly Types Detected
- **Temperature spike** — overheating, cooling failure
- **Pressure drop** — pipe leak, valve failure
- **Vibration surge** — bearing wear, mechanical imbalance
- **Current overload** — electrical fault, motor strain
- **Multi-sensor drift** — general system degradation

## Project Structure
```
iot_anomaly/
├── generate_data.py       ← 30 days of synthetic sensor readings + injected faults
├── preprocess.py          ← feature engineering + 4 EDA charts
├── train_model.py         ← Isolation Forest + 5 evaluation charts
├── predict.py             ← demo / live stream / custom input modes
├── run_pipeline.py        ← runs all 4 steps in one command
├── requirements.txt
├── data/                  ← generated CSVs
├── models/                ← saved model, scaler, features, metrics
└── outputs/               ← 9 PNG charts
```

## Setup & Run
```bash
pip install -r requirements.txt

# Full pipeline (recommended)
python run_pipeline.py

# Or step by step
python generate_data.py
python preprocess.py
python train_model.py
python predict.py

# Live simulation with random fault injection
python predict.py --simulate

# Enter your own sensor readings
python predict.py --custom
```

## How It Works

### Training (Unsupervised)
Isolation Forest is trained **only on normal data** — it learns the
boundary of normal behaviour without ever seeing labeled anomalies.
At prediction time, readings that require fewer splits to isolate
get a high anomaly score.

### Feature Engineering (25+ features)
- Raw sensor values
- Lag features (t-1, t-3)
- Rolling mean & std (30-min and 2-hr windows)
- Rolling Z-score (deviation from local norm)
- Cross-sensor features: temp/pressure ratio, power index, mechanical stress
- Time features: hour sin/cos, working hours flag, weekend flag

### Alert Levels
| Score vs Threshold | Alert |
|---|---|
| ≥ 1.5× threshold | 🔴 CRITICAL |
| ≥ 1.0× threshold | 🟠 WARNING |
| ≥ 0.7× threshold | 🟡 WATCH |
| < 0.7× threshold | 🟢 NORMAL |

## Output Charts
| File | Description |
|---|---|
| `1_sensor_overview.png` | 30-day time-series, all 4 sensors with anomalies highlighted |
| `2_distributions.png` | Histogram + boxplot: normal vs anomaly per sensor |
| `3_correlations.png` | Sensor correlation heatmap |
| `4_anomaly_types.png` | Fault type pie chart + hourly anomaly rate |
| `5_anomaly_scores.png` | Anomaly score timeline + detections |
| `6_roc_curve.png` | ROC-AUC curve |
| `7_confusion_matrix.png` | TP / TN / FP / FN |
| `8_feature_scores.png` | Feature sensitivity analysis |
| `9_anomaly_by_type.png` | Score distribution per fault type |
