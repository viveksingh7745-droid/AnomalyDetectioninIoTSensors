"""
predict.py
Simulates real-time IoT sensor anomaly detection with alerts.
Usage:
    python predict.py               ← demo on last 100 readings
    python predict.py --simulate    ← live stream simulation (Ctrl+C to stop)
    python predict.py --custom      ← enter sensor values manually
"""

import pickle, json, sys, time
import pandas as pd
import numpy as np

MODEL_FILE    = "models/isolation_forest.pkl"
SCALER_FILE   = "models/scaler.pkl"
FEATURES_FILE = "models/feature_names.pkl"
METRICS_FILE  = "models/metrics.json"
PRED_FILE     = "data/predictions.csv"

ALERT_LEVELS = {
    "CRITICAL": 0.75,
    "WARNING":  0.50,
    "WATCH":    0.35,
}

SENSOR_UNITS = {
    "temperature": "°C", "pressure": "bar",
    "vibration": "mm/s", "current_draw": "A",
}

SENSOR_THRESHOLDS = {           # beyond these → immediate flag
    "temperature":  {"warn": 80,  "critical": 95},
    "pressure":     {"warn": 4.0, "critical": 5.0},
    "vibration":    {"warn": 5.0, "critical": 8.0},
    "current_draw": {"warn": 28,  "critical": 35},
}


def load_model():
    with open(MODEL_FILE,    "rb") as f: model    = pickle.load(f)
    with open(SCALER_FILE,   "rb") as f: scaler   = pickle.load(f)
    with open(FEATURES_FILE, "rb") as f: features = pickle.load(f)
    with open(METRICS_FILE)        as f: metrics  = json.load(f)
    return model, scaler, features, metrics["best_threshold"]


def score_reading(model, scaler, features, reading: dict) -> float:
    """Return anomaly score for a single reading (higher = more anomalous)."""
    row = pd.DataFrame([reading])
    for col in features:
        if col not in row.columns:
            row[col] = 0.0
    X   = row[features].values
    X_s = scaler.transform(X)
    return float(-model.score_samples(X_s)[0])


def alert_level(score, threshold):
    ratio = score / (threshold + 1e-9)
    if ratio >= ALERT_LEVELS["CRITICAL"] / (threshold + 1e-9) * threshold:
        if score >= threshold * 1.5: return "🔴 CRITICAL"
        if score >= threshold * 1.0: return "🟠 WARNING"
        if score >= threshold * 0.7: return "🟡 WATCH"
    return "🟢 NORMAL"


def sensor_flags(reading: dict) -> list:
    flags = []
    for sensor, limits in SENSOR_THRESHOLDS.items():
        val = reading.get(sensor, 0)
        if val >= limits["critical"]:
            flags.append(f"  ⚠️  {sensor} = {val:.2f} {SENSOR_UNITS[sensor]}  [CRITICAL THRESHOLD]")
        elif val >= limits["warn"]:
            flags.append(f"  ⚡ {sensor} = {val:.2f} {SENSOR_UNITS[sensor]}  [WARNING THRESHOLD]")
    return flags


def print_reading(ts, reading, score, threshold, idx=None):
    level = alert_level(score, threshold)
    flags = sensor_flags(reading)
    label = f"[{idx}] " if idx is not None else ""
    print(f"\n  {label}{ts}")
    print(f"  Temp={reading.get('temperature',0):.1f}°C  "
          f"Press={reading.get('pressure',0):.2f}bar  "
          f"Vib={reading.get('vibration',0):.2f}mm/s  "
          f"Curr={reading.get('current_draw',0):.1f}A")
    print(f"  Anomaly score : {score:.4f}  |  Threshold : {threshold:.4f}")
    print(f"  Status        : {level}")
    for f in flags:
        print(f)


def run_demo(model, scaler, features, threshold):
    """Score the last 200 readings from the predictions CSV."""
    df = pd.read_csv(PRED_FILE, parse_dates=["timestamp"])
    df_sample = df.tail(200)

    print("\n" + "="*60)
    print("  IoT ANOMALY DETECTOR — Demo (last 200 readings)")
    print(f"  Threshold: {threshold:.4f}")
    print("="*60)

    anomalies_found = 0
    for _, row in df_sample.iterrows():
        score = row.get("anomaly_score", 0)
        is_anom = score >= threshold
        if is_anom:
            anomalies_found += 1
            reading = {s: row[s] for s in ["temperature","pressure","vibration","current_draw"]}
            print_reading(row["timestamp"], reading, score, threshold)

    print(f"\n  Summary: {anomalies_found} anomalies detected in last 200 readings")
    print("="*60)


def run_live_simulation(model, scaler, features, threshold):
    """Simulate a live sensor stream with occasional injected faults."""
    print("\n" + "="*60)
    print("  IoT ANOMALY DETECTOR — Live Simulation")
    print("  Press Ctrl+C to stop")
    print("="*60)

    np.random.seed(99)
    reading_n = 0
    fault_counter = 0

    base = {
        "temperature": 62.0, "pressure": 3.2,
        "vibration": 2.1,    "current_draw": 15.0,
        "hour_sin": 0.5, "hour_cos": 0.5,
        "is_working_hours": 1, "is_weekend": 0,
    }
    history_temp = [62.0] * 24

    try:
        while True:
            reading_n += 1
            r = base.copy()

            # Normal noise
            r["temperature"]  += np.random.normal(0, 1.5)
            r["pressure"]     += np.random.normal(0, 0.08)
            r["vibration"]    += np.random.normal(0, 0.2)
            r["current_draw"] += np.random.normal(0, 0.5)

            # Inject fault every ~25 readings
            fault_injected = ""
            if reading_n % 25 == 0:
                fault_type = np.random.choice(
                    ["temp_spike","pressure_drop","vibration_surge","current_overload"])
                if fault_type == "temp_spike":
                    r["temperature"]  += np.random.uniform(25, 40)
                    r["current_draw"] += np.random.uniform(8, 15)
                    fault_injected = "[INJECTED: temperature spike]"
                elif fault_type == "pressure_drop":
                    r["pressure"]  -= np.random.uniform(1.5, 2.2)
                    r["vibration"] += np.random.uniform(2, 4)
                    fault_injected = "[INJECTED: pressure drop]"
                elif fault_type == "vibration_surge":
                    r["vibration"]   += np.random.uniform(6, 12)
                    r["temperature"] += np.random.uniform(5, 12)
                    fault_injected = "[INJECTED: vibration surge]"
                elif fault_type == "current_overload":
                    r["current_draw"] += np.random.uniform(18, 28)
                    r["temperature"]  += np.random.uniform(10, 18)
                    fault_injected = "[INJECTED: current overload]"
                fault_counter += 1

            # Fill derived features with simple proxies
            history_temp.append(r["temperature"])
            history_temp = history_temp[-24:]
            r["temperature_lag1"]     = history_temp[-2] if len(history_temp) > 1 else r["temperature"]
            r["temperature_lag3"]     = history_temp[-4] if len(history_temp) > 3 else r["temperature"]
            r["pressure_lag1"]        = r["pressure"]
            r["vibration_lag1"]       = r["vibration"]
            r["current_draw_lag1"]    = r["current_draw"]
            r["pressure_lag3"]        = r["pressure"]
            r["vibration_lag3"]       = r["vibration"]
            r["current_draw_lag3"]    = r["current_draw"]
            roll_mean = np.mean(history_temp[-6:])
            roll_std  = max(np.std(history_temp[-6:]), 0.01)
            r["temperature_roll6_mean"]  = roll_mean
            r["temperature_roll6_std"]   = roll_std
            r["pressure_roll6_mean"]     = r["pressure"]
            r["vibration_roll6_mean"]    = r["vibration"]
            r["current_draw_roll6_mean"] = r["current_draw"]
            r["vibration_roll6_std"]     = 0.2
            r["temperature_roll24_mean"] = np.mean(history_temp)
            r["pressure_roll24_mean"]    = r["pressure"]
            r["vibration_roll24_mean"]   = r["vibration"]
            r["current_draw_roll24_mean"]= r["current_draw"]
            r["temperature_zscore"]  = (r["temperature"] - roll_mean) / roll_std
            r["pressure_zscore"]     = 0.0
            r["vibration_zscore"]    = 0.0
            r["current_draw_zscore"] = 0.0
            r["temp_pressure_ratio"] = r["temperature"] / max(r["pressure"], 0.001)
            r["power_index"]         = r["current_draw"] * r["temperature"] / 1000
            r["mech_stress"]         = r["vibration"] * r["pressure"]

            score = score_reading(model, scaler, features, r)
            ts    = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

            if fault_injected:
                print(f"\n  {'─'*52}")
                print(f"  🚨 {fault_injected}")

            print_reading(ts, r, score, threshold, idx=reading_n)
            time.sleep(1.5)

    except KeyboardInterrupt:
        print(f"\n\n  Simulation stopped after {reading_n} readings, {fault_counter} faults injected.")


def run_custom(model, scaler, features, threshold):
    print("\n" + "="*60)
    print("  IoT ANOMALY DETECTOR — Custom Input")
    print("  Type 'quit' to exit")
    print("="*60)
    while True:
        print()
        try:
            inp = input("  Temperature (°C)   : ").strip()
            if inp.lower() in ("quit","q","exit"): break
            temp  = float(inp)
            press = float(input("  Pressure (bar)     : "))
            vib   = float(input("  Vibration (mm/s)   : "))
            curr  = float(input("  Current draw (A)   : "))
        except (ValueError, KeyboardInterrupt):
            print("  Invalid input."); continue

        r = {"temperature": temp, "pressure": press,
             "vibration": vib, "current_draw": curr}
        # fill defaults for derived features
        for feat in features:
            if feat not in r: r[feat] = 0.0
        r["temp_pressure_ratio"] = temp / max(press, 0.001)
        r["power_index"]         = curr * temp / 1000
        r["mech_stress"]         = vib * press

        score = score_reading(model, scaler, features, r)
        ts    = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        print_reading(ts, r, score, threshold)


def main():
    model, scaler, features, threshold = load_model()
    if "--simulate" in sys.argv:
        run_live_simulation(model, scaler, features, threshold)
    elif "--custom" in sys.argv:
        run_custom(model, scaler, features, threshold)
    else:
        run_demo(model, scaler, features, threshold)
        print("\nTips:")
        print("  python predict.py --simulate   ← live stream with random faults")
        print("  python predict.py --custom     ← enter your own sensor readings")


if __name__ == "__main__":
    main()
