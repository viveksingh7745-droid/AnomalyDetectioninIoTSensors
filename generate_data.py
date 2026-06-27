"""
generate_data.py
Generates a realistic multi-sensor IoT dataset for an industrial machine.
Sensors: temperature, pressure, vibration, current_draw
Injects 5 realistic anomaly types into the data.
Saves: data/iot_sensors.csv
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_DAYS        = 30          # days of data
FREQ_MINUTES  = 5           # reading every 5 minutes
ANOMALY_RATE  = 0.04        # ~4% anomalies

# ── Time index ────────────────────────────────────────────────────────────────
n_points = N_DAYS * 24 * 60 // FREQ_MINUTES
timestamps = pd.date_range("2024-01-01", periods=n_points, freq=f"{FREQ_MINUTES}min")
n = len(timestamps)
print(f"Generating {n} readings over {N_DAYS} days ...")

hours   = timestamps.hour
minutes = timestamps.minute
t       = np.arange(n)

# ── Normal signal components ──────────────────────────────────────────────────
# Diurnal cycle (machine heats up during work hours 6–22)
work_cycle = np.sin(2 * np.pi * (hours - 6) / 24) * 0.5 + 0.5
work_cycle = np.clip(work_cycle, 0, 1)

# Weekly pattern (weekends slightly cooler / quieter)
weekday    = (timestamps.dayofweek < 5).astype(float)

# ── Sensor 1: Temperature (°C) ────────────────────────────────────────────────
# Base: 45–75°C depending on load; slight upward drift over 30 days
temp_base  = 55 + 15 * work_cycle * weekday + 0.03 * t / n * 10
temp_noise = np.random.normal(0, 1.2, n)
temperature = temp_base + temp_noise

# ── Sensor 2: Pressure (bar) ──────────────────────────────────────────────────
# Correlated with temperature but with its own noise
pressure_base  = 2.8 + 0.8 * work_cycle * weekday + 0.15 * np.sin(2*np.pi*t/720)
pressure_noise = np.random.normal(0, 0.08, n)
pressure = pressure_base + pressure_noise

# ── Sensor 3: Vibration (mm/s) ───────────────────────────────────────────────
# Higher during peak load, correlated with work cycle
vibration_base  = 1.5 + 2.0 * work_cycle * weekday
vibration_noise = np.random.normal(0, 0.2, n)
vibration = np.abs(vibration_base + vibration_noise)

# ── Sensor 4: Current Draw (A) ───────────────────────────────────────────────
# Strongly tied to work cycle; has startup spikes at shift start
current_base  = 8 + 12 * work_cycle * weekday
current_noise = np.random.normal(0, 0.5, n)
current_draw  = current_base + current_noise

# ── Derived features ──────────────────────────────────────────────────────────
# Rolling stats (last 6 readings = 30 min)
df_temp  = pd.Series(temperature)
temp_roll_mean = df_temp.rolling(6, min_periods=1).mean().values
temp_roll_std  = df_temp.rolling(6, min_periods=1).std().fillna(0).values

# Hour of day and day of week as features
hour_of_day  = hours.values if hasattr(hours, 'values') else np.array(hours)
day_of_week  = timestamps.dayofweek.values

# ── Assemble clean DataFrame ──────────────────────────────────────────────────
df = pd.DataFrame({
    "timestamp":       timestamps,
    "temperature":     temperature.round(2),
    "pressure":        pressure.round(3),
    "vibration":       vibration.round(3),
    "current_draw":    current_draw.round(2),
    "temp_roll_mean":  temp_roll_mean.round(2),
    "temp_roll_std":   temp_roll_std.round(3),
    "hour_of_day":     hour_of_day,
    "day_of_week":     day_of_week,
    "anomaly":         0,        # 0 = normal
    "anomaly_type":    "normal",
})

# ── Inject anomalies ──────────────────────────────────────────────────────────
n_anomalies = int(n * ANOMALY_RATE)
anomaly_indices = np.random.choice(n, size=n_anomalies, replace=False)

ANOMALY_TYPES = {
    "temperature_spike":   0.25,   # sudden heat spike
    "pressure_drop":       0.20,   # pressure loss (possible leak)
    "vibration_surge":     0.20,   # bearing failure signature
    "current_overload":    0.20,   # electrical overload
    "multi_sensor_drift":  0.15,   # slow drift across all sensors
}

type_choices = np.random.choice(
    list(ANOMALY_TYPES.keys()),
    size=n_anomalies,
    p=list(ANOMALY_TYPES.values())
)

for idx, atype in zip(anomaly_indices, type_choices):
    df.at[idx, "anomaly"]      = 1
    df.at[idx, "anomaly_type"] = atype

    if atype == "temperature_spike":
        df.at[idx, "temperature"]  += np.random.uniform(20, 45)   # sudden heat
        df.at[idx, "current_draw"] += np.random.uniform(5, 12)    # usually draws more current

    elif atype == "pressure_drop":
        df.at[idx, "pressure"]     -= np.random.uniform(1.2, 2.0) # pressure loss
        df.at[idx, "vibration"]    += np.random.uniform(1.0, 2.5) # cavitation vibration

    elif atype == "vibration_surge":
        df.at[idx, "vibration"]    += np.random.uniform(5, 12)    # bearing/mechanical failure
        df.at[idx, "temperature"]  += np.random.uniform(5, 15)    # friction heat

    elif atype == "current_overload":
        df.at[idx, "current_draw"] += np.random.uniform(15, 30)   # electrical overload
        df.at[idx, "temperature"]  += np.random.uniform(8, 20)    # heat from overload

    elif atype == "multi_sensor_drift":
        df.at[idx, "temperature"]  += np.random.uniform(10, 20)
        df.at[idx, "pressure"]     += np.random.uniform(0.5, 1.5)
        df.at[idx, "vibration"]    += np.random.uniform(2, 5)
        df.at[idx, "current_draw"] += np.random.uniform(5, 10)

df.to_csv("data/iot_sensors.csv", index=False)

print(f"Dataset saved → data/iot_sensors.csv")
print(f"Total readings : {n}")
print(f"Anomalies      : {df['anomaly'].sum()} ({df['anomaly'].mean()*100:.1f}%)")
print(f"\nAnomaly type breakdown:")
print(df[df["anomaly"]==1]["anomaly_type"].value_counts().to_string())
print(f"\nSensor stats (normal readings only):")
normal = df[df["anomaly"]==0]
for col in ["temperature","pressure","vibration","current_draw"]:
    print(f"  {col:<18}: mean={normal[col].mean():.2f}  std={normal[col].std():.2f}  "
          f"min={normal[col].min():.2f}  max={normal[col].max():.2f}")
