"""
preprocess.py
Loads raw IoT sensor data, runs EDA, engineers features, and saves processed CSV.
Outputs: data/iot_processed.csv  +  outputs/1_eda_*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

os.makedirs("outputs", exist_ok=True)

RAW_FILE  = "data/iot_sensors.csv"
OUT_FILE  = "data/iot_processed.csv"

SENSORS   = ["temperature", "pressure", "vibration", "current_draw"]
SENSOR_UNITS = {
    "temperature":  "°C",
    "pressure":     "bar",
    "vibration":    "mm/s",
    "current_draw": "A",
}
COLORS = {"temperature": "#E53935", "pressure": "#1565C0",
          "vibration": "#2E7D32", "current_draw": "#F57F17"}


def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} readings  |  Anomalies: {df['anomaly'].sum()} ({df['anomaly'].mean()*100:.1f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    for col in SENSORS:
        # Lag features
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)
        # Rolling stats (30-min window = 6 readings)
        df[f"{col}_roll6_mean"] = df[col].rolling(6,  min_periods=1).mean()
        df[f"{col}_roll6_std"]  = df[col].rolling(6,  min_periods=1).std().fillna(0)
        # Rolling stats (2-hr window = 24 readings)
        df[f"{col}_roll24_mean"] = df[col].rolling(24, min_periods=1).mean()
        # Z-score within rolling window (deviation from local norm)
        roll_mean = df[col].rolling(24, min_periods=1).mean()
        roll_std  = df[col].rolling(24, min_periods=1).std().fillna(1).replace(0, 1)
        df[f"{col}_zscore"] = ((df[col] - roll_mean) / roll_std).round(3)

    # Cross-sensor features
    df["temp_pressure_ratio"] = (df["temperature"] / df["pressure"].replace(0, 0.001)).round(3)
    df["power_index"]         = (df["current_draw"] * df["temperature"] / 1000).round(4)
    df["mech_stress"]         = (df["vibration"] * df["pressure"]).round(4)

    # Time features
    df["is_working_hours"] = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] <= 22)).astype(int)
    df["is_weekend"]       = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]         = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]         = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df = df.dropna().reset_index(drop=True)
    return df


# ── EDA Plots ─────────────────────────────────────────────────────────────────

def plot_sensor_overview(df):
    """4-panel time-series with anomalies highlighted."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), facecolor="#F8F8F8", sharex=True)
    fig.suptitle("IoT Sensor Readings — 30-Day Overview", fontsize=15, fontweight="bold")

    anom = df[df["anomaly"] == 1]

    for ax, col in zip(axes, SENSORS):
        color = COLORS[col]
        unit  = SENSOR_UNITS[col]
        ax.plot(df["timestamp"], df[col], linewidth=0.6, color=color, alpha=0.8, label=col)
        ax.scatter(anom["timestamp"], anom[col], color="red", s=18, zorder=5,
                   label=f"Anomaly ({len(anom)})", alpha=0.7)
        ax.set_ylabel(f"{col}\n({unit})", fontsize=9)
        ax.set_facecolor("white")
        ax.legend(fontsize=8, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig("outputs/1_sensor_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → outputs/1_sensor_overview.png")


def plot_distributions(df):
    """Sensor distributions: normal vs anomaly."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor="#F8F8F8")
    fig.suptitle("Sensor Value Distributions: Normal vs Anomaly", fontsize=14, fontweight="bold")

    normal = df[df["anomaly"] == 0]
    anom   = df[df["anomaly"] == 1]

    for col, ax_hist, ax_box in zip(SENSORS, axes[0], axes[1]):
        unit  = SENSOR_UNITS[col]
        color = COLORS[col]

        # Histogram
        ax_hist.hist(normal[col], bins=40, alpha=0.6, color=color,   label="Normal",  density=True)
        ax_hist.hist(anom[col],   bins=20, alpha=0.7, color="red",   label="Anomaly", density=True)
        ax_hist.set_title(f"{col} ({unit})", fontweight="bold")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=8)
        ax_hist.set_facecolor("white")

        # Box plot
        ax_box.boxplot([normal[col].values, anom[col].values],
                       labels=["Normal","Anomaly"],
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.5),
                       medianprops=dict(color="black", linewidth=2))
        ax_box.set_title(f"{col} spread", fontweight="bold")
        ax_box.set_ylabel(unit)
        ax_box.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("outputs/2_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → outputs/2_distributions.png")


def plot_correlations(df):
    """Sensor correlation heatmap."""
    corr = df[SENSORS].corr()
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#F8F8F8")
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(SENSORS, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(SENSORS, fontsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    ax.set_title("Sensor Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/3_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → outputs/3_correlations.png")


def plot_anomaly_types(df):
    """Anomaly type breakdown."""
    anom_counts = df[df["anomaly"]==1]["anomaly_type"].value_counts()
    colors_pie  = ["#E53935","#1565C0","#2E7D32","#F57F17","#7B1FA2"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#F8F8F8")

    # Pie
    ax = axes[0]
    ax.pie(anom_counts.values, labels=anom_counts.index, colors=colors_pie,
           autopct="%1.1f%%", startangle=140,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("Anomaly Type Distribution", fontweight="bold")

    # Hourly anomaly rate
    ax = axes[1]
    hourly = df.groupby("hour_of_day")["anomaly"].mean() * 100
    ax.bar(hourly.index, hourly.values, color="#E53935", alpha=0.8, edgecolor="white")
    ax.set_title("Anomaly Rate by Hour of Day (%)", fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_facecolor("white")
    ax.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig("outputs/4_anomaly_types.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → outputs/4_anomaly_types.png")


def main():
    df     = load(RAW_FILE)
    plot_sensor_overview(df)
    plot_distributions(df)
    plot_correlations(df)
    plot_anomaly_types(df)

    df_proc = engineer_features(df)
    df_proc.to_csv(OUT_FILE, index=False)
    print(f"\nProcessed data saved → {OUT_FILE}")
    print(f"Shape: {df_proc.shape}  ({df_proc.shape[1]-2} features)")


if __name__ == "__main__":
    main()
