# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st


# ============================================================
# 1a. STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(layout="wide")
st.title("Extreme Heat & Heatwave Analysis")

# ======================================================
# 2. Functions
# ======================================================


def load_city_data(city, temp_dir, wind_dir):

    tx = pd.read_csv(temp_dir / f"{city}_daily_TX_raw.csv", parse_dates=["date"])

    wind = pd.read_csv(wind_dir / f"temp_wind_{city.lower()}.csv", parse_dates=["date"])

    tx.columns = tx.columns.str.strip().str.lower()
    wind.columns = wind.columns.str.strip().str.lower()

    # Normalize dates (remove time component but keep datetime type)
    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()
    wind["date"] = pd.to_datetime(wind["date"]).dt.normalize()

    tx = tx.rename(columns={"tx": "tmax"})
    wind = wind.rename(columns={"wind_max_inst_ms": "wind"})

    df = (
        tx.merge(wind, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 3. LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TEMP_DIR = BASE_DIR / "data" / "temp_data"
WIND_DIR = BASE_DIR / "data" / "temp_wind_data"

df = load_city_data("Marseille", TEMP_DIR, WIND_DIR)


# ============================================================
# 4. DATA PREPARATION
# ============================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

summer_df = df[df["month"].isin([5, 6, 7, 8, 9])]


# ============================================================
# 5. Marseille: Wind vs Summer Maximum Temperature
# ============================================================

#st.header("Marseille: Wind vs Summer Maximum Temperature")

#fig1, ax1 = plt.subplots()
#ax1.scatter(summer_df["wind"], summer_df["tmax"], alpha=0.4)
#ax1.set_xlabel("Wind max instantaneous (m/s)")
#ax1.set_ylabel("Max temperature (°C)")
#ax1.set_title("Marseille: Wind vs Summer Maximum Temperature")

#st.pyplot(fig1)


# ============================================================
# 6. Distribution of summer TX (available days only)
# ============================================================

st.header("Distribution of summer TX (available days only)")

fig2, ax2 = plt.subplots()
ax2.hist(summer_df["tmax"], bins=40)
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of summer TX (available days only)")

st.pyplot(fig2)


# ============================================================
# 7. Availability of TX over time (summer only)
# ============================================================

st.header("Availability of TX over time (summer only)")

fig3, ax3 = plt.subplots()

availability = summer_df.groupby("year")["tmax"].count()

ax3.plot(availability.index, availability.values)
ax3.set_xlabel("Year")
ax3.set_ylabel("Number of available TX observations")
ax3.set_title("Availability of TX over time (summer only)")

st.pyplot(fig3)

# ============================================================
# 8. MACHINE LEARNING MODEL (FULL FEATURE SET)
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.header("Machine Learning Model")

feature_cols = [
    "wind_mean_10m_ms",
    "wind",
    "wind_gust_3s_ms",
    "wind_max_hourly_ms",
    "wind_dir_max_deg",
    "wind_dir_inst_deg",
]

feature_cols = [col for col in feature_cols if col in summer_df.columns]

X = summer_df[feature_cols]
y = summer_df["tmax"]

ml_df = pd.concat([X, y], axis=1).dropna()

X = ml_df[feature_cols]
y = ml_df["tmax"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics_df = pd.DataFrame(
    {"model": ["Gradient Boosting (TX regression)"], "MAE": [mae], "R2": [r2]}
)

st.subheader("Model Performance")
st.dataframe(metrics_df)

# ============================================================
# 9. Feature Importance
# ============================================================

importance_df = pd.DataFrame(
    model.feature_importances_, index=feature_cols, columns=["Importance"]
).sort_values(by="Importance", ascending=False)

st.subheader("Feature Importance")
st.dataframe(importance_df)

# ============================================================
# 10. Wind vs Summer Maximum Temperature (Heatwave Overlay)
# ============================================================

st.header("Marseille: Wind vs Summer Maximum Temperature")

# Define heatwave threshold (example: >= 35°C)
heatwave_threshold = 35

heatwave_df = summer_df[summer_df["tmax"] >= heatwave_threshold]

fig_hw, ax_hw = plt.subplots()

# All summer days
ax_hw.scatter(summer_df["wind"], summer_df["tmax"], alpha=0.3, label="All summer days")

# Heatwave days
ax_hw.scatter(
    heatwave_df["wind"],
    heatwave_df["tmax"],
    color="red",
    alpha=0.7,
    label="Heatwave days (≥35°C)",
)

ax_hw.set_xlabel("Wind max instantaneous (m/s)")
ax_hw.set_ylabel("Max temperature (°C)")
ax_hw.set_title("Marseille: Wind vs Summer Maximum Temperature")
ax_hw.legend()

st.pyplot(fig_hw)
