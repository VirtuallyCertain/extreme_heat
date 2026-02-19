import base64
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

PAGE_TITLE = "Data Explanation & First Model"
BASE_DIR = ""

def show_page():

    def get_image_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    img_base64 = get_image_base64(f"{BASE_DIR}figures/datagouv-logo.png")

    st.markdown(
        f'''
        <a href="https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes-stations-complementaires?utm_source=chatgpt.com" target="_blank">
            <img src="data:image/jpeg;base64,{img_base64}" width="200">
        </a>
        ''',
        unsafe_allow_html=True
    )

    st.title(PAGE_TITLE)

    # st.title("GradientBoostingRegressor Training with Data Filtering")


    topics = ["Data Explanation", "Model Train & Predit"]

    tabs = st.tabs(topics)

    # ======================================================
    # Helper function 
    # ======================================================
    def load_city_data(city, temp_dir, wind_dir):
        tx = pd.read_csv(
            f"{temp_dir}{city}_daily_TX_raw.csv",
            parse_dates=["date"]
        )
        wind = pd.read_csv(
            f"{wind_dir}temp_wind_{city.lower()}.csv",
            parse_dates=["date"]
        )

        tx["date"] = pd.to_datetime(tx["date"])
        wind["date"] = pd.to_datetime(wind["date"])

        wind = wind.drop_duplicates(subset=["date", "NUM_POSTE"])

        df_final = pd.merge(
            tx,
            wind,
            on=["date", "NUM_POSTE"],
            how="inner",
            suffixes=("_tx", "_wind")
        )

        return df_final.sort_values(["NUM_POSTE", "date"]).reset_index(drop=True)

    # ======================================================
    # Global Sidebar Settings (Visible in all tabs)
    # ======================================================
    st.sidebar.header("Filter")
    CITY = st.sidebar.selectbox("Select City", ["Marseille", "Lyon", "Paris", "Bordeaux"])
    TEMP_DIR = f"{BASE_DIR}data/1_outputs/"
    WIND_DIR = f"{BASE_DIR}data/2_outputs/"

    # Sidebar for Data Filtering
    st.sidebar.title("Filters")

    # Filter method selection
    filter_method = st.sidebar.radio(
        "Filter Method",
        ["Threshold", "Quantile"]
    )

    TX_THRESHOLD= 35
    quantile_value = 0.95

    if filter_method == "Threshold":
        TX_THRESHOLD = st.sidebar.slider(
            "Heat temperature threshold (°C)",
            min_value=25.0,
            max_value=45.0,
            value=float(TX_THRESHOLD),
            step=0.5,
        )
    else:
        quantile_value = st.sidebar.slider(
            "Quantile",
            min_value=0.85,
            max_value=1.0,
            value=0.95,
            step=0.05
        )

    # Number of cumulative days
    cumulative_days = st.sidebar.number_input(
        "Number of Cumulative Days",
        min_value=3,
        value=3,
        step=1
    )

    df_ml = pd.DataFrame()
    X = pd.DataFrame()
    y = pd.DataFrame()

    if st.sidebar.button("Load Data", type="primary"):
        with st.spinner(f"Loading data for {CITY}..."):
            try:
                df_city = load_city_data(CITY, TEMP_DIR, WIND_DIR)
                st.session_state["df_city"] = df_city
                st.session_state["city_loaded"] = CITY
                st.sidebar.success(f"✅ Data for {CITY} loaded!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

    with tabs[0]:
        st.subheader("Data Preprocessing & Exploration")

        # Only continue if data is loaded
        if "df_city" not in st.session_state:
            st.info("👈 Please select a city and click **Load Data** in the sidebar to begin.")
            st.stop()

        df_city = st.session_state["df_city"]
        city_label = st.session_state["city_loaded"]

        # ======================================================
        # 3. Raw Data Preview
        # ======================================================
        st.subheader("Raw Data Preview")
        st.dataframe(df_city.head(10), width="content")

        st.divider()

        # ======================================================
        # 4. Heatwave Detection Settings
        # ======================================================
        st.subheader("Heatwave Detection")

        # with st.container(border=True):
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         filter_method = st.radio("Threshold Method", ["Fixed Threshold", "Quantile"])
        #     with col2:
        #         if filter_method == "Fixed Threshold":
        #             TX_THRESHOLD = st.number_input("Temperature Threshold (°C)", value=35.0, step=0.5)
        #         else:
        #             quantile_val = st.slider("Quantile", 0.0, 1.0, 0.95, 0.01)
        #             TX_THRESHOLD = df_city["TX"].quantile(quantile_val)
        #             st.info(f"Computed threshold: **{TX_THRESHOLD:.2f} °C** (p{quantile_val*100:.0f})")

        #     cumulative_days = st.number_input(
        #         "Minimum Consecutive Hot Days (Heatwave Definition)",
        #         min_value=3,
        #         value=3,
        #         step=1
        #     )

        # ======================================================
        # 5. Process Data
        # ======================================================
        df = df_city.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["month"] = df["date"].dt.month

        # Apply threshold
        df["hot_day"] = df["TX"] >= TX_THRESHOLD

        # Consecutive hot day counter
        df["hot_spell_len"] = (
            df.groupby((df["hot_day"] != df["hot_day"].shift()).cumsum())["hot_day"]
            .cumsum() * df["hot_day"]
        )

        # Heatwave tags
        df["heatwave_ge_Ndays"] = df["hot_spell_len"] >= cumulative_days
        df["heatwave_gt_Ndays"] = df["hot_spell_len"] > cumulative_days

        # Summer subset
        df_summer = df[df["month"].isin([6, 7, 8])].copy()

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Hot Days", int(df["hot_day"].sum()))
        col2.metric("Total Heatwave Days", int(df["heatwave_ge_Ndays"].sum()))
        if filter_method == "Threshold":
            col3.metric(f"Heat temperature threshold", f"{TX_THRESHOLD} °C")
        else:
            col3.metric(f"P95 Temperature", f"{df['TX'].quantile(quantile_value):.2f} °C")

        st.divider()

        # ======================================================
        # 6. Scatter Plot: Wind vs Temperature
        # ======================================================
        st.subheader("Summer Analysis: Wind vs Maximum Temperature")

        temp_col = "TX"
        wind_col = "wind_max_inst_ms"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Without heatwave overlay
        axes[0].scatter(df_summer[wind_col], df_summer[temp_col], alpha=0.5)
        axes[0].set_xlabel("Wind max instantaneous (m/s)")
        axes[0].set_ylabel("Max temperature (°C)")
        axes[0].set_title(f"{city_label}: Wind vs Summer Max Temperature")

        # Plot 2: With heatwave overlay
        axes[1].scatter(
            df_summer[wind_col], df_summer[temp_col],
            alpha=0.4, label="All summer days"
        )
        hw = df_summer[df_summer["heatwave_ge_Ndays"]]
        axes[1].scatter(
            hw[wind_col], hw[temp_col],
            color="red", label=f"Heatwave days (≥{cumulative_days} days)"
        )
        axes[1].set_xlabel("Wind max instantaneous (m/s)")
        axes[1].set_ylabel("Max temperature (°C)")
        axes[1].set_title(f"{city_label}: Wind vs Summer Max Temperature (Heatwaves)")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

        # ======================================================
        # 7. TX Distribution & Availability
        # ======================================================
        st.subheader("TX Distribution & Availability")

        features = [
            "wind_mean_10m_ms",
            "wind_max_hourly_ms",
            "wind_max_inst_ms",
            "wind_gust_3s_ms",
            "wind_dir_max_deg",
            "wind_dir_inst_deg"
        ]
        target = "TX"

        df_ml = df_summer[features + [target]].dropna()
        X = df_ml[features]
        y = df_ml[target]

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))

        # TX Distribution
        axes2[0].hist(df_ml["TX"], bins=30)
        axes2[0].set_xlabel("Daily maximum temperature (TX)")
        axes2[0].set_ylabel("Number of days")
        axes2[0].set_title(f"{city_label}: Distribution of Summer TX")

        # TX Availability per year
        df_summer_copy = df_summer.copy()
        df_summer_copy["year"] = df_summer_copy["date"].dt.year
        tx_count_per_year = df_summer_copy["TX"].notna().groupby(df_summer_copy["year"]).sum()

        axes2[1].plot(tx_count_per_year.index, tx_count_per_year.values, marker="o")
        axes2[1].set_xlabel("Year")
        axes2[1].set_ylabel("Number of summer days with TX")
        axes2[1].set_title(f"{city_label}: Availability of TX over Time (Summer)")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.divider()
        anzahl_stationen = df_summer_copy["NUM_POSTE"].nunique()

        # ======================================================
        # 8. ML Data Preview
        # ======================================================
        st.subheader("ML-Ready Data Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Rows:** {len(df_ml)}")
            st.write(f"**Features:** {len(features)}")
        with col2:
            st.write(f"**Missing values dropped:** {len(df_summer) - len(df_ml)}")
            st.write(f"**Target:** `{target}`")

        st.dataframe(df_ml.head(10), width="content")

        # Save processed data to session state for other tabs
        st.session_state["df_ml"] = df_ml
        st.session_state["df"] = df
        st.session_state["df_summer"] = df_summer
        st.session_state["features"] = features
        st.session_state["target"] = target
        st.session_state["TX_THRESHOLD"] = TX_THRESHOLD
        st.session_state["cumulative_days"] = cumulative_days

    with tabs[1]:
        # Main area - Display filter settings
        st.subheader("Data Filter Settings:")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Filter Method:** {filter_method}")
            if filter_method == "Threshold":
                st.write(f"**Threshold Value:** {TX_THRESHOLD} (°C)")
            else:
                st.write(f"**Quantile:** {quantile_value}")

        with col2:
            st.write(f"**Cumulative Days:** {cumulative_days}")

        with col3:
            st.write(f"Selected City: {CITY}")

        st.divider()

        # Model Parameters in main window with frame/box
        st.subheader("Model Parameters")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.number_input("n_estimators", 50, 500, 100, 10)
                max_depth = st.number_input("max_depth", 1, 10, 3, 1)
                test_size = st.number_input("test size", 0.1, 0.4, 0.3, 0.01)
            with col2:
                min_samples_split = st.slider("min_samples_split", 10, 100, 20, 1)
                min_samples_leaf = st.slider("min_samples_leaf", 10, 100, 20, 1)
                subsample = st.slider("subsample", 0.5, 1.0, 1.0, 0.1)
                learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)

        st.divider()

        # Training Button
        if st.button("Train and Save Model", type="primary", width="content"):
            if not CITY:
                st.error("Please select at least one city!")
            else:
                with st.spinner("Filtering data and training model..."):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            test_size=test_size,
                            random_state=42
                        )
                        
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            subsample=subsample,
                            random_state=42
                        )
                        
                        model.fit(X_train, y_train)
                        
                        model_filename = f"{BASE_DIR}models/gbr_model_{CITY}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                        joblib.dump(model, model_filename)
                        
                        st.success(f"✅ Model successfully trained and automatically saved as '{model_filename}'!")
                        
                        y_pred = model.predict(X_test)

                        results_df = pd.DataFrame([{
                            "model": "Gradient Boosting (TX regression)",
                            "MAE": mean_absolute_error(y_test, y_pred),
                            "R2": r2_score(y_test, y_pred)
                        }])

                        st.subheader("Model Results")
                        st.dataframe(results_df)

                        importances = pd.Series(
                            model.feature_importances_,
                            index=features
                        ).sort_values(ascending=False)

                        st.subheader("Feature Importances")
                        st.dataframe(importances.reset_index().rename(columns={"index": "Feature", 0: "Importance"}))
                            
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
