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

def show_page():
    st.title(PAGE_TITLE)

    # st.title("GradientBoostingRegressor Training with Data Filtering")

    # Sidebar for Data Filtering
    st.sidebar.header("Data Filtering")

    # Filter method selection
    filter_method = st.sidebar.radio(
        "Filter Method",
        ["Threshold", "Quantile"]
    )

    if filter_method == "Threshold":
        threshold_value = st.sidebar.number_input(
            "Threshold Value",
            min_value=0.0,
            value=35.0,
            step=0.5
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

    # City selection
    st.sidebar.subheader("City Selection")
    all_cities = st.sidebar.checkbox("All Cities", value=False)

    if not all_cities:
        cities = st.sidebar.multiselect(
            "Select Cities",
            ["Marseille", "Bordeaux", "Paris", "Lyon"],
            default=["Paris"]
        )
    else:
        cities = ["Marseille", "Bordeaux", "Paris", "Lyon"]

    topics = ["Data Explanation", "Model & Train", "Predict"]

    tabs = st.tabs(topics)

    # Paris
    with tabs[0]:
        st.subheader("Data Explanation")
    with tabs[1]:
        # Main area - Display filter settings
        st.subheader("Data Filter Settings:")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Filter Method:** {filter_method}")
            if filter_method == "Threshold":
                st.write(f"**Threshold Value:** {threshold_value}")
            else:
                st.write(f"**Quantile:** {quantile_value}")

        with col2:
            st.write(f"**Cumulative Days:** {cumulative_days}")

        with col3:
            st.write(f"**Selected Cities:** {', '.join(cities)}")

        st.divider()

        # Model Parameters in main window with frame/box
        st.subheader("Model Parameters")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.slider("n_estimators", 50, 500, 100, 10)
                learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
                max_depth = st.slider("max_depth", 1, 10, 3, 1)
            
            with col2:
                min_samples_split = st.slider("min_samples_split", 2, 20, 2, 1)
                min_samples_leaf = st.slider("min_samples_leaf", 1, 10, 1, 1)
                subsample = st.slider("subsample", 0.5, 1.0, 1.0, 0.1)

        st.divider()

        # Training Button
        if st.button("Train and Save Model", type="primary", use_container_width=True):
            if not all_cities and len(cities) == 0:
                st.error("Please select at least one city!")
            else:
                with st.spinner("Filtering data and training model..."):
                    try:
                        # Implement your data filtering here
                        # df = pd.read_csv("your_dataset.csv")
                        # df_filtered = df[df['city'].isin(cities)]
                        
                        # Apply filter
                        # if filter_method == "Threshold":
                        #     df_filtered = df_filtered[df_filtered['value'] >= threshold_value]
                        # else:
                        #     threshold = df_filtered['value'].quantile(quantile_value)
                        #     df_filtered = df_filtered[df_filtered['value'] >= threshold]
                        
                        # Calculate cumulative days
                        # df_filtered['cumulative'] = df_filtered.groupby('city')['value'].rolling(
                        #     window=cumulative_days
                        # ).sum().reset_index(0, drop=True)
                        
                        # X_train, y_train = ... (your feature creation)
                        
                        # Create and train model
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            subsample=subsample,
                            random_state=42
                        )
                        
                        # model.fit(X_train, y_train)
                        
                        # Automatically save model
                        model_filename = f"gbr_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                        joblib.dump(model, model_filename)
                        
                        st.success(f"✅ Model successfully trained and automatically saved as '{model_filename}'!")
                        
                        # Optional: Display model metrics
                        # score = model.score(X_test, y_test)
                        # st.metric("R² Score", f"{score:.4f}")
                        
                        # Download button for the model
                        with open(model_filename, "rb") as f:
                            st.download_button(
                                label="📥 Download Model",
                                data=f,
                                file_name=model_filename,
                                mime="application/octet-stream"
                            )
                            
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")

    with tabs[2]:
        st.subheader("Predict")


#def load_city_data(city, temp_dir, wind_dir):
#    tx = pd.read_csv(
#        f"{temp_dir}{city}_daily_TX_raw.csv",
#        parse_dates=["date"]
#    )
#    wind = pd.read_csv(
#        f"{wind_dir}temp_wind_{city.lower()}.csv",
#        parse_dates=["date"]
#    )
#    # We are only able to much if we have 

#    # Best station in Marseille: 13001009 (AIX EN PROVENCE).
#    # Filter by the best station for file {city}_daily_TX_raw.csv
#    # temp_wind_{city}.csv data is already for the station 13001009 (AIX EN PROVENCE)

#    # dynamical part to find the best station on arbitrary city.
#    tx["date"] = pd.to_datetime(tx["date"])
#    wind["date"] = pd.to_datetime(wind["date"])

#    # 2) Group wind by city
#    results = []

#    for city, wind_group in wind.groupby("city"):
#        # 3) Merge this city's wind data with tx on date
#        merged = pd.merge(tx, wind_group, on="date", how="inner")

#        # 4) Count matching TX == temp_max_c
#        match_count = (merged["TX"] == merged["temp_max_c"]).sum()

#        # 5) Store result
#        results.append({
#            "city": city,
#            "NUM_POSTE": merged["NUM_POSTE"].iloc[0] if len(merged) > 0 else None,
#            "NOM_USUEL": merged["NOM_USUEL"].iloc[0] if len(merged) > 0 else None,
#            "matching_days": match_count,
#            "total_days_compared": len(merged)
#        })

#    # 6) Convert to dataframe and sort
#    results_df = pd.DataFrame(results).sort_values("matching_days", ascending=False)
#    print(results_df)
#    print(f"\nBest station: {results_df.iloc[0]['NOM_USUEL']}")

#    tx_best_station = tx[tx['NUM_POSTE'] == results_df.iloc[0]['NUM_POSTE']]

#    return (
#        tx_best_station.merge(wind, on="date", how="inner")
#          .sort_values("date")
#          .reset_index(drop=True)
#    )


## # 3. Load data (Marseille)

## In[3]:


## ======================================================
## 3. Load data
## ======================================================
#CITY = "Marseille" # Marseille, Lyon, Paris or Bordeaux
#TEMP_DIR = "../data/1_outputs/"
#WIND_DIR = "../data/2_outputs/"

#df_marseille = load_city_data(CITY, TEMP_DIR, WIND_DIR)

#df_marseille.head()


## # 4. Explore data

## In[4]:


## ======================================================
## 4. Explore data
## ======================================================


## --- Contiue with the Marseille data ---
#df = df_marseille.copy()

## Ensure datetime
#df["date"] = pd.to_datetime(df["date"])
#df["month"] = df["date"].dt.month

## Keep summer months (JJA)
#df_summer = df[df["month"].isin([6, 7, 8])]

## --- Choose variables ---
#temp_col = "TX"                 # or "temp_max_c" (pick one and be consistent)
#wind_col = "wind_max_inst_ms"

## --- Scatter plot ---
#plt.figure()
#plt.scatter(df_summer[wind_col], df_summer[temp_col])
#plt.xlabel("Wind max instantaneous (m/s)")
#plt.ylabel("Max temperature (°C)")
#plt.title("Marseille: Wind vs Summer Maximum Temperature")
#plt.show()

## Interpretation:
## Wind speed is not linearly correlated with summer maximum temperature, 
##but it may act as a limiting factor for extreme heat and may become 
##informative when modelling heatwave occurrence or when using non-linear 
##models such as Random Forests, decision trees, or gradient boosting.


## # 5. Find heatwaves

## In[5]:


## ======================================================
## 5. Find heatwaves
## ======================================================


#df = df_marseille.copy()
#df["date"] = pd.to_datetime(df["date"])
#df = df.sort_values("date").reset_index(drop=True)

#TX_P95 = df["TX"].quantile(0.95)
#TX_THRESHHOLD = 35
#df["hot_day"] = df["TX"] >= TX_THRESHHOLD

## Explicit consecutive-day counter
#df["hot_spell_len"] = 0
#count = 0

#df["hot_spell_len"] = df.groupby((df["hot_day"] != df["hot_day"].shift()).cumsum())["hot_day"].cumsum() * df["hot_day"]

##for i, is_hot in enumerate(df["hot_day"]):
##    if is_hot:
##        count += 1
##    else:
##        count = 0
##    df.loc[i, "hot_spell_len"] = count

## Heatwave tags: date are greater than or equal to (ge) 3 days.
#df["heatwave_ge_3days"] = df["hot_spell_len"] >= 3
#df["heatwave_gt_3days"] = df["hot_spell_len"] > 3

#print(df['hot_spell_len'].sum())
#print(TX_P95)


## # 6. Summer analysis: wind vs temperature with heatwaves

## In[6]:


## ======================================================
## 6. Summer analysis: wind vs temperature with heatwaves
## ======================================================



## Recreate summer subset AFTER heatwave tags exist
#df["month"] = df["date"].dt.month
#df_summer = df[df["month"].isin([6, 7, 8])]

#temp_col = "TX"
#wind_col = "wind_max_inst_ms"

## All summer days
#plt.figure()
#plt.scatter(
#    df_summer[wind_col],
#    df_summer[temp_col],
#    alpha=0.4,
#    label="All summer days"
#)

## Heatwave days (≥ 3 consecutive days)
#hw = df_summer[df_summer["heatwave_ge_3days"]]

#plt.scatter(
#    hw[wind_col],
#    hw[temp_col],
#    color="red",
#    label="Heatwave days (≥3 days)"
#)

#plt.xlabel("Wind max instantaneous (m/s)")
#plt.ylabel("Max temperature (°C)")
#plt.title("Marseille: Wind vs Summer Maximum Temperature")
#plt.legend()
#plt.show()

#display(df.info())


## # 7. ML classification: heatwave occurrence 

## In[7]:


##df.columns

##missing too many data in "wind_mean_2m_ms"
#features = [
#    "wind_mean_10m_ms",
#    "wind_max_hourly_ms",
#    "wind_max_inst_ms",
#    "wind_gust_3s_ms",
#    "wind_dir_max_deg",
#    "wind_dir_inst_deg"
#]

#target = "TX"
## Drop missing values
#df_ml = df_summer[features + [target]].dropna()

#X = df_ml[features]
#y = df_ml[target]

#print(df_summer[features + [target]].isna().sum())
#print(df_summer.shape)
#print(df_ml.shape)
#display(df_ml.info())
#display(df_ml.head())

#df_ml = df_summer[features + [target]].dropna()
#print(df_ml.shape)

## what is the distribution of Tx?
#plt.figure(figsize=(6, 4))
#plt.hist(df_ml["TX"], bins=30)
#plt.xlabel("Daily maximum temperature (TX)")
#plt.ylabel("Number of days")
#plt.title("Distribution of summer TX (available days only)")
#plt.show()



## In[8]:


## Better understand Tx
## about 20% of the data missing for 2009-2019
## Interpretation: should not impact heavewaves too much as they are only selected
## When there are 3 days of high temperature. 
## Make sure date is datetime
#display(df_summer.head())
#df_summer = df_summer.copy()
#df_summer["year"] = df_summer["date"].dt.year

## Count available TX per year
#tx_count_per_year = (
#    df_summer["TX"]
#    .notna()
#    .groupby(df_summer["year"])
#    .sum()
#)

#plt.figure(figsize=(7, 4))
#plt.plot(tx_count_per_year.index, tx_count_per_year.values, marker="o")
#plt.xlabel("Year")
#plt.ylabel("Marseille: Number of summer days with TX for Marseille")
#plt.title("Availability of TX over time (summer only)")
#plt.show()

## total possible summer days per year (June–Aug)
#total_days_per_year = df_summer.groupby("year").size()

## available TX days per year
#tx_available = df_summer["TX"].notna().groupby(df_summer["year"]).sum()

## % missing TX
#tx_missing_pct = 100 * (1 - tx_available / total_days_per_year)

#print(tx_missing_pct)

#threshold = 10  # percent
#affected_years = tx_missing_pct[tx_missing_pct > threshold]

#affected_years


## In[9]:


## Train / test split (NO stratify for regression)
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y,
#    test_size=0.3,
#    random_state=42
#)

## Model
## Regression not classifiction for continiiys TX values.
## Will use MAE, R² to eveluate. 
#model = GradientBoostingRegressor(random_state=42)
#model.fit(X_train, y_train)

## Predictions
#y_pred = model.predict(X_test)

## Evaluation
#results_df = pd.DataFrame([{
#    "model": "Gradient Boosting (TX regression)",
#    "MAE": mean_absolute_error(y_test, y_pred),
#    "R2": r2_score(y_test, y_pred)
#}])

#display(results_df)


## Feature importance
#importances = pd.Series(
#    model.feature_importances_,
#    index=features
#).sort_values(ascending=False)


##print("Feature importance")
#display(importances)

#print('\n\n')
#print(f"Gradient Boosting for {CITY}")

