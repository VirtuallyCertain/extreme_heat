#!/usr/bin/env python
# coding: utf-8

# #### 0  Intro

# #### 🌡️ Extreme Temperatures
# 
# Paris → 75  
# Lyon → 69  
# Bordeaux → 33  
# Marseille → 13  
# 
# We deliberately use daily data rather than hourly data because daily extremes are the standard basis 
# for climate and public-health heat indicators, while hourly data would substantially increase 
# complexity without improving robustness or interpretability for this project. Daily temperature profiles 
# could be used once heatwaves are IDed.  
# 

# #### 1  Import libraries

# In[1]:


#math libraries
import pandas as pd

#plotting
import matplotlib.pyplot as plt
import seaborn as sns

#other
import os



# #### 2 Functions

# In[2]:


# define path for raw data
def load_city_data(path):
    """
    Load a Meteo-France CSV file and parse the date column.
    """
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.astype(str).str.strip()
    df["date"] = pd.to_datetime(df["AAAAMMJJ"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.drop('AAAAMMJJ', axis=1) # delete as AAAAMMJJ == date
    return df


# Select the top 5 meteo stations per city
def select_top_stations(df, min_tx_days=8000, min_end_year=2015, n_stations=5):
    """
    Select the best weather stations based on:
    - number of available TX (daily max temperature) values
    - recent data availability
    """
    summary = (
        df.groupby(["NUM_POSTE", "NOM_USUEL"])
          .agg(
              n_TX=("TX", "count"),
              start_date=("date", "min"),
              end_date=("date", "max")
          )
          .reset_index()
    )

    selected = (
        summary[
            (summary["n_TX"] >= min_tx_days) &
            (summary["end_date"].dt.year >= min_end_year)
        ]
        .sort_values("n_TX", ascending=False)
        .head(n_stations)
    )

    return selected


# Filter results by city/stations
def filter_city_by_stations(df, stations_df):
    """
    Keep only observations from the selected stations.
    """
    station_ids = stations_df["NUM_POSTE"].tolist()
    return df[df["NUM_POSTE"].isin(station_ids)].copy()


# In[3]:


# =========================
# Base path to data folder.
# =========================
DATA_DIR = "../data/0_initial"

# =========================
# File paths
# =========================
PATH_75 = f"{DATA_DIR}/Paris_Q_75_previous-1950-2024_RR-T-Vent.zip"
PATH_13 = f"{DATA_DIR}/Marseille_Q_13_previous-1950-2024_RR-T-Vent.zip"
PATH_33 = f"{DATA_DIR}/Bordeaux_Q_33_previous-1950-2024_RR-T-Vent.zip"
PATH_69 = f"{DATA_DIR}/Lyon_Q_69_previous-1950-2024_RR-T-Vent.zip"

# =========================
# Load data
# =========================
df_75 = load_city_data(PATH_75)
df_13 = load_city_data(PATH_13)
df_33 = load_city_data(PATH_33)
df_69 = load_city_data(PATH_69)


# =========================
# Preview data
# =========================
print('Paris')
display(df_75.head())
display(df_75.columns)
print('Marseille')
display(df_13.head())
display(df_13.columns)
print('Bordeaux')
display(df_33.head())
display(df_33.columns)
print('Lyon')
display(df_69.head())
display(df_69.columns)


# #### 4 Select the 5 stations to keep per city  (same logic for selection) 

# In[4]:


# 4. Select top 5 stations per city (single source of truth)
stations_75 = select_top_stations(df_75)  # Paris
stations_13 = select_top_stations(df_13)  # Marseille
stations_33 = select_top_stations(df_33)  # Bordeaux
stations_69 = select_top_stations(df_69)  # Lyon


# #### 5  Apply station selection 

# In[5]:


df_75_top = filter_city_by_stations(df_75, stations_75)  # Paris
df_13_top = filter_city_by_stations(df_13, stations_13)  # Marseille
df_33_top = filter_city_by_stations(df_33, stations_33)  # Bordeaux
df_69_top = filter_city_by_stations(df_69, stations_69)  # Lyon


# #### 6 Select variables and rename (work only from *_top dataframes)

# In[6]:


# Variables retained and renamed for clarity
rename_map = {
    'TM': 'temp_mean_c',
    'TX': 'temp_max_c',

    'FFM': 'wind_mean_10m_ms',
    'FF2M': 'wind_mean_2m_ms',
    'FXY': 'wind_max_hourly_ms',
    'FXI': 'wind_max_inst_ms',
    'FXI3S': 'wind_gust_3s_ms',

    'DXY': 'wind_dir_max_deg',
    'DXI': 'wind_dir_inst_deg'
}

cols_to_keep = ['date'] + list(rename_map.keys())

# Apply variable selection and renaming
df_75_climate = df_75_top[cols_to_keep].rename(columns=rename_map).copy()  # Paris
df_13_climate = df_13_top[cols_to_keep].rename(columns=rename_map).copy()  # Marseille
df_33_climate = df_33_top[cols_to_keep].rename(columns=rename_map).copy()  # Bordeaux
df_69_climate = df_69_top[cols_to_keep].rename(columns=rename_map).copy()  # Lyon


# 7 save the dates

# In[7]:


# Add city labels
df_75_climate["city"] = "Paris"
df_13_climate["city"] = "Marseille"
df_33_climate["city"] = "Bordeaux"
df_69_climate["city"] = "Lyon"

# Merge all cities into one dataframe
df_climate_all = pd.concat(
    [df_75_climate, df_13_climate, df_33_climate, df_69_climate],
    ignore_index=True
)

# Basic sanity check
df_climate_all = df_climate_all.sort_values(["city", "date"]).reset_index(drop=True)

# Save final preprocessed dataset with wind

# Create output folder for per-city files
CITY_OUT_DIR = "../data/2_outputs"
os.makedirs(CITY_OUT_DIR, exist_ok=True)

# Save one file per city
df_75_climate.to_csv(f"{CITY_OUT_DIR}/temp_wind_paris.csv", index=False)
df_13_climate.to_csv(f"{CITY_OUT_DIR}/temp_wind_marseille.csv", index=False)
df_33_climate.to_csv(f"{CITY_OUT_DIR}/temp_wind_bordeaux.csv", index=False)
df_69_climate.to_csv(f"{CITY_OUT_DIR}/temp_wind_lyon.csv", index=False)


# #### 8.1 Analyzsis of wind data

# In[8]:


# ===============================
# Wind variable availability per city
# ===============================

wind_vars = [
    "wind_mean_10m_ms",
    "wind_mean_2m_ms",
    "wind_max_hourly_ms",
    "wind_max_inst_ms",
    "wind_gust_3s_ms",
    "wind_dir_max_deg",
    "wind_dir_inst_deg"
]

city_dfs = {
    "Paris": df_75_climate,
    "Marseille": df_13_climate,
    "Bordeaux": df_33_climate,
    "Lyon": df_69_climate
}

rows = []

for city, df in city_dfs.items():
    for var in wind_vars:
        rows.append({
            "city": city,
            "variable": var,
            "n_non_null": df[var].notna().sum(),
            "coverage_%": round(df[var].notna().mean() * 100, 1)
        })

wind_availability = pd.DataFrame(rows)

# ML-ready variables (example threshold)
ml_ready_wind = wind_availability[wind_availability["coverage_%"] >= 80]

# Optional: wide view for quick inspection
wind_availability_wide = wind_availability.pivot(
    index="variable",
    columns="city",
    values="coverage_%"
)

wind_availability, ml_ready_wind, wind_availability_wide


# #### 8.2 Intrepretation of wind data: which  feature has the best coverage?
# 
# -------------------
# wind_max_inst_ms
# --------------------
# 
# From above, Marseille has the best coverage

# In[9]:


# ===============================
# Select ONE wind variable for all cities
# ===============================

# minimum coverage across cities per variable
score_table = (
    wind_availability
    .groupby("variable")["coverage_%"]
    .min()
    .reset_index()
    .rename(columns={"coverage_%": "min_coverage_%"})
    .sort_values("min_coverage_%", ascending=False)
)

score_table


# ##### Additional wind temp data analysis + pre-processing steps 

# In[10]:


# Read csv files from 2_outputs

files = {
    "Paris": "temp_wind_paris.csv",
    "Lyon": "temp_wind_marseille.csv",
    "Bordeaux": "temp_wind_bordeaux.csv",
    "Marseille": "temp_wind_lyon.csv",
}

dfs = []
for city, path in files.items():
    df = pd.read_csv(f"{CITY_OUT_DIR}/" + path)
    df["city"] = city
    dfs.append(df)

# Concatenating files together as column city is unique
data = pd.concat(dfs, ignore_index=True)

display(data.head())
display(data.info())


# In[11]:


def show_avail_pct_by_city(df):
    """ Show the available data by city in percentage."""
    avail_pct_by_city = (
    df
    .groupby("city")
    .apply(lambda g: (g.notna().mean()).mul(100), include_groups=False)
    )

    return avail_pct_by_city.round(2).style.highlight_max(axis=0, color='lightcoral')

show_avail_pct_by_city(data)

# Wind data for Marseille are most complete


# In[12]:


data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')


# In[13]:


# Setting date as index in data.

wind = data.set_index("city", append=True).swaplevel(0,1)
wind = wind.sort_index()

display(wind.index)
display(wind.head())


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

def show_lp_avail_data_by_city(data, col="wind_max_inst_ms"):
    """Show line plot of available wind data by city for a variable""" 
    availability = (
        wind[col]
        .notna()                                # True = vorhanden
        .groupby(level="city")
        .resample("YS", level="date")           # jährlich
        .mean()
        .mul(100)
        .rename("availability_pct")
        .reset_index()
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=availability, x="date", y="availability_pct", hue="city")
    plt.ylabel("% available")
    plt.xlabel("Year")
    plt.title(f"% Available Values per Year: {col}")
    plt.tight_layout()
    plt.show()

show_lp_avail_data_by_city(wind, col="wind_max_inst_ms")
show_lp_avail_data_by_city(wind, col="wind_mean_10m_ms")

