import base64
import pandas as pd
import streamlit as st


# CITY_FILES = [
#     "../data/0_initial/Paris_Q_75_previous-1950-2024_RR-T-Vent.zip",
#     "../data/0_initial/Marseille_Q_13_previous-1950-2024_RR-T-Vent.zip",
#     "../data/0_initial/Lyon_Q_69_previous-1950-2024_RR-T-Vent.zip",
#     "../data/0_initial/Bordeaux_Q_33_previous-1950-2024_RR-T-Vent.zip",
# ]

# For streamlit we need to change the "start" of the directory to the repo root
BASE_DIR = "../"
CITY_FILES = [
    f"{BASE_DIR}data/0_initial/Paris_Q_75_previous-1950-2024_RR-T-Vent.zip",
    f"{BASE_DIR}data/0_initial/Marseille_Q_13_previous-1950-2024_RR-T-Vent.zip",
    f"{BASE_DIR}data/0_initial/Lyon_Q_69_previous-1950-2024_RR-T-Vent.zip",
    f"{BASE_DIR}data/0_initial/Bordeaux_Q_33_previous-1950-2024_RR-T-Vent.zip",
]

COL_STATION = "NOM_USUEL"
COL_LAT = "LAT"
COL_LON = "LON"
COL_DATETIME = "AAAAMMJJ"
COL_TEMPERATURE = "TX"

HEAT_TEMPERATURE_THRESHOLD = 35.0
MIN_CONSECUTIVE_DAYS = 3

APP_TITLE = "Introduction & Motivation"
SIDEBAR_TITLE = "Filters"

@st.cache_data
def load_data(files):
    """Load and prepare data from multiple CSV files."""
    usecols = [COL_STATION, COL_LAT, COL_LON, COL_DATETIME, COL_TEMPERATURE]
    
    dfs = [
        pd.read_csv(
            f, 
            sep=';',
            usecols=usecols,
            dtype={
                COL_STATION: 'category',
                COL_LAT: 'float32',
                COL_LON: 'float32',
            }
        ) for f in files
    ]
    
    data = pd.concat(dfs, ignore_index=True)
    data[COL_DATETIME] = pd.to_datetime(data[COL_DATETIME], format="%Y%m%d")
    data[COL_TEMPERATURE] = pd.to_numeric(data[COL_TEMPERATURE], errors="coerce")
    data = data.dropna(subset=[COL_LAT, COL_LON, COL_TEMPERATURE, COL_DATETIME])
    data["date"] = data[COL_DATETIME].dt.date
    
    return data


@st.cache_data
def detect_heatwaves(_df, temp_threshold, min_days):
    """Detect heatwaves: min_days consecutive days above temp_threshold."""
    df = _df.copy()
    df["is_hot"] = df[COL_TEMPERATURE] >= temp_threshold
    
    heat_events = []
    
    for station, group in df.groupby(COL_STATION, observed=True):
        group = group.sort_values("date").reset_index(drop=True)
        
        hot = group["is_hot"].values
        dates = group["date"].values
        lat = float(group[COL_LAT].iloc[0])
        lon = float(group[COL_LON].iloc[0])
        
        streak_start = None
        streak_len = 0
        
        for i, is_hot in enumerate(hot):
            if is_hot:
                if streak_start is None:
                    streak_start = dates[i]
                streak_len += 1
            else:
                if streak_len >= min_days:
                    heat_events.append({
                        "station": station,
                        "start_date": streak_start,
                        "end_date": dates[i - 1],
                        "days": streak_len,
                        "lat": lat,
                        "lon": lon
                    })
                streak_start = None
                streak_len = 0
        
        if streak_len >= min_days:
            heat_events.append({
                "station": station,
                "start_date": streak_start,
                "end_date": dates[-1],
                "days": streak_len,
                "lat": lat,
                "lon": lon
            })
    
    return pd.DataFrame(heat_events)


def show_temperature_figure():
    st.subheader("Temperature Trends by City for best Station")
    cities = ["Paris", "Marseille", "Lyon", "Bordeaux", "All Cities"]
    all_cities = "All_Cities"

    tabs = st.tabs(cities)

    # Paris
    with tabs[0]:
        st.write(f"### Analysis for {cities[0]}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(f"{BASE_DIR}figures/1_figs/{cities[0]}_annual_mean.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[0]}_daily.png", width='content')
        with col2:
            st.image(f"{BASE_DIR}figures/1_figs/{cities[0]}_annual_mean_trend.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[0]}_summer_mean.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[0]}_daily_30drolling.png", width='content')
    
    # Marseille
    with tabs[1]:
        st.write(f"### Analysis for {cities[1]}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(f"{BASE_DIR}figures/1_figs/{cities[1]}_annual_mean.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[1]}_daily.png", width='content')
        with col2:
            st.image(f"{BASE_DIR}figures/1_figs/{cities[1]}_annual_mean_trend.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[1]}_summer_mean.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[1]}_daily_30drolling.png", width='content')
    
    # Lyon
    with tabs[2]:
        st.write(f"### Analysis for {cities[2]}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(f"{BASE_DIR}figures/1_figs/{cities[2]}_annual_mean.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[2]}_daily.png", width='content')
        with col2:
            st.image(f"{BASE_DIR}figures/1_figs/{cities[2]}_annual_mean_trend.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[2]}_summer_mean.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[2]}_daily_30drolling.png", width='content')
    
    # Bordeaux
    with tabs[3]:
        st.write(f"### Analysis for {cities[3]}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(f"{BASE_DIR}figures/1_figs/{cities[3]}_annual_mean.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[3]}_daily.png", width='content')
        with col2:
            st.image(f"{BASE_DIR}figures/1_figs/{cities[3]}_annual_mean_trend.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[3]}_summer_mean.png", width='content')
            st.image(f"{BASE_DIR}figures/1_figs/{cities[3]}_daily_30drolling.png", width='content')
    
    # All Cities
    with tabs[4]:
        st.write(f"### Analysis for {cities[4]}")
        st.image(f"{BASE_DIR}figures/1_figs/{all_cities}_annual_summer_mean.png", width='content')
        st.image(f"{BASE_DIR}figures/1_figs/{all_cities}_number_extreme_days.png", width='content')
        st.image(f"{BASE_DIR}figures/1_figs/{all_cities}_annual_share_extreme_days.png", width='content')


def show_page():

    def get_image_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    img_base64 = get_image_base64(f"{BASE_DIR}figures/datagouv-logo.png")

    st.markdown(
        f'''
        <a href="https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes" target="_blank">
            <img src="data:image/jpeg;base64,{img_base64}" width="200">
        </a>
        ''',
        unsafe_allow_html=True
    )
    st.title(APP_TITLE)

    st.sidebar.title(SIDEBAR_TITLE)
    
    with st.spinner("Loading data..."):
        df = load_data(CITY_FILES)
    
    user_temp_threshold = st.sidebar.slider(
        "Heat temperature threshold (°C)",
        min_value=25.0,
        max_value=45.0,
        value=float(HEAT_TEMPERATURE_THRESHOLD),
        step=0.5,
    )
    
    with st.spinner("Detecting heatwaves..."):
        heatwaves_df = detect_heatwaves(df, user_temp_threshold, MIN_CONSECUTIVE_DAYS)
    
    if heatwaves_df.empty:
        st.warning("No heatwaves detected with current settings.")
        return
    
    min_date = heatwaves_df["start_date"].min()
    max_date = heatwaves_df["end_date"].max()
    
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    
    all_stations = sorted(heatwaves_df["station"].unique())
    selected_stations = st.sidebar.multiselect(
        "Stations",
        options=all_stations,
        default=all_stations,
    )
    
    show_raw_data = st.sidebar.checkbox("Show raw heatwave data", value=False)
    
    start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
    
    filtered_hw = heatwaves_df[
        (heatwaves_df["start_date"] >= start_date) &
        (heatwaves_df["end_date"] <= end_date) &
        (heatwaves_df["station"].isin(selected_stations))
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total heat events", len(filtered_hw))
    
    with col2:
        st.metric("Stations affected", filtered_hw["station"].nunique())
    
    with col3:
        if not filtered_hw.empty:
            st.metric("Avg. duration (days)", f"{filtered_hw['days'].mean():.1f}")
        else:
            st.metric("Avg. duration (days)", "–")
    
    st.subheader("Heat events per station (all data)")
    
    if not filtered_hw.empty:
        station_counts = (
            filtered_hw
            .groupby(["station", "lat", "lon"], observed=True)
            .size()
            .reset_index(name="count")
        )
        station_counts["size"] = station_counts["count"] * 100
        
        st.map(station_counts, latitude="lat", longitude="lon", size="size")
        
    else:
        st.info("No heat events found for selected filters.")
    
    st.subheader("Heat events over time")
    
    if not filtered_hw.empty:
        timeline = (
            filtered_hw
            .assign(month=pd.to_datetime(filtered_hw["start_date"]).dt.to_period("M"))
            .groupby("month")
            .size()
            .reset_index(name="count")
        )
        timeline["month"] = timeline["month"].dt.to_timestamp()
        
        st.bar_chart(timeline.set_index("month")["count"])
    else:
        st.info("No data to display.")

    show_temperature_figure()
    
    if show_raw_data:
        st.subheader("Filtered heatwave data")
        st.dataframe(filtered_hw, width='content')


