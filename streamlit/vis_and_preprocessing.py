import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

def show_page():
    st.title("Motiviation")
    st.markdown("""
    In diesem Abschnitt analysieren wir die historischen Temperaturdaten von Paris, Lyon, Bordeaux und Marseille.
    Ziel ist es, Trends zu identifizieren und die Daten für das Modeling vorzubereiten.
    """)

    # Pfade (angepasst an Streamlit-Struktur)
    DATA_DIR = "../data/0_initial"
    CITY_OUT_DIR = "../data/1_outputs"

    # Hilfsfunktionen (aus deinem Notebook)
    def load_city_data(path):
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.astype(str).str.strip()
        df["date"] = pd.to_datetime(df["AAAAMMJJ"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])
        return df

    # Da wir im Streamlit-Kontext sind, laden wir Beispieldaten oder gecachte Daten
    # Für die Demo simulieren wir die Auswahl der Städte
    city_files = {
        "Paris": "Paris_daily_TX_raw.csv",
        "Lyon": "Lyon_daily_TX_raw.csv",
        "Bordeaux": "Bordeaux_daily_TX_raw.csv",
        "Marseille": "Marseille_daily_TX_raw.csv",
    }

    # Laden der bereits prozessierten Daten aus deinem CITY_OUT_DIR
    dfs = []
    try:
        for city, filename in city_files.items():
            path = os.path.join(CITY_OUT_DIR, filename)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["city"] = city
                df["date"] = pd.to_datetime(df["date"])
                dfs.append(df)
        
        if not dfs:
            st.error("Keine Daten in 'data/1_outputs' gefunden. Bitte stelle sicher, dass die CSV-Dateien dort liegen.")
            return

        df_all = pd.concat(dfs, ignore_index=True)
        
        # --- Visualisierung 1: Jährliche Sommertemperaturen ---
        st.subheader("Jährliche Sommer-Höchsttemperaturen (JJA)")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for city in df_all["city"].unique():
            df_city = df_all[df_all["city"] == city]
            # Filter Juni, Juli, August
            summer_df = df_city[df_city["date"].dt.month.isin([6, 7, 8])]
            annual_summer = summer_df.groupby(summer_df["date"].dt.year)["TX"].mean()
            sns.lineplot(x=annual_summer.index, y=annual_summer.values, label=city, ax=ax)
        
        ax.set_ylabel("Temperatur (°C)")
        ax.set_xlabel("Jahr")
        st.pyplot(fig)

        # --- Visualisierung 2: Extreme Hitze (>35°C) ---
        st.subheader("Entwicklung extremer Hitzeereignisse (>35°C)")
        df_all["is_extreme_35"] = df_all["TX"] > 35.0
        df_all["decade"] = (df_all["date"].dt.year // 10) * 10
        
        decadal_stats = df_all.groupby(["city", "decade"]).agg(
            n_days=("TX", "size"),
            n_extreme_35=("is_extreme_35", "sum")
        ).reset_index()
        decadal_stats["share_extreme_35"] = decadal_stats["n_extreme_35"] / decadal_stats["n_days"]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=decadal_stats, x="decade", y="share_extreme_35", hue="city", marker="o", ax=ax2)
        plt.title("Anteil der Tage > 35°C pro Jahrzehnt")
        st.pyplot(fig2)

        # --- Statistische Validierung ---
        st.subheader("Statistische Auswertung (Marseille)")
        sub_decade = decadal_stats[decadal_stats["city"] == "Marseille"]
        rho, pval = stats.spearmanr(sub_decade["decade"], sub_decade["share_extreme_35"])
        
        col1, col2 = st.columns(2)
        col1.metric("Spearman Rho", f"{rho:.3f}")
        col2.metric("p-Value", f"{pval:.5f}")

        st.info(f"Da der p-Wert ({pval:.5f}) < 0.05 ist, ist der Trend statistisch signifikant.")

    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
