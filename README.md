# 🌡️ Extreme Heat Events in France: Predictive Modeling
**An Early Warning System for Dangerous Heat Conditions (1990–2025)**

![Python Version](https://img.shields.io/badge/python-3.10.19-blue.svg) [![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://extremeheatevents-france.streamlit.app)

## 🎯 Research Question
**When are people exposed to dangerous heat conditions in France?**

The objective of this project is to build a predictor that serves as an early heat alarm. We focus on **extreme daily temperatures**, specifically the daily maximum temperature (**TX**), to identify patterns and predict events early enough to act. 

> **Scientific Definition:** In this study, a heatwave is defined as a period where the daily maximum temperature exceeds the 95th percentile (0.95 quantile) of the historical local climate for at least three consecutive days.

---

## 🚀 Interactive Web App
Explore our models and climate analysis in the live Streamlit application.

🔗 **[Live Demo: Extreme Heat Events App](https://extremeheatevents-france.streamlit.app)**

### Suggested Walkthrough
1. **Introduction & Motivation:** Context on regionality (e.g., Paris's Urban Heat Island vs. Marseille's coastal effects).
2. **Data Explanation & First Model:** View the initial approach using Météo-France daily data and Gradient Boosting.
3. **Improvements & Final Model:** Explore the optimized **XGBoost** model trained on **Copernicus** data (1990–2025).
4. **Conclusion & Next Steps:** Summary of key findings and our technical roadmap.

---

## 🛠️ Project Evolution & Methodology

### Study Regions
We analyzed four French départements representing diverse climate zones:
* **Paris (75)**, **Lyon (69)**, **Bordeaux (33)**, and **Marseille (13)**.

### Data Sources & Technical Stack
The project evolved through two main stages:

* **First Model Phase:** Used Météo-France daily climatological data (*Données climatologiques de base – quotidiennes*) via [data.gouv.fr](https://www.data.gouv.fr/).
* **Final Model Phase:** Climate data from the **[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)** (ERA5).
* **Predictive Modeling:** Transitioned from Gradient Boosting to an optimized **XGBoost** architecture, incorporating atmospheric features like wind stagnation and persistence signals.

### 🛰️ Data Acquisition (Copernicus API)
The repository includes `copernicus_api_script.py`, which allows users to programmatically fetch updated climate data directly from the Copernicus Climate Data Store for further analysis or model retraining.

---

## 💡 Key Insights & Business Value
* **Regionality Matters:** Paris shows stronger urban amplification, while Marseille is shaped by coastal effects.
* **Feature Engineering:** Adding atmospheric stagnation and 48-hour rolling persistence features significantly made the model more robust.
* **Value Proposition:** The system is **lightweight**, runs on standard hardware, and is regionally customizable, providing a foundation for early warning systems relevant to health services and urban planning.

---

## 🔮 Roadmap & Next Steps
To evolve this into a production-ready system:
* **Sequence Modeling:** Benchmark **LSTM or Transformer** models to evaluate longer temporal dependencies.
* **Spatial Expansion:** Extend coverage to Spain, Italy, and Germany.
* **Climate-Adaptive Thresholds:** Use rolling percentiles to account for long-term climate trends.
* **Probabilistic Outputs:** Generate calibrated probabilities for better stakeholder risk assessment.

---

## 💻 Installation & Usage

### Local Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App:**
    ```bash
    streamlit run streamlit/streamlit_main.py
    ```

---
*Note: This project was developed as part of a Data Science certification ([Liora](https://liora.io/)).*
