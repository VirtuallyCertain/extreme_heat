# 🌡️ Extreme Heat Events in France: Predictive Modeling
**An Early Warning System for Dangerous Heat Conditions (1990–2025)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://extremeheatevents-france.streamlit.app)

## 🎯 Research Question
**When are people exposed to dangerous heat conditions in France?**

The objective of this project is to build a predictor that serves as an early heat alarm. We focus on **extreme daily temperatures**, specifically the daily maximum temperature (**TX**), to identify patterns and predict events early enough to act.

---

## 🚀 Interactive Web App
Explore our models and climate analysis in the live Streamlit application.

🔗 **[Live Demo: Extreme Heat Events App](https://extremeheatevents-france.streamlit.app)**

### Suggested Walkthrough
1.  [cite_start]**Introduction & Motivation:** Context on regionality (e.g., Paris's Urban Heat Island vs. Marseille's coastal effects)[cite: 7, 8].
2.  **Data & Baseline:** View the initial approach using Météo-France daily data and Gradient Boosting.
3.  **Final Model:** Explore the optimized **XGBoost** model trained on **Copernicus** data (1990–2025).
4.  [cite_start]**Conclusion:** Summary of key findings and our technical roadmap[cite: 5].

---

## 🛠️ Project Evolution & Methodology

### Study Regions
We analyzed four French départements representing diverse climate zones:
* **Paris (75)**, **Lyon (69)**, **Bordeaux (33)**, and **Marseille (13)**.

### Data Sources & Technical Stack
The project evolved through two main stages:

* **First Model Phase:** Used Météo-France daily climatological data (*Données climatologiques de base – quotidiennes*) via [data.gouv.fr](https://www.data.gouv.fr/).
* **Final Model Phase:** Climate data from the **[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)** (ERA5).
* [cite_start]**Predictive Modeling:** Transitioned from Gradient Boosting to an optimized **XGBoost** architecture, incorporating atmospheric features like wind stagnation and persistence signals[cite: 9, 10].

### 🛰️ Data Acquisition (Copernicus API)
The repository includes `copernicus_api_script.py`, which allows users to programmatically fetch updated climate data directly from the Copernicus Climate Data Store for further analysis or model retraining.

---

## 💡 Key Insights & Business Value
* [cite_start]**Regionality Matters:** Paris shows stronger urban amplification, while Marseille is shaped by coastal effects[cite: 8].
* [cite_start]**Feature Engineering:** Adding atmospheric stagnation and 48-hour rolling persistence features significantly made the model more robust[cite: 9, 15].
* [cite_start]**Value Proposition:** The system is **lightweight**, runs on standard hardware, and is regionally customizable, providing a foundation for early warning systems for health services and urban planning[cite: 24, 27].

---

## 🔮 Roadmap & Next Steps
[cite_start]To evolve this into a production-ready system[cite: 14]:
* [cite_start]**Sequence Modeling:** Benchmark **LSTM or Transformer** models to evaluate longer temporal dependencies[cite: 20].
* [cite_start]**Spatial Expansion:** Extend coverage to Spain, Italy, and Germany[cite: 17].
* [cite_start]**Climate-Adaptive Thresholds:** Use rolling percentiles to account for long-term climate trends[cite: 18].
* [cite_start]**Probabilistic Outputs:** Generate calibrated probabilities for better stakeholder risk assessment[cite: 19].

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
*Note: This project was developed as part of a Data Science certification (DataScientest).*
