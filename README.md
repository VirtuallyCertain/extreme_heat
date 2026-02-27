# 🌡️ Extreme Heat Events in France

## Research question
🚨 **When are people exposed to dangerous heat conditions in France?**

The objective is to build a predictor that can serve as an early heat alarm.

The project focuses on **extreme daily temperatures**, with a second-stage
hourly exploration for selected historical events (e.g. the 2003 heatwave).
Methods are chosen with attention to minimizing computational impact.

---

## Project approach (DataScientest milestones)

### Stage 1 – Data mining & visualization (Dec 19)
- Start with temperature data only
- Select stations
- Explore daily temperature data
- Understand data structure, biases, and coverage
- Produce validated visualizations with commentary

### Stage 2 – Pre-processing & feature engineering (Jan 9)
- Clean data
- Select variables
- Engineer features for machine learning

### Stage 3 – Modeling (Jan–Feb)
- Baseline models (Random Forest)
- Optimization and interpretation

### Stage 4 – Application & defense (Feb)
- Streamlit app to explore extreme heat events
- Final report and GitHub repository

---

## Data sources

### Daily data (main dataset)
**Météo-France daily climatological data**, accessed via the official open-data
portal **data.gouv.fr**.

- Dataset: *Données climatologiques de base – quotidiennes*
- Period: **1950–2024**
- Key variable: **TX** (daily maximum temperature)

https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes

**Search terms on data.gouv.fr**
- `données climatologiques de base quotidiennes`
- `quot_departement_XX`

**Processed temperature data**
https://drive.google.com/file/d/1Gk5nXnru6G_UXIPqt9Y_SqYJfYldZPkg

---

### Hourly data (optional, targeted use)
Used only for in-depth analysis of selected extreme events.

- Dataset: *Données climatologiques de base – horaires*

https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires/

**Search terms**
- `données climatologiques de base horaires`
- `hor_departement_XX`

Preprocesssed temperature data: **https://drive.google.com/file/d/1Gk5nXnru6G_UXIPqt9Y_SqYJfYldZPkg/view?usp=drive_link **
---

## Study regions
Four French départements representing different regions:
- Paris (75)
- Lyon (69)
- Bordeaux (33)
- Marseille (13)

For each département, the corresponding **RR-T-Vent daily file** is downloaded.
Départements are identified by the first two digits of the postal code.

---

## Station selection
For each city, approximately **five stations** are selected using objective criteria:
- long TX record
- recent data availability
- data completeness

The same selection logic is applied to all cities using reproducible code.

---

## Current focus
👉 **Data visualization**
- explore distributions of TX
- identify extreme days and years
- compare cities
- detect trends and potential biases

This corresponds to **Stage 1: Data mining & Data visualization**.

---

## Scope limitations
This project does **not** include:
- long-term climate projections
- full physiological wet-bulb modeling
- operational weather forecasting


## Streamlit App (Live Demo)
Live demo: [Extreme Heat Events – Streamlit App](https://extremeheatevents-nyaxau3gelzchhcsyuwhna.streamlit.app/)

### How to use
Use the **sidebar** to navigate through the four sections:

| Section | Content |
|---|---|
| **1. Introduction & Motivation** | Context, research question, study cities |
| **2. Data Explanation & First Model** | EDA, feature engineering, baseline Gradient Boosting |
| **3. Improvements & Final Model** | Hyperparameter tuning, atmospheric features, evaluation |
| **4. Conclusion & Next Steps** | Key findings, roadmap, business value |

### Suggested walkthrough (3–5 minutes)
1. Start with **Introduction & Motivation** for the overall context and scope.
2. Continue to **Data Explanation & First Model** to understand the dataset and baseline modeling approach.
3. Move to **Improvements & Final Model** to see what feature engineering and tuning improved predictions.
4. Finish with **Conclusion & Next Steps** for the summary, roadmap, and practical impact.

### Notes
- The app is designed to be **efficient and lightweight** (laptop-friendly).
- Plots and metrics displayed in the app are based on the processed datasets and modeling pipeline described in the report.
