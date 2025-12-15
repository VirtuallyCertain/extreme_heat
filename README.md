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
- Period: **1950–2023**
- Key variable: **TX** (daily maximum temperature)

https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes

**Search terms on data.gouv.fr**
- `données climatologiques de base quotidiennes`
- `quot_departement_XX`

---

### Hourly data (optional, targeted use)
Used only for in-depth analysis of selected extreme events.

- Dataset: *Données climatologiques de base – horaires*

https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires/

**Search terms**
- `données climatologiques de base horaires`
- `hor_departement_XX`

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
