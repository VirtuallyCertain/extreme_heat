Data & Features

Data is sourced from the Open-Meteo Historical API (2010–2024). This dataset is designed to beat the baseline model by providing physical atmospheric context for heatwave intensification.

## Data Source
All data is retrieved via the **Open-Meteo Historical Weather API**. This source provides consistent, hourly meteorological records that are aggregated into daily metrics for model training.

## Feature Overview & Predictive Value

We focus on specific atmospheric markers that provide a deeper context than simple temperature readings:

### Dataset Features & ML Predictive Value

### 📊 Dataset Features & ML Predictive Value

| Variable | Meaning | ML Predictive Value | Unit |
| :--- | :--- | :--- | :--- |
| **date** | Timestamp | Captures seasonality (e.g., summer peaks vs. shoulder seasons). | YYYY-MM-DD |
| **temp_max_ref** | Max daily temperature | The target baseline for heat intensity levels. | °C |
| **gph500_mean** | Air pressure (MSL) | Identifies "Heat Domes" and stable high-pressure systems. | hPa |
| **humi** | Rel. Humidity | Distinguishes between humid/muggy vs. dry heatwaves. | % |
| **dew** | Dew point | Measures absolute moisture; critical for nighttime cooling potential. | °C |
| **temp_dew_spread** | Temp-Dew Point Diff. | **Key Indicator:** Measures air dryness and solar heating potential. | Δ°C |
| **city** | City identifier | Provides regional context (e.g., coastal Marseille vs. inland Paris). | Text |

## Why these features?
By using the **Dew Point Spread** and **Air Pressure** as features, the model can distinguish between a standard hot summer day and a dangerous, stagnant heatwave event. These variables act as "early warning" signals that traditional baseline models often miss.

## Data Integrity
Every data fetch includes an automated **Plausibility Check**. This ensures that all retrieved values (Pressure, Temperature, and Humidity) align with physical meteorological limits before being used in the training pipeline.
