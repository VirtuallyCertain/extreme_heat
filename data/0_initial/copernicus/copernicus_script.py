# ==============================================================================
# SETUP GUIDE: HOW TO CONFIGURE YOUR COPERNICUS API CREDENTIALS
# ==============================================================================
#
# 1. REGISTRATION:
#    - Website: https://cds.climate.copernicus.eu/
#    - Create an account and verify your email.
#
# 2. RETRIEVE YOUR API KEY:
#    - Login -> Click your name (top right) -> "Profile".
#    - Copy the "API Key" block (URL and Key).
#
# 3. CREATE THE CONFIGURATION FILE (.cdsapirc):
#    - Location: Place it in your User folder (e.g., C:\Users\Name\ or /home/name/)
#    - Content: Paste the URL and Key exactly as shown in your profile.
#
# 4. CRITICAL: ACCEPT THE LICENSE ONLINE
#    - The code below uses the dataset: 'reanalysis-era5-pressure-levels'
#    - You MUST accept the license here:
#      https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
#    - Scroll to the bottom and click "Accept Terms".
#
# 5. INSTALLATION:
#    - Command: pip install cdsapi xarray pandas h5netcdf
#
# ==============================================================================

import cdsapi
import xarray as xr
import pandas as pd
import os
import time

# Initialize the CDS Client
c = cdsapi.Client()

# Configuration for target locations
LOCATIONS = [
    {"name": "paris", "lat": 48.75, "lon": 2.25},
    {"name": "marseille_center", "lat": 43.25, "lon": 5.25},
    {"name": "marseille_marignane", "lat": 43.50, "lon": 5.25},
    {"name": "lyon", "lat": 45.75, "lon": 4.75},
    {"name": "bordeaux", "lat": 44.75, "lon": -0.50}
]

OUTPUT_DIR = "pressure_data_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_period(year, month, days, label):
    """Downloads a specific time period from the CDS server."""
    y_str, m_str = str(year), f"{month:02d}"
    temp_nc = f"temp_{y_str}_{m_str}_{label}.nc"
    
    # Calculate bounding box for the request based on defined locations
    lats = sorted(list(set([l['lat'] for l in LOCATIONS])), reverse=True)
    lons = sorted(list(set([l['lon'] for l in LOCATIONS])))
#here is the list of variables: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
    c.retrieve('reanalysis-era5-pressure-levels', {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'pressure_level': ['500', '850'],
        'variable': ['temperature', 'u_component_of_wind', 'v_component_of_wind'],
        'year': y_str,
        'month': m_str,
        'day': days,
        'time': [f"{h:02d}:00" for h in range(24)],
        'area': [max(lats), min(lons), min(lats), max(lons)], 
    }, temp_nc)
    
    ds = xr.open_dataset(temp_nc, engine='h5netcdf')
    return ds, temp_nc

# --- START DOWNLOAD MARATHON ---
print(" Starting download marathon 1990-2025...")

for year in range(2024, 2025):
    for month in range(1, 13):
        y_str, m_str = str(year), f"{month:02d}"
        
        # Check if files already exist to avoid redundant downloads
        if all(os.path.exists(f"{OUTPUT_DIR}/pre_{l['name']}_{y_str}_{m_str}.csv") for l in LOCATIONS):
            continue

        print(f"\n --- Processing Month {m_str}/{y_str} ---")
        
        try:
            # Step 1: Download First Half (01-15)
            ds1, nc1 = fetch_period(year, month, [f"{d:02d}" for d in range(1, 16)], "h1")
            # Step 2: Download Second Half (16-End)
            ds2, nc2 = fetch_period(year, month, [f"{d:02d}" for d in range(16, 32)], "h2")
            
            # Merge both parts
            ds_month = xr.concat([ds1, ds2], dim='valid_time')
            
            # Detect dimension name (handles variations like 'level' or 'isobaricInhPa')
            lev_dim = [d for d in ds_month.dims if 'level' in d or 'isobaric' in d][0]

            for loc in LOCATIONS:
                # Extract specific point data
                ds_point = ds_month.sel(latitude=loc['lat'], longitude=loc['lon'], method='nearest')
                df = ds_point.to_dataframe().drop(columns=['expver', 'number', 'latitude', 'longitude'], errors='ignore')
                
                # Reshape: Convert levels into columns (e.g., temperature_500hPa)
                df_final = df.unstack(level=lev_dim)
                df_final.columns = [f"{col[0]}_{int(col[1])}hPa" for col in df_final.columns]
                
                # Save to CSV
                out_name = f"{OUTPUT_DIR}/pre_{loc['name']}_{y_str}_{m_str}.csv"
                df_final.reset_index().to_csv(out_name, index=False)

            # Cleanup: Close datasets and delete temporary .nc files
            ds1.close(); ds2.close()
            if os.path.exists(nc1): os.remove(nc1)
            if os.path.exists(nc2): os.remove(nc2)
            
            print(f" Month {m_str}/{y_str} successfully processed!")
            time.sleep(2) # Brief cooldown for the API server

        except Exception as e:
            print(f" Error in month {m_str}/{y_str}: {e}")
            print("Skipping to next month...")

print("\n CONGRATULATIONS! Your dataset for the heat model is complete.")
