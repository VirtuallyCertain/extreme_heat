import xarray as xr
import pandas as pd

# Deine Orte
LOCATIONS = [
    {"name": "paris", "lat": 48.75, "lon": 2.25},
    {"name": "marseille_center", "lat": 43.25, "lon": 5.25},
    {"name": "marseille_marignane", "lat": 43.50, "lon": 5.25},
    {"name": "lyon", "lat": 45.75, "lon": 4.75},
    {"name": "bordeaux", "lat": 44.75, "lon": -0.50}
]

# 1. Die NetCDF Datei öffnen
file_name = "temp_2024_01_h1.nc"
ds = xr.open_dataset(file_name, engine='h5netcdf')

# 2. Dimensionsnamen finden (level oder isobaricInhPa)
lev_dim = [d for d in ds.dims if 'level' in d or 'isobaric' in d][0]

for loc in LOCATIONS:
    # Daten für den spezifischen Ort extrahieren
    ds_point = ds.sel(latitude=loc['lat'], longitude=loc['lon'], method='nearest')
    
    # In Tabelle (DataFrame) umwandeln
    df = ds_point.to_dataframe().drop(columns=['expver', 'number', 'latitude', 'longitude'], errors='ignore')
    
    # Umstrukturieren: Levels (500, 850) als eigene Spalten anlegen
    df_final = df.unstack(level=lev_dim)
    df_final.columns = [f"{col[0]}_{int(col[1])}hPa" for col in df_final.columns]
    
    # Als CSV speichern
    out_name = f"daten_{loc['name']}_2024_01.csv"
    df_final.reset_index().to_csv(out_name, index=False)
    print(f"Datei erstellt: {out_name}")

ds.close()
