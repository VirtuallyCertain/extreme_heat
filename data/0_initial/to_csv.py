import pandas as pd

df = pd.read_parquet("marseille_1990-2025_vars13_v1.parquet", engine="pyarrow")
df.to_csv("marseille_1990-2025_vars13_v1.csv", index=False)
df = pd.read_parquet("paris_1990-2025_vars13_v1.parquet", engine="pyarrow")
df.to_csv("paris_1990-2025_vars13_v1.csv", index=False)
df = pd.read_parquet("marseillemarignane_1990-2025_vars13_v1.parquet", engine="pyarrow")
df.to_csv("marseillemarignane_1990-2025_vars13_v1.csv", index=False)
df = pd.read_parquet("bordeaux_1990-2025_vars13_v1.parquet", engine="pyarrow")
df.to_csv("bordeaux_1990-2025_vars13_v1.csv", index=False)
#pd.read_csv("lyon_1990-2025_vars13_v1.csv").head()
