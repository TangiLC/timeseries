import requests
import pandas as pd

BASE = "https://opendata.enedis.fr/data-fair/api/v1/datasets"
DATASET = "consommation-electrique-par-secteur-dactivite-epci"

# Brest Métropole
params = {
    "size": 1000,
    "q": "242900314",
}

r = requests.get(f"{BASE}/{DATASET}/lines", params=params, timeout=10)

# Ensure we got a successful response
r.raise_for_status()

data = r.json()
df = pd.DataFrame(data.get("results", []))
df = df[df["code_epci"] == "242900314"]
df = df[df["annee"].astype(int) >= 2020]

print(df.groupby(["annee", "code_grand_secteur"])["conso_totale_mwh"].sum())
