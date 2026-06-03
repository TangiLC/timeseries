import requests
import pandas as pd
from datetime import date, timedelta

BASE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
DATASET = "eco2mix-metropoles-tr"
URL = f"{BASE}/{DATASET}/records"

EPCI = "242900314"

date_debut = date.today() - timedelta(days=6 * 365)
annee_debut = date_debut.year
annee_fin = date.today().year

all_rows = []

for year in range(annee_debut, annee_fin + 1):
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01T00:00:00+00:00"

        if month == 12:
            end = f"{year + 1}-01-01T00:00:00+00:00"
        else:
            end = f"{year}-{month + 1:02d}-01T00:00:00+00:00"

        params = {
            "where": (
                f"code_insee_epci = '{EPCI}' "
                f"AND date_heure >= date'{start}' "
                f"AND date_heure < date'{end}'"
            ),
            "limit": 100,
        }

        r = requests.get(URL, params=params, timeout=30)

        if r.status_code != 200:
            print("Erreur", r.status_code, start, end)
            print(r.text)
            continue

        rows = r.json().get("results", [])
        all_rows.extend(rows)

if not all_rows:
    raise RuntimeError("Aucune donnée récupérée")

df = pd.DataFrame(all_rows)

df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
df["date"] = df["date_heure"].dt.date
df["consommation"] = pd.to_numeric(df["consommation"], errors="coerce")

df_daily = (
    df.dropna(subset=["consommation"])
    .groupby("date", as_index=False)
    .agg(
        conso_moy_mw=("consommation", "mean"),
        conso_min_mw=("consommation", "min"),
        conso_max_mw=("consommation", "max"),
        nb_mesures=("consommation", "count"),
    )
)

df_daily["conso_mwh"] = df_daily["conso_moy_mw"] * 24

df_daily.to_csv("brest_metropole_daily.csv", index=False)

print(df_daily.head())
print(df_daily.tail())
print(df_daily.shape)
