from pathlib import Path

import joblib
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="CSV Daily Viewer",
    layout="wide",
)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

CSV_FILES = {
    "Météo quotidienne": DATA_DIR / "brest_meteo_daily.csv",
    "Consommation quotidienne": DATA_DIR / "brest_metropole_daily.csv",
    "Population quotidienne": DATA_DIR / "brest_population_daily_est.csv",
    "Calendrier jours off": DATA_DIR / "brest_calendar_off_days.csv",
}

EXCLUDED_MODEL_FILES = {
    "sarimax_metadata.joblib",
}

DEFAULT_EXOG_BY_MODEL_NAME = {
    "sarima_temp_moy_c_conso_mwh": ["temp_moy_c"],
    "sarimax_model": ["temp_moy_c", "sin_year", "cos_year", "lag_365"],
}


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

    return df


@st.cache_resource
def load_pickle(path: Path):
    return joblib.load(path)


@st.cache_data
def load_base_data() -> pd.DataFrame:
    conso_path = DATA_DIR / "brest_metropole_daily.csv"
    meteo_path = DATA_DIR / "brest_meteo_daily.csv"

    if not conso_path.exists():
        return pd.DataFrame()

    conso = load_csv(conso_path)
    if "date" not in conso.columns:
        return pd.DataFrame()

    cols = ["date"]
    if "conso_mwh" in conso.columns:
        cols.append("conso_mwh")

    base = conso[cols].copy()

    if meteo_path.exists():
        meteo = load_csv(meteo_path)
        if {"date", "temp_moy_c"}.issubset(meteo.columns):
            base = base.merge(
                meteo[["date", "temp_moy_c"]],
                on="date",
                how="left",
            )

    return base.sort_values("date")


def list_model_files() -> dict[str, Path]:
    files = []
    for pattern in ("*.pkl", "*.joblib"):
        files.extend(MODELS_DIR.glob(pattern))

    model_files = {}
    for path in sorted(files):
        if path.name in EXCLUDED_MODEL_FILES:
            continue
        model_files[path.stem] = path

    return model_files


def load_sarimax_metadata() -> dict:
    metadata_path = MODELS_DIR / "sarimax_metadata.joblib"
    if metadata_path.exists():
        try:
            metadata = load_pickle(metadata_path)
            if isinstance(metadata, dict):
                return metadata
        except Exception:
            return {}
    return {}


def unwrap_model(obj):
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        metadata = {k: v for k, v in obj.items() if k != "model"}
        return model, metadata

    return obj, {}


def model_uses_exog(model) -> bool:
    return bool(getattr(model, "fit_with_exog_", False))


def infer_exog_cols(
    model_name: str, model_metadata: dict, global_metadata: dict, model
) -> list[str]:
    if "exog_cols" in model_metadata:
        return list(model_metadata["exog_cols"])

    if model_name == "sarimax_model" and "exog_cols" in global_metadata:
        return list(global_metadata["exog_cols"])

    if model_name in DEFAULT_EXOG_BY_MODEL_NAME:
        return DEFAULT_EXOG_BY_MODEL_NAME[model_name]

    if model_uses_exog(model):
        return ["temp_moy_c"]

    return []


def build_future_exog(
    exog_cols: list[str],
    start_date: pd.Timestamp,
    horizon: int,
    future_temps: list[float],
    base_data: pd.DataFrame,
) -> pd.DataFrame:
    future_dates = pd.date_range(start=start_date, periods=horizon, freq="D")
    X_future = pd.DataFrame(index=future_dates)

    if "temp_moy_c" in exog_cols:
        X_future["temp_moy_c"] = future_temps[:horizon]

    if "dayofyear" in exog_cols or "sin_year" in exog_cols or "cos_year" in exog_cols:
        dayofyear = X_future.index.dayofyear
        if "dayofyear" in exog_cols:
            X_future["dayofyear"] = dayofyear
        if "sin_year" in exog_cols:
            X_future["sin_year"] = np.sin(2 * np.pi * dayofyear / 365.25)
        if "cos_year" in exog_cols:
            X_future["cos_year"] = np.cos(2 * np.pi * dayofyear / 365.25)

    if "lag_365" in exog_cols:
        if base_data.empty or not {"date", "conso_mwh"}.issubset(base_data.columns):
            raise ValueError("Impossible de calculer lag_365 : conso_mwh indisponible.")

        history = (
            base_data[["date", "conso_mwh"]]
            .dropna(subset=["date", "conso_mwh"])
            .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
            .set_index("date")["conso_mwh"]
        )

        lag_values = []
        missing_dates = []
        for date in future_dates:
            lag_date = (date - pd.Timedelta(days=365)).normalize()
            if lag_date in history.index:
                lag_values.append(float(history.loc[lag_date]))
            else:
                lag_values.append(np.nan)
                missing_dates.append(lag_date.date().isoformat())

        if missing_dates:
            raise ValueError(
                "Impossible de calculer lag_365 pour les dates historiques : "
                + ", ".join(missing_dates)
            )

        X_future["lag_365"] = lag_values

    missing_cols = [col for col in exog_cols if col not in X_future.columns]
    if missing_cols:
        raise ValueError(
            "Variables exogènes non gérées par l'application : "
            + ", ".join(missing_cols)
        )

    return X_future[exog_cols]


def predict_model(
    model, exog_cols: list[str], X_future: pd.DataFrame | None, horizon: int
) -> list:
    if exog_cols:
        preds = model.predict(n_periods=horizon, X=X_future)
    else:
        preds = model.predict(n_periods=horizon)

    preds = list(preds)
    if len(preds) < horizon:
        preds += [None] * (horizon - len(preds))

    return preds[:horizon]


def predict_sqlite_approximation(forecast_start_date, future_temps, horizon):
    conn = sqlite3.connect("timeserie.db")
    preds = []

    for i in range(horizon):
        target_date = pd.Timestamp(forecast_start_date) + pd.Timedelta(days=i)
        is_off = int(target_date.weekday() >= 5)

        result = pd.read_sql_query(
            """
            SELECT AVG(consommation_mwh) AS moyenne
            FROM timeserie
            WHERE temp_moy_c BETWEEN ? AND ?
              AND week_end_ferie = ?
            """,
            conn,
            params=(future_temps[i] - 0.2, future_temps[i] + 0.2, is_off),
        )

        preds.append(result["moyenne"].iloc[0])

    conn.close()
    return preds


st.title("Exploration de CSV quotidiens")

with st.sidebar:
    st.header("Données")
    dataset_name = st.selectbox("Fichier", list(CSV_FILES.keys()))

csv_path = CSV_FILES[dataset_name]

if not csv_path.exists():
    st.error(f"Fichier introuvable : {csv_path}")
    st.stop()

df = load_csv(csv_path)
base_data = load_base_data()

st.header("Prévision")

horizon = 3

if not base_data.empty and "date" in base_data.columns:
    default_start = base_data["date"].max().date() + pd.Timedelta(days=1)
else:
    default_start = pd.Timestamp("2026-01-01").date()

forecast_start_date = st.date_input(
    "Date de début de prévision",
    value=default_start,
)

col1, col2, col3 = st.columns(3)
with col1:
    temp_j1 = st.number_input("Température J+1", value=12.0, format="%.2f")
with col2:
    temp_j2 = st.number_input("Température J+2", value=12.0, format="%.2f")
with col3:
    temp_j3 = st.number_input("Température J+3", value=12.0, format="%.2f")

future_temps = [temp_j1, temp_j2, temp_j3]

auto_compute = st.checkbox("Calcul automatique des prévisions", value=True)

if "forecast_cache" not in st.session_state:
    st.session_state["forecast_cache"] = {}

if "forecast_errors" not in st.session_state:
    st.session_state["forecast_errors"] = {}

do_compute = auto_compute
if not auto_compute:
    do_compute = st.button("Recalc forecast")

if do_compute:
    model_files = list_model_files()

    if not model_files:
        st.warning("Aucun modèle trouvé dans le dossier 'models/'.")

    global_metadata = load_sarimax_metadata()
    forecasts = {}
    forecast_errors = {}
    exog_preview = {}

    for model_name, model_path in model_files.items():
        try:
            loaded_obj = load_pickle(model_path)
            model, model_metadata = unwrap_model(loaded_obj)
            exog_cols = infer_exog_cols(
                model_name=model_name,
                model_metadata=model_metadata,
                global_metadata=global_metadata,
                model=model,
            )

            X_future = None
            if exog_cols:
                X_future = build_future_exog(
                    exog_cols=exog_cols,
                    start_date=pd.Timestamp(forecast_start_date),
                    horizon=horizon,
                    future_temps=future_temps,
                    base_data=base_data,
                )
                exog_preview[model_name] = X_future.reset_index(names="date")

            forecasts[model_name] = (
                [
                    v * 24 if model_name.startswith("Hmw") and v is not None else v
                    for v in predict_model(
                        model=model,
                        exog_cols=exog_cols,
                        X_future=X_future,
                        horizon=horizon,
                    )
                ]
                if not model_name.startswith("Dmw")
                else predict_model(
                    model=model, exog_cols=exog_cols, X_future=X_future, horizon=horizon
                )
            )

        except Exception as e:
            forecasts[model_name] = [None] * horizon
            forecast_errors[model_name] = str(e)

    forecasts["approximation sqlite"] = predict_sqlite_approximation(
        forecast_start_date, future_temps, horizon
    )
    st.session_state["forecast_cache"] = forecasts
    st.session_state["forecast_errors"] = forecast_errors
    st.session_state["exog_preview"] = exog_preview
else:
    forecasts = st.session_state.get("forecast_cache", {})
    forecast_errors = st.session_state.get("forecast_errors", {})
    exog_preview = st.session_state.get("exog_preview", {})

baseline_rows = []

try:
    conso_df = load_csv(DATA_DIR / "brest_metropole_daily.csv")
    meteo_df = load_csv(DATA_DIR / "brest_meteo_daily.csv")

    conso_df["date"] = pd.to_datetime(conso_df["date"]).dt.normalize()
    conso_df = conso_df.set_index("date").sort_index()

    meteo_df["date"] = pd.to_datetime(meteo_df["date"]).dt.normalize()
    meteo_df = meteo_df.set_index("date").sort_index()

    today = pd.Timestamp.today().normalize()

    forecast_ref = pd.Timestamp(forecast_start_date).normalize()

    targets = {
        "j+1": forecast_ref,
        "j+2": forecast_ref + pd.Timedelta(days=1),
        "j+3": forecast_ref + pd.Timedelta(days=2),
    }

    def get_value(df, date_ref, col):
        date_ref = pd.Timestamp(date_ref).normalize()
        if date_ref in df.index:
            value = df.loc[date_ref, col]
            if pd.notna(value):
                return float(value)
        return None

    def same_weekday_ref(target_date, years_back, max_shift=6):
        """
        Cherche une date autour de target_date - years_back
        qui tombe sur le même jour de semaine que target_date.
        """
        base_date = target_date - pd.DateOffset(years=years_back)

        candidates = [
            base_date + pd.Timedelta(days=shift)
            for shift in range(-max_shift, max_shift + 1)
        ]

        same_weekday = [d for d in candidates if d.weekday() == target_date.weekday()]

        if not same_weekday:
            return base_date

        return min(same_weekday, key=lambda d: abs((d - base_date).days)).normalize()

    def build_shifted_baseline(years_back):
        dates = {
            key: same_weekday_ref(target_date, years_back)
            for key, target_date in targets.items()
        }

        conso = {
            key: get_value(conso_df, date_ref, "conso_mwh")
            for key, date_ref in dates.items()
        }

        temp = {
            key: get_value(meteo_df, date_ref, "temp_moy_c")
            for key, date_ref in dates.items()
        }

        return dates, conso, temp

    def format_temp(temp_j1, temp_j3):
        if temp_j1 is None and temp_j3 is None:
            return "temp. n/a"
        if temp_j1 is None:
            return f"J+3 {temp_j3:.1f}°C"
        if temp_j3 is None:
            return f"J+1 {temp_j1:.1f}°C"
        return f"J+1 {temp_j1:.1f}°C / J+3 {temp_j3:.1f}°C"

    y1_dates, y1_conso, y1_temp = build_shifted_baseline(1)
    y2_dates, y2_conso, y2_temp = build_shifted_baseline(2)

    baseline_rows.append(
        {
            "modele": f"Rappel Y-1 aligné weekday ({format_temp(y1_temp['j+1'], y1_temp['j+3'])})",
            "conso_mwh_j+1": y1_conso["j+1"],
            "conso_mwh_j+2": y1_conso["j+2"],
            "conso_mwh_j+3": y1_conso["j+3"],
        }
    )

    baseline_rows.append(
        {
            "modele": f"Rappel Y-2 aligné weekday ({format_temp(y2_temp['j+1'], y2_temp['j+3'])})",
            "conso_mwh_j+1": y2_conso["j+1"],
            "conso_mwh_j+2": y2_conso["j+2"],
            "conso_mwh_j+3": y2_conso["j+3"],
        }
    )

    avg_row = {"modele": "Moyenne Y-1/Y-2 alignée weekday"}

    for key in ["j+1", "j+2", "j+3"]:
        values = [
            y1_conso[key],
            y2_conso[key],
        ]
        values = [v for v in values if v is not None]

        avg_row[f"conso_mwh_{key}"] = sum(values) / len(values) if values else None

    baseline_rows.append(avg_row)

except Exception as e:
    st.warning(f"Impossible de calculer les baselines : {e}")

rows = baseline_rows.copy()

for model_name, preds in (forecasts or {}).items():
    rows.append(
        {
            "modele": model_name,
            "conso_mwh_j+1": preds[0] if len(preds) > 0 else None,
            "conso_mwh_j+2": preds[1] if len(preds) > 1 else None,
            "conso_mwh_j+3": preds[2] if len(preds) > 2 else None,
            "statut": "Erreur" if model_name in forecast_errors else "OK",
        }
    )

results = pd.DataFrame(rows)
model_colors = {
    m: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
    for i, m in enumerate(results["modele"].dropna().unique())
}

if not results.empty:
    st.dataframe(
        results.style.apply(
            lambda r: [f"background-color: {model_colors.get(r['modele'], '')}22"]
            * len(r),
            axis=1,
        ),
        width="stretch",
    )
    st.header("Graphiques de prévision par modèle")

plot_df = results.dropna(subset=["modele"]).copy()

for col, title in [
    ("conso_mwh_j+1", "Prévisions J+1 par modèle"),
    ("conso_mwh_j+3", "Prévisions J+3 par modèle"),
]:
    fig = px.scatter(
        plot_df,
        x="modele",
        y=col,
        color="modele",
        color_discrete_map=model_colors,
        title=title,
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, width="stretch")
else:
    st.info(
        "Aucun résultat de prévision disponible. Cliquez sur 'Recalc forecast' ou activez le calcul automatique."
    )

if forecast_errors:
    with st.expander("Erreurs de prédiction"):
        for model_name, error in forecast_errors.items():
            st.warning(f"{model_name} : {error}")

if exog_preview:
    with st.expander("Variables exogènes envoyées aux modèles"):
        for model_name, X_future in exog_preview.items():
            st.subheader(model_name)
            st.dataframe(X_future, width="stretch")

st.header("Exploration")

if "date" not in df.columns:
    st.warning("Ce fichier ne contient pas de colonne `date`.")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not numeric_cols:
    st.warning("Aucune colonne numérique détectée.")
    st.stop()

min_date = df["date"].min().date()
max_date = df["date"].max().date()

start_date, end_date = st.slider(
    "Période",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)

filtered_df = df[
    (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
].copy()

selected_cols = st.multiselect(
    "Colonnes à afficher",
    numeric_cols,
    default=numeric_cols[:1],
)

if selected_cols:
    fig = px.line(
        filtered_df,
        x="date",
        y=selected_cols,
        title="Série temporelle",
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Statistiques descriptives"):
        st.dataframe(filtered_df[selected_cols].describe(), width="stretch")
else:
    st.info("Sélectionne au moins une colonne numérique.")
