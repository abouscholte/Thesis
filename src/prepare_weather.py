"""Shared NOAA quality filtering and backward weather matching.

Retains the established 2024 policy. Precipitation/gust/ceiling fields are
preserved for audit, not automatically added to the four-feature model.
"""
from pathlib import Path
import json
import pandas as pd

weather_vars = ["temperature", "dew_point_temperature", "wind_speed", "wind_gust",
                "visibility", "ceiling_height", "precipitation"]
WEATHER_FEATURES = weather_vars + ["precipitation_trace", "ceiling_unlimited"]
STATIONS = {"ATL": "USW00013874", "DFW": "USW00003927", "DEN": "USW00003017",
            "ORD": "USW00094846", "CLT": "USW00013881", "LAX": "USW00023174",
            "PHX": "USW00023183", "LAS": "USW00023169", "SEA": "USW00024233",
            "LGA": "USW00014732"}

def clean_weather(df):
    result = df.copy()

    for var in weather_vars:
        source = pd.to_numeric(df[f"{var}_Source_Code"])
        quality = df[f"{var}_Quality_Code"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        keep = (source.eq(223) & quality.eq("1")) | (source.eq(343) & quality.isin(["1", "5", "A"])).fillna(False)

        result[var] = df[var].where(keep)

    result["precipitation_trace"] = result["precipitation_Measurement_Code"].eq("T").astype("boolean").where(result["precipitation"].notna())
    result["ceiling_unlimited"] = result["ceiling_height"].eq(22000).astype("boolean").where(result["ceiling_height"].notna())
    result["ceiling_height"] = result["ceiling_height"].mask(result["ceiling_unlimited"].fillna(False))
    result["weather_time_utc"] = pd.to_datetime(result["DATE"], utc=True, errors="raise")

    return result.sort_values("weather_time_utc")

def prepare_weather(project_root, year):
    """Match saved flights to observations at/before cutoff, at most 60 min old.

    Load the observation year plus January 1 of the following UTC year.
    Return combined flights and coverage, missingness, quality, station audits.
    """
    root = Path(project_root)
    airports = json.loads((root / "outputs/2024_audit/airport_selection.json").read_text())["airports"]
    flights = pd.read_csv(root / f"data/processed/flights_{year}_top10.csv.gz",
                          dtype={"CRSDepTime": "string"})
    assert set(flights["Origin"]) == set(airports)
    assert pd.to_datetime(flights["FlightDate"]).dt.year.eq(year).all()
    flights["prediction_cutoff_utc"] = pd.to_datetime(flights["prediction_cutoff_utc"], utc=True)
    assert flights["prediction_cutoff_utc"].notna().all()
    assert flights["prediction_cutoff_utc"].min() >= pd.Timestamp(f"{year}-01-01", tz="UTC")
    assert flights["prediction_cutoff_utc"].max() < pd.Timestamp(f"{year+1}-01-02", tz="UTC")
    flights["flight_id"] = range(len(flights))
    parts, checks, stations = [], [], []
    for airport in airports:
        yearly = []
        for observation_year in [year, year + 1]:
            path = root / f"data/raw/noaa/{observation_year}/noaa_{airport.lower()}_{observation_year}.psv"
            df = pd.read_csv(path, sep="|", low_memory=False)
            assert not df.empty, path.name
            assert df["STATION"].eq(STATIONS[airport]).all(), path.name
            dates = pd.to_datetime(df["DATE"], utc=True, errors="raise")
            assert dates.notna().all() and dates.dt.year.eq(observation_year).all(), path.name
            row = df.iloc[0]
            stations.append(dict(airport=airport, year=observation_year, station=row["STATION"],
                                 name=row["Station_name"], latitude=row["LATITUDE"], longitude=row["LONGITUDE"]))
            if observation_year == year + 1:
                df = df.loc[dates.lt(f"{year+1}-01-02")]
            for var in weather_vars:
                codes = df[[f"{var}_Source_Code", f"{var}_Quality_Code"]].copy()
                codes.columns = ["source", "quality"]
                for col in codes:
                    codes[col] = codes[col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).fillna("missing")
                checks.append(codes.value_counts().rename("rows").reset_index().assign(airport=airport, var=var))
            yearly.append(df)
        weather = clean_weather(pd.concat(yearly, ignore_index=True))
        assert weather["weather_time_utc"].notna().all(), airport
        assert not weather["weather_time_utc"].duplicated().any(), airport
        subset = flights.loc[flights["Origin"].eq(airport)]
        joined = pd.merge_asof(subset.sort_values("prediction_cutoff_utc"),
            weather[["weather_time_utc"] + WEATHER_FEATURES],
            left_on="prediction_cutoff_utc", right_on="weather_time_utc",
            direction="backward", tolerance=pd.Timedelta("60min"))
        parts.append(joined)
        print(f"{year} {airport}: {joined['weather_time_utc'].notna().mean():.2%} matched", flush=True)
    combined = pd.concat(parts, ignore_index=True).sort_values("flight_id")
    combined["weather_age_minutes"] = (combined["prediction_cutoff_utc"] - combined["weather_time_utc"]).dt.total_seconds() / 60
    assert len(combined) == len(flights) and combined["flight_id"].is_unique
    assert combined["weather_age_minutes"].dropna().between(0,60).all()
    coverage = combined.groupby("Origin").agg(flights=("flight_id", "size"),
        matched=("weather_time_utc", "count"), median_age_minutes=("weather_age_minutes", "median"))
    coverage["unmatched"] = coverage["flights"] - coverage["matched"]
    coverage["matched_pct"] = 100 * coverage["matched"] / coverage["flights"]
    missingness = combined[WEATHER_FEATURES].isna().groupby(combined["Origin"]).mean().mul(100)
    return combined, coverage, missingness, pd.concat(checks, ignore_index=True), pd.DataFrame(stations)


def save_weather(project_root, year, combined, coverage, missingness, quality_counts, stations):
    root = Path(project_root)
    out = root / "outputs/weather_audit"
    out.mkdir(parents=True, exist_ok=True)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    combined.to_csv(root / f"data/processed/flights_weather_{year}_top10.csv.gz", index=False)
    coverage.to_csv(out / f"all_airports_{year}_coverage.csv")
    missingness.to_csv(out / f"all_airport_{year}_missingness.csv")
    quality_counts.to_csv(out / f"all_airport_{year}_quality_codes.csv", index=False)
    stations.to_csv(out / f"station_mapping_{year}.csv", index=False)
