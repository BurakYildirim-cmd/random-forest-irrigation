import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    df = df.sort_values(["line", "timestamp"])

    df["instant_water"] = df.groupby("line")["current_volume"].diff().fillna(0)
    df = df[df["instant_water"] >= 0]

    if "humidity_soil" in df:
        df["soil_temp_ratio"] = df["humidity_soil"] / (df["temperature_soil"].abs() + 1)

    if "solar_radiation" in df and "wind_speed" in df:
        df["rad_wind"] = df["solar_radiation"] * df["wind_speed"]

    if "temperature_meteo" in df and "humidity_env_meteo" in df:
        df["vpd_like"] = df["temperature_meteo"] * (100 - df["humidity_env_meteo"]) / 100

    df["humidity_soil_lag1"] = df.groupby("line")["humidity_soil"].shift(1)

    return df.dropna()