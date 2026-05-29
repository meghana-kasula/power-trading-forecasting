import numpy as np
import pandas as pd


WEATHER_RENAME_MAP = {
    "temperature_2m": "national_temperature_2m",
    "apparent_temperature": "national_apparent_temperature",
    "relative_humidity_2m": "national_relative_humidity_2m",
    "precipitation": "national_precipitation",
    "wind_speed_10m": "national_wind_speed_10m",
    "rain": "national_rain",
    "wind_direction_10m": "national_wind_direction_10m",
}

LAG_CONFIG = {
    "MCP (Rs/MWh) *": [1, 4, 8, 12, 96],
    "Purchase Bid (MW)": [1, 4, 8],
    "Sell Bid (MW)": [1, 4, 8],
    "MCV (MW)": [1, 4, 8],
    "Final Scheduled Volume (MW)": [1, 4, 8],
    "national_temperature_2m": [1, 96],
    "national_apparent_temperature": [1, 96],
    "national_precipitation": [1, 96],
    "national_wind_speed_10m": [1, 96],
    "national_rain": [1, 96],
    "national_wind_direction_10m": [1, 3, 12],
}

BID_BINS = [0, 5000, 10000, 20000, 50000]


def clean_input_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns=WEATHER_RENAME_MAP, inplace=True)
    df = df.bfill().ffill()

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["Date"] = df["Date"].astype("int64") // 10**9

    if "Time Block" in df.columns:
        df["Time Block"] = df["Time Block"].astype(str).str.split(" - ").str[0]
        df["Time Block"] = pd.to_datetime(
            df["Time Block"],
            format="%H:%M",
            errors="coerce",
        )
        df["Time Block"] = df["Time Block"].dt.hour * 60 + df["Time Block"].dt.minute

    if {"Purchase Bid (MW)", "Sell Bid (MW)"}.issubset(df.columns):
        df["Demand_Supply_Gap"] = df["Purchase Bid (MW)"] - df["Sell Bid (MW)"]

    if "Purchase Bid (MW)" in df.columns:
        df["Bid_Category"] = pd.cut(
            df["Purchase Bid (MW)"],
            bins=BID_BINS,
            labels=False,
            include_lowest=True,
        )

    return df


def add_recursive_features(df):
    df = df.copy()
    df["MCP_log"] = np.log1p(df["MCP (Rs/MWh) *"])

    for col, lags in LAG_CONFIG.items():
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df["MCP_roll_mean_4"] = df["MCP (Rs/MWh) *"].rolling(4).mean()
    df["MCP_roll_mean_12"] = df["MCP (Rs/MWh) *"].rolling(12).mean()
    df["MCP_roll_std_4"] = df["MCP (Rs/MWh) *"].rolling(4).std()
    df["MCP_roll_std_12"] = df["MCP (Rs/MWh) *"].rolling(12).std()

    return df.dropna()


def infer_display_index(raw_df, periods=96):
    raw_df = raw_df.copy()
    raw_df.columns = raw_df.columns.str.strip()

    if len(raw_df) > periods:
        forecast_rows = raw_df.tail(periods)
        parsed = parse_timestamp_index(forecast_rows)
        if parsed is not None:
            return parsed

    fallback = pd.date_range(start="2025-04-06", periods=periods, freq="15min")
    parsed = parse_timestamp_index(raw_df.tail(min(len(raw_df), periods)))
    if parsed is None:
        return fallback

    start = parsed.iloc[-1] + pd.Timedelta(minutes=15)
    return pd.date_range(start=start, periods=periods, freq="15min")


def parse_timestamp_index(df):
    if "Date" not in df.columns:
        return None

    dates = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if dates.isna().any():
        return None

    if "Time Block" not in df.columns:
        return pd.Series(dates).reset_index(drop=True)

    start_times = df["Time Block"].astype(str).str.split(" - ").str[0]
    hours = []
    minutes = []

    for value in start_times:
        try:
            hour_text, minute_text = value.split(":")[:2]
            hours.append(int(hour_text))
            minutes.append(int(minute_text))
        except ValueError:
            return None

    offsets = [pd.Timedelta(hours=hour, minutes=minute) for hour, minute in zip(hours, minutes)]
    timestamps = pd.Series(dates).reset_index(drop=True) + pd.Series(offsets)
    return timestamps
