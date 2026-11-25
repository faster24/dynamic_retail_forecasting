"""Naive forecasts, read from the leakage-safe lag columns. Missing history falls back so every row gets a value."""
import pandas as pd


def predict_baselines(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "naive_last": df["lag_1"].fillna(df["roll_mean_4"]).fillna(df["lag_52"]).fillna(0),
            "naive_last_year": df["lag_52"].fillna(df["lag_1"]).fillna(df["roll_mean_4"]).fillna(0),
            "mean_4": df["roll_mean_4"].fillna(df["lag_52"]).fillna(0),
        },
        index=df.index,
    )
