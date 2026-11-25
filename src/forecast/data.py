"""Load the Walmart CSVs, merge them and put every Store x Dept series on a full weekly grid."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MARKDOWNS = [f"MarkDown{i}" for i in range(1, 6)]
REQUIRED = ("train.csv", "stores.csv", "features.csv")


def load_raw(raw_dir: Path = RAW_DIR):
    raw_dir = Path(raw_dir)
    missing = [f for f in REQUIRED if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. Download the Kaggle 'Walmart Recruiting - Store Sales "
            f"Forecasting' data and extract the CSVs there."
        )
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["Date"])
    stores = pd.read_csv(raw_dir / "stores.csv")
    features = pd.read_csv(raw_dir / "features.csv", parse_dates=["Date"])
    return train, stores, features


def build_panel(train: pd.DataFrame, stores: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per Store x Dept x week (all weeks in train). Missing weeks have NaN Weekly_Sales."""
    dates = pd.DatetimeIndex(sorted(train["Date"].unique()))
    series = train[["Store", "Dept"]].drop_duplicates()
    grid = series.merge(pd.DataFrame({"Date": dates}), how="cross")
    df = grid.merge(train[["Store", "Dept", "Date", "Weekly_Sales"]], on=["Store", "Dept", "Date"], how="left")

    feats = features[["Store", "Date", "IsHoliday", *MARKDOWNS]].copy()
    feats["md_available"] = feats[MARKDOWNS].notna().any(axis=1).astype(int)
    feats[MARKDOWNS] = feats[MARKDOWNS].fillna(0).clip(lower=0)  # 44 negative markdowns in the raw data
    feats["md_total"] = feats[MARKDOWNS].sum(axis=1)
    df = df.merge(feats, on=["Store", "Date"], how="left")
    df = df.merge(stores, on="Store", how="left")
    if df["IsHoliday"].isna().any() or df["Size"].isna().any():
        raise ValueError("features.csv / stores.csv do not cover every Store x Date in train.csv")

    df["IsHoliday"] = df["IsHoliday"].astype(bool)
    df["y"] = df["Weekly_Sales"].clip(lower=0)  # training target: returns clipped to 0
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


def load_panel(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return build_panel(*load_raw(raw_dir))
