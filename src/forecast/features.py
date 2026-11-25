"""Leakage-safe features: every sales-based feature only uses weeks before the target week."""
import numpy as np
import pandas as pd

from .data import MARKDOWNS

KEYS = ["Store", "Dept"]

# Week-ending Fridays of each flagged holiday week (Kaggle data description), plus 2009 so
# "weeks since" is defined for early 2010.
HOLIDAYS = {
    "superbowl": ["2009-02-13", "2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "labor": ["2009-09-11", "2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "thanksgiving": ["2009-11-27", "2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "xmas": ["2009-12-31", "2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}
HOLIDAY_CODE = {name: i + 1 for i, name in enumerate(HOLIDAYS)}  # 0 = no holiday

LAGS = [1, 2, 4, 52]
ROLL = [4, 12]
SALES_FEATURES = [f"lag_{k}" for k in LAGS] + [f"roll_{s}_{w}" for w in ROLL for s in ("mean", "std")]
CALENDAR_FEATURES = ["week", "month", "year"]
HOLIDAY_FEATURES = (
    ["IsHoliday", "holiday_name", "xmas_shop_days"]
    + [f"weeks_to_{h}" for h in HOLIDAYS]
    + [f"weeks_since_{h}" for h in HOLIDAYS]
)
MARKDOWN_FEATURES = MARKDOWNS + ["md_total", "md_available"]
STORE_FEATURES = ["Store", "Dept", "Type", "Size"]
FEATURES = SALES_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES + MARKDOWN_FEATURES + STORE_FEATURES


def _holiday_calendar(dates: pd.Series) -> pd.DataFrame:
    """Per unique date: holiday name code, weeks to next / since last of each holiday, xmas shopping days."""
    d = pd.DatetimeIndex(sorted(dates.unique()))
    out = pd.DataFrame({"Date": d})
    out["holiday_name"] = 0
    for name, days in HOLIDAYS.items():
        h = pd.DatetimeIndex(days).values
        idx = np.searchsorted(h, d.values)  # first holiday >= date
        nxt = h[np.minimum(idx, len(h) - 1)]
        prev_idx = np.searchsorted(h, d.values, side="right") - 1  # last holiday <= date
        prv = h[np.maximum(prev_idx, 0)]
        out[f"weeks_to_{name}"] = (nxt - d.values) // np.timedelta64(7, "D")
        out[f"weeks_since_{name}"] = (d.values - prv) // np.timedelta64(7, "D")
        out.loc[out["Date"].isin(h), "holiday_name"] = HOLIDAY_CODE[name]
    # Dates are week-ending Fridays; count days of this week in the last 7 shopping days (Dec 18-24).
    def shop_days(end):
        week = pd.date_range(end - pd.Timedelta(days=6), end)
        return int(((week.month == 12) & (week.day >= 18) & (week.day <= 24)).sum())

    out["xmas_shop_days"] = [shop_days(x) for x in d]
    return out


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features to a panel from data.build_panel (must be a full grid, sorted by Store, Dept, Date)."""
    df = panel.sort_values(KEYS + ["Date"]).reset_index(drop=True)
    g = df.groupby(KEYS, sort=False)["y"]
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k)  # grid is complete, so a row shift is an exact k-week lag
    prev = df.groupby(KEYS, sort=False)["lag_1"]
    for w in ROLL:
        df[f"roll_mean_{w}"] = prev.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = prev.transform(lambda s: s.rolling(w, min_periods=2).std())

    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df = df.merge(_holiday_calendar(df["Date"]), on="Date", how="left")
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df["Type"] = df["Type"].map({"A": 0, "B": 1, "C": 2})
    return df


def model_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with an observed target (grid filler weeks are only there to make lags exact)."""
    return df[df["Weekly_Sales"].notna()].reset_index(drop=True)
