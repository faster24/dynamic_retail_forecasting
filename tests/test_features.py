import numpy as np
import pandas as pd

from forecast.data import build_panel
from forecast.features import FEATURES, build_features


def toy_panel(drop_dates=(), weeks=110):
    dates = pd.date_range("2010-02-05", periods=weeks, freq="7D")
    train = pd.DataFrame({"Store": 1, "Dept": 1, "Date": dates, "Weekly_Sales": np.arange(weeks, dtype=float) + 1,
                          "IsHoliday": False})
    train = train[~train["Date"].isin(pd.to_datetime(list(drop_dates)))]
    # second series covering all weeks keeps every date in the grid
    other = pd.DataFrame({"Store": 1, "Dept": 2, "Date": dates, "Weekly_Sales": 5.0, "IsHoliday": False})
    train = pd.concat([train, other])
    features = pd.DataFrame({"Store": 1, "Date": dates, "IsHoliday": dates.isin(pd.to_datetime(["2010-11-26", "2011-11-25"]))})
    for i in range(1, 6):
        features[f"MarkDown{i}"] = np.nan
    stores = pd.DataFrame({"Store": [1], "Type": ["A"], "Size": [100000]})
    return build_panel(train, stores, features)


def s1(df):
    return df[df["Dept"] == 1].set_index("Date")


def test_lag_is_exact_across_gap():
    df = s1(build_features(toy_panel(drop_dates=["2010-03-05"])))
    assert np.isnan(df.loc["2010-03-12", "lag_1"])  # previous week missing -> NaN, not the week before
    assert df.loc["2010-03-12", "lag_2"] == 4.0  # 2010-02-26 is week 4 (sales = 4)


def test_lag_52_hits_same_holiday_week():
    df = s1(build_features(toy_panel()))
    assert df.loc["2011-11-25", "lag_52"] == df.loc["2010-11-26", "y"]


def test_no_feature_uses_current_week_sales():
    panel = toy_panel()
    row = panel.index[(panel["Dept"] == 1) & (panel["Date"] == "2011-06-03")][0]
    changed = panel.copy()
    changed.loc[row, "y"] = 1e9  # only this week's target changes
    a, b = build_features(panel), build_features(changed)
    key = (a["Dept"] == 1) & (a["Date"] == "2011-06-03")
    pd.testing.assert_frame_equal(a.loc[key, FEATURES], b.loc[key, FEATURES])
    later = (a["Dept"] == 1) & (a["Date"] == "2011-06-10")
    assert b.loc[later, "lag_1"].item() == 1e9  # ...and it does reach next week's lag


def test_negative_sales_clipped_in_target_only():
    train, stores, features = (pd.DataFrame({"Store": [1], "Dept": [1], "Date": pd.to_datetime(["2010-02-05"]),
                                             "Weekly_Sales": [-5.0], "IsHoliday": [False]}),
                               pd.DataFrame({"Store": [1], "Type": ["A"], "Size": [1]}),
                               pd.DataFrame({"Store": [1], "Date": pd.to_datetime(["2010-02-05"]), "IsHoliday": [False],
                                             **{f"MarkDown{i}": [np.nan] for i in range(1, 6)}}))
    p = build_panel(train, stores, features)
    assert p.loc[0, "Weekly_Sales"] == -5.0 and p.loc[0, "y"] == 0.0
