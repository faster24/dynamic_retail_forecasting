"""Stocking signal: [forecast, forecast + margin], margin = per-Dept q90 of past under-forecast."""
import pandas as pd


def dept_margins(oof: pd.DataFrame, q: float = 0.9) -> pd.Series:
    """oof needs Dept, Weekly_Sales, pred. Returns margin per Dept; index -1 holds the global fallback."""
    resid = oof["Weekly_Sales"] - oof["pred"]
    m = resid.groupby(oof["Dept"]).quantile(q).clip(lower=0)
    m.loc[-1] = max(resid.quantile(q), 0.0)
    return m


def stock_upper(pred: pd.Series, dept: pd.Series, margins: pd.Series) -> pd.Series:
    margin = dept.map(margins).fillna(margins.loc[-1])
    return pred.clip(lower=0) + margin.to_numpy()


def coverage(actual, upper) -> float:
    return float((pd.Series(actual).to_numpy() <= pd.Series(upper).to_numpy()).mean())
