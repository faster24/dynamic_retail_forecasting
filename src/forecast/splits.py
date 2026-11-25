"""Time-based splits. Nothing is ever split randomly."""
import pandas as pd

TEST_START = pd.Timestamp("2012-08-10")  # last 12 weeks: 2012-08-10 .. 2012-10-26
TEST_END = pd.Timestamp("2012-10-26")
N_FOLDS, FOLD_WEEKS = 3, 12
BACKTEST_TRAIN_END = pd.Timestamp("2011-10-28")  # 2011 holiday-season backtest
BACKTEST_START, BACKTEST_END = pd.Timestamp("2011-11-04"), pd.Timestamp("2011-12-30")


def cv_folds(dates: pd.Series):
    """3 expanding-window folds of 12 validation weeks each, all before the test window.
    Yields (train_end_exclusive, val_start, val_end)."""
    weeks = sorted(d for d in pd.Series(dates).unique() if d < TEST_START)
    val_weeks = weeks[-N_FOLDS * FOLD_WEEKS:]
    for i in range(N_FOLDS):
        chunk = val_weeks[i * FOLD_WEEKS:(i + 1) * FOLD_WEEKS]
        yield chunk[0], chunk[0], chunk[-1]


def between(df: pd.DataFrame, start, end) -> pd.Series:
    return (df["Date"] >= start) & (df["Date"] <= end)
