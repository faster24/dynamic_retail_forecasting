import pandas as pd

from forecast.splits import BACKTEST_START, BACKTEST_TRAIN_END, TEST_START, cv_folds


def test_folds_before_test_and_disjoint():
    dates = pd.Series(pd.date_range("2010-02-05", "2012-10-26", freq="7D"))
    folds = list(cv_folds(dates))
    assert len(folds) == 3
    assert folds[-1][2] < TEST_START
    for (_, _, end), (nxt_train_end, nxt_start, _) in zip(folds, folds[1:]):
        assert end < nxt_start
    assert all(train_end == start for train_end, start, _ in folds)  # train strictly before val


def test_test_window_is_12_weeks():
    dates = pd.date_range("2010-02-05", "2012-10-26", freq="7D")
    assert (dates >= TEST_START).sum() == 12
    assert BACKTEST_TRAIN_END < BACKTEST_START
