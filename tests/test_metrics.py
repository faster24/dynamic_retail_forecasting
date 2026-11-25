import numpy as np

from forecast.metrics import report, wmae
from forecast.stock import coverage, dept_margins, stock_upper
import pandas as pd


def test_wmae_hand_computed():
    # errors 10 (holiday, w=5) and 2 (normal, w=1): (50 + 2) / 6
    assert np.isclose(wmae([110, 50], [100, 48], [True, False]), 52 / 6)


def test_report_splits_holiday():
    r = report([110, 50], [100, 48], [True, False])
    assert r["MAE_holiday"] == 10 and r["MAE_normal"] == 2


def test_stock_margin_and_coverage():
    oof = pd.DataFrame({"Dept": [1] * 10, "Weekly_Sales": np.arange(10.0), "pred": np.zeros(10)})
    m = dept_margins(oof)
    up = stock_upper(pd.Series([0.0, 0.0]), pd.Series([1, 99]), m)  # 99 unseen -> global fallback
    assert np.allclose(up, [m.loc[1], m.loc[-1]])
    assert coverage([1, 100], up) == 0.5
