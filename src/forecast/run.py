"""Full pipeline: `uv run python -m forecast.run [--trials N]`.

Writes outputs/best_params.json, outputs/xgb_final.json, outputs/predictions.parquet, outputs/metrics.csv.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import baselines, metrics, stock
from .data import load_panel
from .features import FEATURES, MARKDOWN_FEATURES, build_features, model_rows
from .splits import BACKTEST_END, BACKTEST_START, BACKTEST_TRAIN_END, TEST_END, TEST_START, between
from .train import cv_predict, fit_rf, fit_xgb, tune_xgb

OUT = Path(__file__).resolve().parents[2] / "outputs"
NO_MD = [f for f in FEATURES if f not in MARKDOWN_FEATURES]


def evaluate_window(name, df, train_mask, eval_mask, params, n_trees, margins):
    tr, ev = df[train_mask], df[eval_mask].copy()
    xgb_model = fit_xgb(params, tr, FEATURES, n_estimators=n_trees)
    ev["pred_xgb"] = xgb_model.predict(ev[FEATURES])
    ev["pred_xgb_no_markdown"] = fit_xgb(params, tr, NO_MD, n_estimators=n_trees).predict(ev[NO_MD])
    ev["pred_rf"] = fit_rf(tr, FEATURES).predict(ev[FEATURES].fillna(-1))
    for col, values in baselines.predict_baselines(ev).items():
        ev[f"pred_{col}"] = values
    ev["stock_upper"] = stock.stock_upper(ev["pred_xgb"], ev["Dept"], margins)
    ev["window"] = name

    rows = []
    for col in [c for c in ev.columns if c.startswith("pred_")]:
        r = metrics.report(ev["Weekly_Sales"], ev[col], ev["IsHoliday"])
        rows.append({"window": name, "model": col.removeprefix("pred_"), **r})
    rows.append({"window": name, "model": "stock_coverage", "coverage": stock.coverage(ev["Weekly_Sales"], ev["stock_upper"])})
    return ev, xgb_model, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    df = model_rows(build_features(load_panel()))
    print(f"rows={len(df):,} features={len(FEATURES)}  ({time.time() - t0:.0f}s)")

    pre_test = df[df["Date"] < TEST_START]
    study = tune_xgb(pre_test, FEATURES, n_trials=args.trials)
    params, n_trees = study.best_params, study.best_trial.user_attrs["n_estimators"]
    print(f"best CV WMAE={study.best_value:,.0f} trees={n_trees} params={params}  ({time.time() - t0:.0f}s)")
    (OUT / "best_params.json").write_text(json.dumps({"params": params, "n_estimators": n_trees, "cv_wmae": study.best_value}, indent=2))

    oof, _, _ = cv_predict(params, pre_test, FEATURES)
    margins = stock.dept_margins(oof)

    test_ev, final_model, test_rows = evaluate_window(
        "test_2012", df, df["Date"] < TEST_START, between(df, TEST_START, TEST_END), params, n_trees, margins)
    final_model.save_model(OUT / "xgb_final.json")
    bt_ev, _, bt_rows = evaluate_window(
        "holiday_backtest_2011", df, df["Date"] <= BACKTEST_TRAIN_END, between(df, BACKTEST_START, BACKTEST_END),
        params, n_trees, margins)  # margins come from 2012 folds; backtest coverage is indicative only

    keep = ["window", "Store", "Dept", "Date", "IsHoliday", "Weekly_Sales", "stock_upper"]
    preds = pd.concat([test_ev, bt_ev])
    preds[keep + [c for c in preds.columns if c.startswith("pred_")]].to_parquet(OUT / "predictions.parquet", index=False)
    table = pd.DataFrame(test_rows + bt_rows)
    table.to_csv(OUT / "metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(1).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
