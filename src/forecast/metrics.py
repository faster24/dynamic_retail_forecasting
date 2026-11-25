"""WMAE (Kaggle metric: holiday weeks weigh 5x) plus MAE / RMSE and holiday vs normal MAE."""
import numpy as np


def holiday_weights(is_holiday):
    return np.where(np.asarray(is_holiday, dtype=bool), 5.0, 1.0)


def wmae(y, yhat, is_holiday) -> float:
    w = holiday_weights(is_holiday)
    return float(np.sum(w * np.abs(np.asarray(y) - np.asarray(yhat))) / np.sum(w))


def report(y, yhat, is_holiday) -> dict:
    y, yhat, h = np.asarray(y, float), np.asarray(yhat, float), np.asarray(is_holiday, bool)
    err = y - yhat
    return {
        "WMAE": wmae(y, yhat, h),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE_holiday": float(np.mean(np.abs(err[h]))) if h.any() else np.nan,
        "MAE_normal": float(np.mean(np.abs(err[~h]))) if (~h).any() else np.nan,
        "n": int(len(y)),
    }
