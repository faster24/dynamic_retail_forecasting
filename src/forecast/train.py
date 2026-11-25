"""XGBoost (Optuna-tuned, WMAE-matched loss) and a fixed Random Forest."""
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from .metrics import holiday_weights, wmae
from .splits import between, cv_folds

SEED = 42
MAX_TREES = 2000
EARLY_STOP = 50


def make_xgb(params: dict, n_estimators: int = MAX_TREES, early_stopping: bool = True) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist",
        n_estimators=n_estimators,
        early_stopping_rounds=EARLY_STOP if early_stopping else None,
        eval_metric="mae",
        random_state=SEED,
        n_jobs=-1,
        **params,
    )


def fit_xgb(params, train: pd.DataFrame, features, val: pd.DataFrame | None = None, n_estimators=MAX_TREES):
    w = holiday_weights(train["IsHoliday"])
    if val is None:
        model = make_xgb(params, n_estimators, early_stopping=False)
        return model.fit(train[features], train["y"], sample_weight=w)
    model = make_xgb(params, n_estimators)
    model.fit(
        train[features], train["y"], sample_weight=w,
        eval_set=[(val[features], val["y"])], sample_weight_eval_set=[holiday_weights(val["IsHoliday"])],
        verbose=False,
    )
    return model


def cv_predict(params, df: pd.DataFrame, features):
    """Fit on each expanding fold; return (out-of-fold predictions, mean WMAE, mean best tree count)."""
    oof, scores, iters = [], [], []
    for train_end, val_start, val_end in cv_folds(df["Date"]):
        tr, va = df[df["Date"] < train_end], df[between(df, val_start, val_end)]
        model = fit_xgb(params, tr, features, val=va)
        pred = model.predict(va[features])
        scores.append(wmae(va["Weekly_Sales"], pred, va["IsHoliday"]))
        iters.append(model.best_iteration + 1)
        oof.append(va.assign(pred=pred))
    return pd.concat(oof), float(np.mean(scores)), int(np.mean(iters))


def tune_xgb(df: pd.DataFrame, features, n_trials: int = 50):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        _, score, n_trees = cv_predict(params, df, features)
        trial.set_user_attr("n_estimators", n_trees)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def fit_rf(train: pd.DataFrame, features) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=100, max_features=0.5, min_samples_leaf=5, max_samples=0.3, n_jobs=-1, random_state=SEED
    )
    return rf.fit(train[features].fillna(-1), train["y"], sample_weight=holiday_weights(train["IsHoliday"]))
