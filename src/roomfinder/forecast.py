"""Hourly demand forecast evaluated against honest baselines on a time split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from .config import OPEN_END, OPEN_START


def demand_table(requests: pd.DataFrame) -> pd.DataFrame:
    """Requests per (date, start_hour) including zero-demand cells."""
    days = pd.date_range(requests["request_date"].min(), requests["request_date"].max())
    hours = np.arange(OPEN_START, OPEN_END + 1)
    full = pd.MultiIndex.from_product([days, hours], names=["date", "hour"])
    counts = requests.groupby([requests["request_date"].rename("date"), "start_hour"]).size()
    counts.index.names = ["date", "hour"]
    df = counts.reindex(full, fill_value=0).rename("n").reset_index()
    df["weekday"] = df["date"].dt.dayofweek
    return df


def _design(df: pd.DataFrame) -> pd.DataFrame:
    wd = pd.get_dummies(df["weekday"], prefix="wd").reindex(
        columns=[f"wd_{i}" for i in range(7)], fill_value=0)
    hr = pd.get_dummies(df["hour"], prefix="h").reindex(
        columns=[f"h_{h}" for h in range(OPEN_START, OPEN_END + 1)], fill_value=0)
    return pd.concat([wd, hr], axis=1).astype(float)


def evaluate_forecast(requests: pd.DataFrame, test_from: str = "2025-11-01") -> dict:
    """Train on dates before ``test_from``, test on the rest (no shuffling, no leakage)."""
    df = demand_table(requests)
    train, test = df[df["date"] < test_from], df[df["date"] >= test_from]
    y_tr, y_te = train["n"].to_numpy(), test["n"].to_numpy()

    global_mean = np.full(len(test), y_tr.mean())
    profile = train.groupby(["weekday", "hour"])["n"].mean()
    profile_pred = np.array(
        [profile.get((w, h), y_tr.mean())
         for w, h in zip(test["weekday"], test["hour"], strict=True)])
    glm = PoissonRegressor(alpha=1.0, max_iter=500).fit(_design(train), y_tr)
    glm_pred = glm.predict(_design(test))

    def metrics(pred: np.ndarray) -> dict:
        pred = np.clip(pred, 1e-6, None)
        return {"mae": float(mean_absolute_error(y_te, pred)),
                "poisson_deviance": float(mean_poisson_deviance(y_te, pred))}

    res = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "global_mean_baseline": metrics(global_mean),
        "weekday_hour_profile": metrics(profile_pred),
        "poisson_glm": metrics(glm_pred),
    }
    best_base = min(res["global_mean_baseline"]["mae"], res["weekday_hour_profile"]["mae"])
    res["glm_beats_best_baseline"] = bool(res["poisson_glm"]["mae"] < best_base)
    res["glm_relative_mae_change_vs_best_baseline"] = float(
        (res["poisson_glm"]["mae"] - best_base) / best_base)
    return res
