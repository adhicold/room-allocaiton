"""Can request features (plus occupancy) predict Waitlisted/Rejected outcomes?

Evaluated with a chronological split and compared with a prior-only baseline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

FEATURES = ["start_hour", "duration_hours", "required_capacity", "projector_required",
            "computers_required", "weekday", "month", "n_feasible_rooms", "purpose_code"]


def build_features(requests: pd.DataFrame, n_feasible: pd.DataFrame) -> pd.DataFrame:
    """Pre-decision features only; ``status`` is never used as an input."""
    df = requests.merge(n_feasible, on="request_id")
    df["weekday"] = df["request_date"].dt.dayofweek
    df["month"] = df["request_date"].dt.month
    df["purpose_code"] = df["purpose"].astype("category").cat.codes
    return df


def evaluate_risk_model(requests: pd.DataFrame, n_feasible: pd.DataFrame,
                        train_frac: float = 0.8) -> dict:
    """Binary task: unmet (Waitlisted/Rejected) vs Approved; Cancelled rows excluded."""
    df = build_features(requests, n_feasible)
    df = df[df["status"] != "Cancelled"].sort_values(["request_date", "request_id"])
    y = df["status"].isin(["Waitlisted", "Rejected"]).astype(int)
    cut = int(len(df) * train_frac)
    x_tr, x_te = df[FEATURES].iloc[:cut], df[FEATURES].iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]

    models = {
        "prior_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=RANDOM_STATE),
    }
    out: dict = {"train_rows": int(cut), "test_rows": int(len(df) - cut),
                 "test_positive_rate": float(y_te.mean())}
    for name, m in models.items():
        p = m.fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
        out[name] = {"roc_auc": float(roc_auc_score(y_te, p)),
                     "pr_auc": float(average_precision_score(y_te, p)),
                     "log_loss": float(log_loss(y_te, p))}
    best_auc = max(out["logistic_regression"]["roc_auc"], out["hist_gradient_boosting"]["roc_auc"])
    out["predictive_signal_found"] = bool(best_auc >= 0.60)
    return out
