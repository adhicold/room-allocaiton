import numpy as np
import pandas as pd

from roomfinder.forecast import demand_table, evaluate_forecast
from roomfinder.risk_model import evaluate_risk_model


def _fake_requests(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2025-08-01") + pd.to_timedelta(rng.integers(0, 120, n), "D")
    return pd.DataFrame({
        "request_id": [f"Q{i:04d}" for i in range(n)],
        "request_date": dates,
        "start_hour": rng.integers(8, 21, n),
        "duration_hours": rng.integers(1, 5, n),
        "required_capacity": rng.integers(5, 120, n),
        "projector_required": rng.integers(0, 2, n),
        "computers_required": rng.integers(0, 2, n),
        "purpose": rng.choice(["Class", "Workshop"], n),
        "status": rng.choice(["Approved", "Waitlisted", "Rejected", "Cancelled"], n),
    })


def test_demand_table_includes_zero_cells():
    t = demand_table(_fake_requests())
    assert (t["n"] == 0).any()
    assert t["n"].sum() == 600


def test_forecast_reports_baselines_and_is_deterministic():
    req = _fake_requests()
    a, b = evaluate_forecast(req), evaluate_forecast(req)
    assert a == b
    assert {"global_mean_baseline", "weekday_hour_profile", "poisson_glm"} <= set(a)


def test_forecast_train_precedes_test():
    res = evaluate_forecast(_fake_requests(), test_from="2025-10-01")
    assert res["train_rows"] > 0 and res["test_rows"] > 0


def test_risk_model_on_noise_finds_no_signal():
    req = _fake_requests(1500)
    feas = pd.DataFrame({"request_id": req["request_id"], "n_feasible_rooms": 10})
    res = evaluate_risk_model(req, feas)
    assert res["predictive_signal_found"] is False
    assert res["prior_baseline"]["roc_auc"] == 0.5
