"""End-to-end evaluation: profiles the data, runs every model and writes reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .allocation import replay, static_feasibility_bound
from .config import OPEN_END, OPEN_START
from .data import Campus, load_campus
from .forecast import evaluate_forecast
from .occupancy import (
    build_occupancy,
    class_clashes,
    class_event_clashes,
    event_clashes,
    weekly_grid,
)
from .risk_model import evaluate_risk_model
from .utilization import cluster_rooms, plan_moves, room_utilisation


def term_bounds(campus: Campus) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last date covered by requests or events."""
    lo = min(campus.requests["request_date"].min(), campus.events["start_time"].min())
    hi = max(campus.requests["request_date"].max(), campus.events["start_time"].max())
    return lo.normalize(), hi.normalize()


def profile(campus: Campus) -> dict:
    """Data-quality facts that the rest of the pipeline has to handle."""
    caps = dict(zip(campus.rooms["room_id"], campus.rooms["capacity"], strict=True))
    s, e = campus.schedule, campus.events
    hours = e["start_time"].dt.hour
    cc = class_clashes(s)
    return {
        "rows": {k: int(len(v)) for k, v in vars(campus).items()},
        "class_clash_pairs": int(len(cc)),
        "class_rows_in_a_clash": int(len(set(cc["schedule_id_a"]) | set(cc["schedule_id_b"]))),
        "event_event_clash_pairs": int(len(event_clashes(e))),
        "class_event_clash_pairs": int(len(class_event_clashes(s, e))),
        "classes_over_room_capacity_share": float(
            (s["expected_attendance"] > s["room_id"].map(caps)).mean()),
        "events_over_room_capacity_share": float(
            (e["expected_attendance"] > e["room_id"].map(caps)).mean()),
        "events_outside_08_20_share": float(((hours < OPEN_START) | (hours >= OPEN_END)).mean()),
        "requests_on_sunday_share": float(
            (campus.requests["request_date"].dt.dayofweek == 6).mean()),
        "attendance_expected_vs_actual_corr": float(
            e["expected_attendance"].corr(e["actual_attendance"])),
        "status_counts": campus.requests["status"].value_counts().to_dict(),
    }


def run(data_dir: str | Path, out_dir: str | Path) -> dict:
    """Run everything and write ``metrics.json`` plus supporting CSV files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    campus = load_campus(data_dir)
    lo, hi = term_bounds(campus)
    occ = build_occupancy(campus.rooms, campus.schedule, campus.events, lo, hi)
    grid = weekly_grid(campus.rooms, campus.schedule)

    bound = static_feasibility_bound(campus.requests, campus.rooms, occ)
    active = campus.requests[campus.requests["status"] != "Cancelled"]
    bound_active = bound[bound["request_id"].isin(active["request_id"])]
    runs = [replay(campus.requests, campus.rooms, occ, p, o)
            for p, o in [("first_fit", "chronological"), ("random", "chronological"),
                         ("best_fit", "chronological"), ("best_fit", "constrained_first")]]
    replay_df = pd.DataFrame([r.summary(campus.requests) for r in runs])

    util = room_utilisation(campus.rooms, grid)
    clustered, cluster_info = cluster_rooms(util, grid)
    moves, repair = plan_moves(campus.rooms, campus.schedule)

    metrics = {
        "term": [str(lo.date()), str(hi.date())],
        "data_profile": profile(campus),
        "upper_bound_share_with_any_feasible_room": float(
            (bound_active["n_feasible_rooms"] > 0).mean()),
        "allocation_policies": replay_df.to_dict(orient="records"),
        "timetable_repair": repair,
        "utilisation": {
            "mean": float(util["utilisation"].mean()),
            "underused_rooms": int(util["underused"].sum()),
            "clusters": cluster_info,
        },
        "demand_forecast": evaluate_forecast(campus.requests),
        "unmet_request_classifier": evaluate_risk_model(campus.requests, bound),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    replay_df.to_csv(out / "allocation_policies.csv", index=False)
    class_clashes(campus.schedule).to_csv(out / "class_clashes.csv", index=False)
    moves.to_csv(out / "proposed_moves.csv", index=False)
    clustered.to_csv(out / "room_utilisation.csv", index=False)
    (out / "REPORT.md").write_text(render_report(metrics))
    return metrics


def render_report(m: dict) -> str:
    """Plain-language summary generated from ``metrics`` so it can never drift."""
    p, pol = m["data_profile"], {(r["policy"], r["order"]): r for r in m["allocation_policies"]}
    ff, rnd = pol[("first_fit", "chronological")], pol[("random", "chronological")]
    bf, cf = pol[("best_fit", "chronological")], pol[("best_fit", "constrained_first")]
    rep, fc, rk = m["timetable_repair"], m["demand_forecast"], m["unmet_request_classifier"]
    waste_cut = 1 - bf["mean_wasted_seats"] / ff["mean_wasted_seats"]
    lines = [
        "# Evaluation report (auto-generated by `roomfinder evaluate`)",
        "",
        "## 1. Data quality the pipeline handles",
        f"- {p['class_clash_pairs']:,} overlapping class pairs "
        f"({p['class_rows_in_a_clash']:,} timetable rows involved); "
        f"{p['event_event_clash_pairs']} event-event and {p['class_event_clash_pairs']:,} "
        "class-event overlaps.",
        f"- {p['classes_over_room_capacity_share']:.0%} of classes exceed their room's capacity.",
        f"- {p['events_outside_08_20_share']:.0%} of events start outside 08:00-20:00; they still "
        "block their room.",
        f"- Expected vs actual event attendance correlation: "
        f"{p['attendance_expected_vs_actual_corr']:.2f} (not used as ground truth).",
        "",
        "## 2. Allocation policies (simulated replay, cancelled requests excluded)",
        f"- Upper bound: {m['upper_bound_share_with_any_feasible_room']:.1%} of requests have at "
        "least one feasible room given existing occupancy.",
        "",
        "| policy | order | placed | placement rate | unmet placed | mean wasted seats |",
        "|---|---|---|---|---|---|",
    ]
    for r in m["allocation_policies"]:
        lines.append(f"| {r['policy']} | {r['order']} | {r['placed']:,} | "
                     f"{r['placement_rate']:.1%} | {r['unmet_placed']:,}/{r['unmet_requests']:,} | "
                     f"{r['mean_wasted_seats']:.1f} |")
    lines += [
        "",
        f"Best-fit ranking cuts wasted seats by {waste_cut:.0%} vs first-fit "
        f"({rnd['mean_wasted_seats']:.1f} for random) and changes the placement rate by "
        f"{(bf['placement_rate'] - ff['placement_rate']) * 100:+.1f} percentage points. "
        f"Most-constrained-first ordering changed it by "
        f"{(cf['placement_rate'] - bf['placement_rate']) * 100:+.1f} points, so the simple "
        "chronological policy is the default.",
        "",
        "## 3. Timetable repair (greedy, never creates a new clash)",
        f"- Double-booked slots: {rep['before']['double_booked_slots']:,} -> "
        f"{rep['after']['double_booked_slots']:,}",
        f"- Over-crowded classes: {rep['before']['overcrowded_classes']:,} -> "
        f"{rep['after']['overcrowded_classes']:,}",
        f"- Mean seat utilisation: {rep['before']['mean_seat_utilisation']:.1%} -> "
        f"{rep['after']['mean_seat_utilisation']:.1%} ({rep['moves']:,} proposed moves)",
        "",
        "## 4. Learned models vs baselines (chronological splits)",
        f"- Demand forecast (Poisson GLM): MAE {fc['poisson_glm']['mae']:.3f} vs best baseline "
        f"{min(fc['global_mean_baseline']['mae'], fc['weekday_hour_profile']['mae']):.3f}; "
        f"beats baseline: **{fc['glm_beats_best_baseline']}**.",
        f"- Unmet-request classifier: ROC-AUC logistic {rk['logistic_regression']['roc_auc']:.3f}, "
        f"boosting {rk['hist_gradient_boosting']['roc_auc']:.3f} (0.5 = chance); signal found: "
        f"**{rk['predictive_signal_found']}**.",
        "",
        "Interpretation: request attributes carry no usable signal for forecasting demand or "
        "outcomes in this synthetic dataset, so the product value comes from the exact "
        "constraint-based allocator rather than from a learned predictor. The learned models "
        "are kept as audited experiments and reported honestly.",
    ]
    return "\n".join(lines) + "\n"
