"""Replay every historical request against the occupancy model.

Because ``room_requests.csv`` has no assigned room (it is demand *before*
allocation) there is no ground-truth label. We therefore evaluate allocation
policies by simulation: place requests one by one, let each placement block its
room, and compare how many requests each policy manages to serve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, Weights
from .matcher import Request, feasible_mask, score_rooms
from .occupancy import Occupancy

Policy = Literal["first_fit", "random", "best_fit"]
Order = Literal["chronological", "constrained_first"]


def to_request(row: pd.Series | tuple) -> Request:
    """Convert a ``room_requests.csv`` row to a :class:`Request`."""
    r = row if isinstance(row, pd.Series) else pd.Series(row._asdict())
    return Request(
        date=pd.Timestamp(r["request_date"]),
        start_hour=int(r["start_hour"]),
        duration_hours=int(r["duration_hours"]),
        required_capacity=int(r["required_capacity"]),
        projector=bool(r["projector_required"]),
        computers=bool(r["computers_required"]),
    )


@dataclass
class ReplayResult:
    policy: str
    order: str
    assignments: pd.DataFrame  # one row per request: request_id, room_id (or None), wasted_seats

    def summary(self, requests: pd.DataFrame) -> dict:
        df = self.assignments.merge(requests[["request_id", "status"]], on="request_id")
        placed = df["room_id"].notna()
        unmet = df["status"].isin(["Waitlisted", "Rejected"])
        return {
            "policy": self.policy,
            "order": self.order,
            "requests": int(len(df)),
            "placed": int(placed.sum()),
            "placement_rate": float(placed.mean()),
            "unmet_requests": int(unmet.sum()),
            "unmet_placed": int((placed & unmet).sum()),
            "unmet_placement_rate": float((placed & unmet).sum() / max(unmet.sum(), 1)),
            "mean_wasted_seats": float(df.loc[placed, "wasted_seats"].mean()),
        }


def replay(
    requests: pd.DataFrame,
    rooms: pd.DataFrame,
    occ: Occupancy,
    policy: Policy = "best_fit",
    order: Order = "chronological",
    weights: Weights | None = None,
    include_cancelled: bool = False,
    seed: int = RANDOM_STATE,
) -> ReplayResult:
    """Simulate allocation of ``requests``; placements block rooms for later ones."""
    weights = weights or Weights()
    occ = occ.copy()
    rng = np.random.default_rng(seed)
    reqs = requests if include_cancelled else requests[requests["status"] != "Cancelled"]
    reqs = reqs.sort_values(["request_date", "start_hour", "request_id"]).reset_index(drop=True)

    if order == "constrained_first":
        n_static = np.array(
            [feasible_static(rooms, to_request(r)).sum() for r in reqs.itertuples()]
        )
        reqs = (
            reqs.assign(_n=n_static)
            .sort_values(["request_date", "_n", "duration_hours", "request_id"],
                         ascending=[True, True, False, True])
            .reset_index(drop=True)
        )

    caps = rooms["capacity"].to_numpy()
    ids = rooms["room_id"].to_numpy()
    out = []
    for r in reqs.itertuples():
        req = to_request(r)
        mask = feasible_mask(rooms, occ, req)
        if not mask.any():
            out.append((r.request_id, None, np.nan))
            continue
        idx = np.flatnonzero(mask)
        if policy == "first_fit":
            pick = idx[0]
        elif policy == "random":
            pick = int(rng.choice(idx))
        else:
            scores = score_rooms(rooms, req, weights)[idx]
            pick = idx[int(np.argmax(scores))]  # argmax = first max => room_id order
        occ.book(ids[pick], occ.slot(req.date, req.start_hour), req.duration_hours)
        out.append((r.request_id, ids[pick], int(caps[pick] - req.required_capacity)))
    df = pd.DataFrame(out, columns=["request_id", "room_id", "wasted_seats"])
    return ReplayResult(policy, order, df)


def feasible_static(rooms: pd.DataFrame, req: Request) -> np.ndarray:
    """Capacity/facility feasibility only (ignores time); used for ordering."""
    mask = rooms["capacity"].to_numpy() >= req.required_capacity
    if req.projector:
        mask &= rooms["projector"].to_numpy() == 1
    if req.computers:
        mask &= rooms["computers"].to_numpy() == 1
    return mask


def static_feasibility_bound(
    requests: pd.DataFrame, rooms: pd.DataFrame, occ: Occupancy
) -> pd.DataFrame:
    """Per request: how many rooms are feasible given only *existing* occupancy.

    This ignores contention between requests, so it is an upper bound on what
    any allocator can serve.
    """
    rows = []
    for r in requests.itertuples():
        req = to_request(r)
        rows.append((r.request_id, int(feasible_mask(rooms, occ, req).sum())))
    return pd.DataFrame(rows, columns=["request_id", "n_feasible_rooms"])
