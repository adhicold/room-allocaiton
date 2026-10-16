"""Hard-constraint filtering and explainable ranking of rooms for a request."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import OPEN_END, OPEN_START, Weights
from .occupancy import Occupancy


@dataclass(frozen=True)
class Request:
    """A room request. ``date`` and ``start_hour`` define the interval start."""

    date: pd.Timestamp
    start_hour: int
    duration_hours: int
    required_capacity: int
    projector: bool = False
    computers: bool = False
    accessible: bool = False
    preferred_building: str | None = None

    def __post_init__(self) -> None:
        if self.duration_hours < 1:
            raise ValueError("duration_hours must be >= 1")
        if self.required_capacity < 1:
            raise ValueError("required_capacity must be >= 1")
        if not 0 <= self.start_hour <= 23:
            raise ValueError("start_hour must be in 0..23")


@dataclass
class Recommendation:
    room_id: str
    score: float
    capacity: int
    building: str
    reasons: list[str] = field(default_factory=list)


def feasible_mask(rooms: pd.DataFrame, occ: Occupancy, req: Request) -> np.ndarray:
    """Boolean mask of rooms satisfying every HARD constraint.

    Hard constraints: capacity, requested facilities, accessibility and being
    free for the entire requested interval.
    """
    mask = (rooms["capacity"].to_numpy() >= req.required_capacity)
    if req.projector:
        mask &= rooms["projector"].to_numpy() == 1
    if req.computers:
        mask &= rooms["computers"].to_numpy() == 1
    if req.accessible:
        mask &= rooms["accessible"].to_numpy() == 1
    slot = occ.slot(req.date, req.start_hour)
    return mask & occ.free_mask(slot, req.duration_hours)


def score_rooms(rooms: pd.DataFrame, req: Request, w: Weights) -> np.ndarray:
    """Score in [0, 1] for every room (higher is better); ignores feasibility."""
    w = w.normalised()
    cap = rooms["capacity"].to_numpy(dtype=float)
    fit = np.clip(req.required_capacity / cap, 0.0, 1.0)  # 1.0 = perfectly sized

    extra = (
        (rooms["projector"].to_numpy() == 1) & (not req.projector)
    ).astype(float) + (
        (rooms["computers"].to_numpy() == 1) & (not req.computers)
    ).astype(float)
    facility_efficiency = 1.0 - extra / 2.0  # do not waste scarce equipment

    building = (
        (rooms["building"].to_numpy() == req.preferred_building).astype(float)
        if req.preferred_building
        else np.zeros(len(rooms))
    )
    access = (
        rooms["accessible"].to_numpy().astype(float) if not req.accessible else np.zeros(len(rooms))
    )
    return (
        w.capacity_fit * fit
        + w.facility_waste * facility_efficiency
        + w.building * building
        + w.accessibility * access
    )


def explain(room: pd.Series, req: Request, free_note: str) -> list[str]:
    """Human-readable evidence for a recommendation."""
    reasons = [
        f"Fits {req.required_capacity} of {int(room['capacity'])} seats "
        f"({req.required_capacity / room['capacity']:.0%} full)",
    ]
    if req.projector:
        reasons.append("Projector available")
    if req.computers:
        reasons.append("Computers available")
    if req.accessible:
        reasons.append("Step-free accessible room")
    reasons.append(free_note)
    if req.preferred_building and room["building"] == req.preferred_building:
        reasons.append(f"In preferred building ({room['building']})")
    return reasons


def recommend(
    rooms: pd.DataFrame,
    occ: Occupancy,
    req: Request,
    weights: Weights | None = None,
    top_k: int = 5,
) -> list[Recommendation]:
    """Return up to ``top_k`` feasible rooms, best first, with explanations."""
    weights = weights or Weights()
    mask = feasible_mask(rooms, occ, req)
    if not mask.any():
        return []
    scores = score_rooms(rooms, req, weights)
    idx = np.flatnonzero(mask)
    # deterministic order: score desc, then room_id asc
    order = sorted(idx, key=lambda i: (-scores[i], rooms["room_id"].iat[i]))[:top_k]
    end_hour = req.start_hour + req.duration_hours
    note = f"Free {req.start_hour:02d}:00-{end_hour:02d}:00 on {req.date:%a %d %b}, no clash"
    return [
        Recommendation(
            room_id=rooms["room_id"].iat[i],
            score=float(scores[i]),
            capacity=int(rooms["capacity"].iat[i]),
            building=str(rooms["building"].iat[i]),
            reasons=explain(rooms.iloc[i], req, note),
        )
        for i in order
    ]


def suggest_alternatives(
    rooms: pd.DataFrame,
    occ: Occupancy,
    req: Request,
    weights: Weights | None = None,
    max_shift: int = 4,
) -> list[dict]:
    """When nothing is feasible, offer the nearest start times that work.

    Tries shifting the start by +-1..``max_shift`` hours on the same date inside
    opening hours; closest shift wins, earlier start breaks ties.
    """
    out: list[dict] = []
    for shift in sorted(range(-max_shift, max_shift + 1), key=lambda s: (abs(s), s)):
        if shift == 0:
            continue
        hour = req.start_hour + shift
        if hour < OPEN_START or hour > OPEN_END:
            continue
        alt = Request(**{**req.__dict__, "start_hour": hour})
        recs = recommend(rooms, occ, alt, weights, top_k=1)
        if recs:
            out.append({"start_hour": hour, "shift_hours": shift, "room": recs[0]})
    return out[:3]
