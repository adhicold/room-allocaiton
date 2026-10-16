"""Utilisation statistics, room clustering and timetable repair suggestions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, TEACHING_HOURS, UNDERUSED_THRESHOLD, WEEKDAYS
from .occupancy import weekly_grid

TEACH_DAYS = 6  # Mon-Sat carry classes in the data
_H0, _H1 = TEACHING_HOURS.start, TEACHING_HOURS.stop


def room_utilisation(rooms: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Share of the 60 weekly teaching slots each room is occupied (clashes count once)."""
    occupied = grid[:, :TEACH_DAYS, _H0:_H1] > 0
    out = rooms[["room_id", "building", "room_type", "capacity"]].copy()
    out["utilisation"] = occupied.mean(axis=(1, 2))
    out["double_booked_slots"] = (grid[:, :TEACH_DAYS, _H0:_H1] > 1).sum(axis=(1, 2))
    out["underused"] = out["utilisation"] < UNDERUSED_THRESHOLD
    return out


def campus_heatmap(grid: np.ndarray) -> pd.DataFrame:
    """Mean occupancy (0..1) by weekday x hour over all rooms."""
    data = (grid[:, :TEACH_DAYS, _H0:_H1] > 0).mean(axis=0)
    return pd.DataFrame(data, index=list(WEEKDAYS[:TEACH_DAYS]), columns=list(TEACHING_HOURS))


def cluster_rooms(
    util: pd.DataFrame, grid: np.ndarray, k_range: range = range(2, 6)
) -> tuple[pd.DataFrame, dict]:
    """K-Means on each room's weekday utilisation profile; k chosen by silhouette."""
    per_day = (grid[:, :TEACH_DAYS, _H0:_H1] > 0).mean(axis=2)
    feats = StandardScaler().fit_transform(np.column_stack([per_day, util["capacity"].to_numpy()]))
    best = None
    for k in k_range:
        labels = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(feats)
        score = float(silhouette_score(feats, labels))
        if best is None or score > best[0]:
            best = (score, k, labels)
    score, k, labels = best
    out = util.assign(cluster=labels)
    return out, {"k": int(k), "silhouette": score}


def plan_moves(rooms: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Greedy timetable repair: fix clashes, then over-crowded, then oversized classes.

    A move is only proposed if the target room is free for the whole slot on that
    weekday and seats the expected attendance; it never creates a new clash.
    Events are dated and are ignored here (this is a weekly-plan repair).
    """
    row = {rid: i for i, rid in enumerate(rooms["room_id"])}
    ids = rooms["room_id"].to_numpy()
    caps = rooms["capacity"].to_numpy()
    day = {n: i for i, n in enumerate(WEEKDAYS)}
    grid = weekly_grid(rooms, schedule)
    cur = {r.schedule_id: r.room_id for r in schedule.itertuples()}
    moves: list[dict] = []

    def free_rooms(d: int, h0: int, h1: int) -> np.ndarray:
        return ~(grid[:, d, h0:h1] > 0).any(axis=1)

    def try_move(r, reason: str, require_smaller: bool) -> None:
        d, h0, h1 = day[r.weekday], r.start_hour, r.start_hour + r.duration_hours
        i_old = row[cur[r.schedule_id]]
        grid[i_old, d, h0:h1] -= 1  # lift the class off its room
        ok = free_rooms(d, h0, h1) & (caps >= r.expected_attendance)
        if require_smaller:
            ok &= caps < caps[i_old]
        ok[i_old] = False
        if ok.any():
            idx = np.flatnonzero(ok)
            new = idx[int(np.argmin(caps[idx]))]  # smallest room that fits
            grid[new, d, h0:h1] += 1
            cur[r.schedule_id] = ids[new]
            moves.append({"schedule_id": r.schedule_id, "course_code": r.course_code,
                          "weekday": r.weekday, "start_hour": r.start_hour,
                          "from_room": ids[i_old], "to_room": ids[new], "reason": reason,
                          "attendance": r.expected_attendance,
                          "from_capacity": int(caps[i_old]), "to_capacity": int(caps[new])})
        else:
            grid[i_old, d, h0:h1] += 1  # could not move: put it back

    ordered = schedule.sort_values("schedule_id")
    before = _snapshot(rooms, schedule, cur, grid)
    for r in ordered.itertuples():  # 1) clashes: later rows of a double booking move
        d, h0, h1 = day[r.weekday], r.start_hour, r.start_hour + r.duration_hours
        if (grid[row[cur[r.schedule_id]], d, h0:h1] > 1).any():
            try_move(r, "clash", False)
    for r in ordered.itertuples():  # 2) over-crowded
        if r.expected_attendance > caps[row[cur[r.schedule_id]]]:
            try_move(r, "overcrowded", False)
    for r in ordered.itertuples():  # 3) oversized (<25 % full) -> smaller room that still fits
        if r.expected_attendance < UNDERUSED_THRESHOLD * caps[row[cur[r.schedule_id]]]:
            try_move(r, "oversized", True)
    after = _snapshot(rooms, schedule, cur, grid)
    return pd.DataFrame(moves), {"before": before, "after": after, "moves": len(moves)}


def _snapshot(rooms, schedule, cur, grid) -> dict:
    caps = dict(zip(rooms["room_id"], rooms["capacity"], strict=True))
    room_now = schedule["schedule_id"].map(cur)
    cap_now = room_now.map(caps)
    att = schedule["expected_attendance"]
    return {
        "double_booked_slots": int((grid > 1).sum()),
        "overcrowded_classes": int((att > cap_now).sum()),
        "oversized_classes": int((att < UNDERUSED_THRESHOLD * cap_now).sum()),
        "mean_seat_utilisation": float((att / cap_now).clip(upper=1).mean()),
    }
