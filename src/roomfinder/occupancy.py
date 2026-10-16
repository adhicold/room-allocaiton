"""Occupancy model: one integer counter per (room, hour) over the whole term.

Time is a flat hourly axis (``day_index * 24 + hour``) so intervals that cross
midnight need no special handling. A cell value of 0 means free, 1 booked and
>1 a double booking, which is how clashes are detected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HOURS_PER_DAY, WEEKDAYS


@dataclass
class Occupancy:
    """Hourly booking counters for every room across a date range."""

    room_ids: list[str]
    start_date: pd.Timestamp
    n_days: int
    grid: np.ndarray  # shape (n_rooms, n_days * 24), dtype int16

    def __post_init__(self) -> None:
        self._row = {rid: i for i, rid in enumerate(self.room_ids)}

    # ----------------------------------------------------------------- helpers
    def room_row(self, room_id: str) -> int:
        return self._row[room_id]

    def slot(self, date: pd.Timestamp, hour: int) -> int:
        """Flat hour index of ``date`` at ``hour`` (hour may exceed 23)."""
        day = int((pd.Timestamp(date).normalize() - self.start_date).days)
        if not 0 <= day < self.n_days:
            raise ValueError(f"{date} outside modelled term")
        return day * HOURS_PER_DAY + hour

    def free_mask(self, start_slot: int, duration: int) -> np.ndarray:
        """Boolean mask over rooms that are free for the whole interval."""
        end = min(start_slot + duration, self.grid.shape[1])
        return ~(self.grid[:, start_slot:end] > 0).any(axis=1)

    def book(self, room_id: str, start_slot: int, duration: int) -> None:
        self.grid[self._row[room_id], start_slot : start_slot + duration] += 1

    def copy(self) -> Occupancy:
        return Occupancy(self.room_ids, self.start_date, self.n_days, self.grid.copy())


def build_occupancy(
    rooms: pd.DataFrame,
    schedule: pd.DataFrame,
    events: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> Occupancy:
    """Project the weekly timetable and dated events onto the term calendar.

    The weekly timetable repeats every week of ``[start, end]`` (assumption:
    no holidays are given in the data).
    """
    start_date = pd.Timestamp(start).normalize()
    n_days = int((pd.Timestamp(end).normalize() - start_date).days) + 1
    room_ids = rooms["room_id"].tolist()
    row = {rid: i for i, rid in enumerate(room_ids)}
    grid = np.zeros((len(room_ids), n_days * HOURS_PER_DAY), dtype=np.int16)

    weekday_of_day = (start_date + pd.to_timedelta(np.arange(n_days), "D")).dayofweek.to_numpy()
    days_by_weekday = {name: np.flatnonzero(weekday_of_day == i) for i, name in enumerate(WEEKDAYS)}

    for r in schedule.itertuples(index=False):
        days = days_by_weekday[r.weekday]
        hours = np.arange(r.start_hour, r.start_hour + r.duration_hours)
        idx = (days[:, None] * HOURS_PER_DAY + hours[None, :]).ravel()
        grid[row[r.room_id], idx] += 1

    for e in events.itertuples(index=False):
        day = int((e.start_time.normalize() - start_date).days)
        first = day * HOURS_PER_DAY + e.start_time.hour
        lo, hi = max(first, 0), min(first + e.duration_hours, grid.shape[1])
        if lo < hi:
            grid[row[e.room_id], lo:hi] += 1
    return Occupancy(room_ids, start_date, n_days, grid)


def weekly_grid(rooms: pd.DataFrame, schedule: pd.DataFrame) -> np.ndarray:
    """Counter grid (rooms x 7 weekdays x 24 hours) of the recurring timetable."""
    row = {rid: i for i, rid in enumerate(rooms["room_id"])}
    grid = np.zeros((len(row), len(WEEKDAYS), HOURS_PER_DAY), dtype=np.int16)
    day = {name: i for i, name in enumerate(WEEKDAYS)}
    for r in schedule.itertuples(index=False):
        grid[row[r.room_id], day[r.weekday], r.start_hour : r.start_hour + r.duration_hours] += 1
    return grid


def class_clashes(schedule: pd.DataFrame) -> pd.DataFrame:
    """Pairs of timetable rows in the same room that overlap in time."""
    s = schedule.assign(end=schedule["start_hour"] + schedule["duration_hours"])
    pairs = s.merge(s, on=["room_id", "weekday"], suffixes=("_a", "_b"))
    mask = (
        (pairs["schedule_id_a"] < pairs["schedule_id_b"])
        & (pairs["start_hour_a"] < pairs["end_b"])
        & (pairs["start_hour_b"] < pairs["end_a"])
    )
    cols = ["room_id", "weekday", "schedule_id_a", "schedule_id_b",
            "course_code_a", "course_code_b", "start_hour_a", "start_hour_b"]
    return pairs.loc[mask, cols].reset_index(drop=True)


def event_clashes(events: pd.DataFrame) -> pd.DataFrame:
    """Pairs of dated events in the same room that overlap in time."""
    e = events.assign(end=events["start_time"] + pd.to_timedelta(events["duration_hours"], "h"))
    pairs = e.merge(e, on="room_id", suffixes=("_a", "_b"))
    mask = (
        (pairs["event_id_a"] < pairs["event_id_b"])
        & (pairs["start_time_a"] < pairs["end_b"])
        & (pairs["start_time_b"] < pairs["end_a"])
    )
    cols = ["room_id", "event_id_a", "event_id_b", "start_time_a", "start_time_b"]
    return pairs.loc[mask, cols].reset_index(drop=True)


def class_event_clashes(
    schedule: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Dated events that land on top of the recurring timetable."""
    rows = []
    sched = schedule.assign(end=schedule["start_hour"] + schedule["duration_hours"])
    by_room = {k: v for k, v in sched.groupby("room_id")}
    for e in events.itertuples(index=False):
        s = by_room.get(e.room_id)
        if s is None:
            continue
        wd = WEEKDAYS[e.start_time.dayofweek]
        h0 = e.start_time.hour
        hit = s[(s["weekday"] == wd) & (s["start_hour"] < h0 + e.duration_hours) & (s["end"] > h0)]
        for c in hit.itertuples(index=False):
            rows.append((e.room_id, e.event_id, c.schedule_id, e.start_time))
    return pd.DataFrame(rows, columns=["room_id", "event_id", "schedule_id", "start_time"])
