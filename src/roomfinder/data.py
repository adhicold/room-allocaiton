"""Loading and validating the four campus CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_FILES

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "rooms": {"room_id", "building", "floor", "room_type", "capacity", "projector",
              "computers", "accessible", "power_outlets"},
    "schedule": {"schedule_id", "room_id", "course_code", "weekday", "start_hour",
                 "duration_hours", "expected_attendance", "department"},
    "requests": {"request_id", "request_date", "start_hour", "duration_hours",
                 "required_capacity", "projector_required", "computers_required",
                 "purpose", "status"},
    "events": {"event_id", "room_id", "event_type", "start_time", "duration_hours",
               "expected_attendance", "actual_attendance"},
}


@dataclass
class Campus:
    """All four tables, validated and typed."""

    rooms: pd.DataFrame
    schedule: pd.DataFrame
    requests: pd.DataFrame
    events: pd.DataFrame


def _find(data_dir: Path, filename: str) -> Path:
    """Locate a CSV, tolerating upload prefixes such as ``1792145_rooms.csv``."""
    exact = data_dir / filename
    if exact.exists():
        return exact
    matches = sorted(data_dir.glob(f"*{filename}"))
    if not matches:
        raise FileNotFoundError(f"{filename} not found in {data_dir}")
    return matches[0]


def validate(name: str, df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if a table is missing columns or has null keys."""
    missing = REQUIRED_COLUMNS[name] - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{name}: null values in {bad}")


def load_campus(data_dir: str | Path) -> Campus:
    """Read, validate and type all tables from ``data_dir``."""
    data_dir = Path(data_dir)
    tables = {k: pd.read_csv(_find(data_dir, f)) for k, f in DATA_FILES.items()}
    for name, df in tables.items():
        validate(name, df)

    rooms = tables["rooms"].sort_values("room_id").reset_index(drop=True)
    if rooms["room_id"].duplicated().any():
        raise ValueError("rooms: duplicate room_id")
    known = set(rooms["room_id"])

    schedule = tables["schedule"]
    events = tables["events"].copy()
    requests = tables["requests"].copy()
    for name, df in (("schedule", schedule), ("events", events)):
        unknown = set(df["room_id"]) - known
        if unknown:
            raise ValueError(f"{name}: unknown room_id {sorted(unknown)[:5]}")

    events["start_time"] = pd.to_datetime(events["start_time"])
    requests["request_date"] = pd.to_datetime(requests["request_date"])
    return Campus(rooms, schedule, requests, events)
