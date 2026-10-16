"""Small synthetic campus used by the unit tests (independent of the real CSVs)."""

import pandas as pd
import pytest

from roomfinder.occupancy import build_occupancy


@pytest.fixture()
def rooms() -> pd.DataFrame:
    return pd.DataFrame({
        "room_id": ["R1", "R2", "R3"],
        "building": ["A", "A", "B"],
        "floor": [0, 1, 2],
        "room_type": ["Classroom", "Lab", "Seminar"],
        "capacity": [30, 60, 120],
        "projector": [1, 0, 1],
        "computers": [0, 1, 1],
        "accessible": [1, 1, 0],
        "power_outlets": [5, 10, 20],
    })


@pytest.fixture()
def schedule() -> pd.DataFrame:
    # 2025-09-01 is a Monday. R1 is busy Mon 09-11; R3 has a double booking Tue 10-12.
    return pd.DataFrame({
        "schedule_id": ["S1", "S2", "S3"],
        "room_id": ["R1", "R3", "R3"],
        "course_code": ["C1", "C2", "C3"],
        "weekday": ["Mon", "Tue", "Tue"],
        "start_hour": [9, 10, 11],
        "duration_hours": [2, 2, 1],
        "expected_attendance": [20, 100, 25],
        "department": ["CSE"] * 3,
    })


@pytest.fixture()
def events() -> pd.DataFrame:
    # Event late on Mon 1 Sep crossing midnight in R2: 23:00 for 3 hours.
    return pd.DataFrame({
        "event_id": ["E1"],
        "room_id": ["R2"],
        "event_type": ["Exam"],
        "start_time": [pd.Timestamp("2025-09-01 23:00")],
        "duration_hours": [3],
        "expected_attendance": [40],
        "actual_attendance": [35],
    })


@pytest.fixture()
def occ(rooms, schedule, events):
    return build_occupancy(rooms, schedule, events, "2025-09-01", "2025-09-14")
