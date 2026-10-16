import pandas as pd

from roomfinder.allocation import replay


def _requests(rows):
    return pd.DataFrame(rows, columns=[
        "request_id", "request_date", "start_hour", "duration_hours", "required_capacity",
        "projector_required", "computers_required", "purpose", "status"])


def test_placement_blocks_room_for_later_requests(rooms, occ):
    day = pd.Timestamp("2025-09-03")  # Wednesday: nothing scheduled
    reqs = _requests([
        ("Q1", day, 10, 1, 100, 0, 0, "Class", "Approved"),
        ("Q2", day, 10, 1, 100, 0, 0, "Class", "Approved"),  # only R3 seats 100 -> taken
    ])
    res = replay(reqs, rooms, occ, "best_fit").assignments
    assert res.loc[res["request_id"] == "Q1", "room_id"].iloc[0] == "R3"
    assert pd.isna(res.loc[res["request_id"] == "Q2", "room_id"].iloc[0])


def test_cancelled_requests_are_excluded(rooms, occ):
    day = pd.Timestamp("2025-09-03")
    reqs = _requests([("Q1", day, 10, 1, 10, 0, 0, "Class", "Cancelled")])
    assert replay(reqs, rooms, occ).assignments.empty


def test_best_fit_wastes_fewer_seats_than_first_fit(rooms, occ):
    day = pd.Timestamp("2025-09-03")
    reqs = _requests([("Q1", day, 10, 1, 55, 0, 0, "Class", "Approved")])
    best = replay(reqs, rooms, occ, "best_fit").assignments["wasted_seats"].iloc[0]
    assert best == 5  # R2 (60 seats)


def test_replay_does_not_mutate_input_occupancy(rooms, occ):
    before = occ.grid.copy()
    day = pd.Timestamp("2025-09-03")
    replay(_requests([("Q1", day, 10, 1, 10, 0, 0, "Class", "Approved")]), rooms, occ)
    assert (occ.grid == before).all()
