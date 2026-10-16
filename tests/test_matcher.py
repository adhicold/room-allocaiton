import pandas as pd
import pytest

from roomfinder.config import Weights
from roomfinder.matcher import Request, feasible_mask, recommend, suggest_alternatives

MON = pd.Timestamp("2025-09-01")


def test_capacity_is_hard_constraint(rooms, occ):
    mask = feasible_mask(rooms, occ, Request(MON, 14, 1, 50))
    assert list(rooms["room_id"][mask]) == ["R2", "R3"]


def test_facilities_are_hard_constraints(rooms, occ):
    mask = feasible_mask(rooms, occ, Request(MON, 14, 1, 10, projector=True, computers=True))
    assert list(rooms["room_id"][mask]) == ["R3"]


def test_accessibility_filter(rooms, occ):
    mask = feasible_mask(rooms, occ, Request(MON, 14, 1, 10, accessible=True))
    assert "R3" not in set(rooms["room_id"][mask])


def test_time_conflict_excludes_room(rooms, occ):
    mask = feasible_mask(rooms, occ, Request(MON, 10, 1, 10))  # R1 busy 09-11
    assert "R1" not in set(rooms["room_id"][mask])


def test_partial_overlap_excludes_room(rooms, occ):
    mask = feasible_mask(rooms, occ, Request(MON, 8, 2, 10))  # 08-10 overlaps 09-11
    assert "R1" not in set(rooms["room_id"][mask])


def test_adjacent_slot_is_free(rooms, occ):
    mask = feasible_mask(rooms, occ, Request(MON, 11, 1, 10))
    assert "R1" in set(rooms["room_id"][mask])


def test_best_fit_wins_and_is_explained(rooms, occ):
    recs = recommend(rooms, occ, Request(MON, 14, 1, 25))
    assert recs[0].room_id == "R1"  # smallest room that fits
    assert any("Fits 25 of 30" in r for r in recs[0].reasons)
    assert any("no clash" in r for r in recs[0].reasons)


def test_ranking_is_deterministic(rooms, occ):
    a = recommend(rooms, occ, Request(MON, 14, 1, 10))
    b = recommend(rooms, occ, Request(MON, 14, 1, 10))
    assert [r.room_id for r in a] == [r.room_id for r in b]


def test_no_projector_waste(rooms, occ):
    # R1 (30, projector) vs R2 (60, computers): without facility needs, penalty hits both,
    # smaller R1 should still win on fit.
    assert recommend(rooms, occ, Request(MON, 14, 1, 25))[0].room_id == "R1"


def test_preferred_building_bonus(rooms, occ):
    w = Weights(capacity_fit=0.1, facility_waste=0.0, building=1.0, accessibility=0.0)
    recs = recommend(rooms, occ, Request(MON, 14, 1, 10, preferred_building="B"), w)
    assert recs[0].room_id == "R3"


def test_returns_empty_when_nothing_fits(rooms, occ):
    assert recommend(rooms, occ, Request(MON, 14, 1, 500)) == []


def test_alternatives_shift_start_when_slot_taken(rooms, occ):
    # only R1 can host a projector + 25 people on Monday 09-11 -> shift to 11:00
    req = Request(MON, 9, 2, 25, projector=True)
    full = recommend(rooms, occ, req)
    assert all(r.room_id != "R1" for r in full)  # R1 busy
    assert full and full[0].room_id == "R3"
    tight = Request(MON, 9, 2, 20, projector=True, accessible=True)
    assert recommend(rooms, occ, tight) == []
    alts = suggest_alternatives(rooms, occ, tight)
    assert alts and alts[0]["room"].room_id == "R1"


@pytest.mark.parametrize("kwargs", [{"duration_hours": 0}, {"required_capacity": 0}])
def test_invalid_requests_rejected(kwargs):
    base = {"date": MON, "start_hour": 9, "duration_hours": 1, "required_capacity": 10}
    with pytest.raises(ValueError):
        Request(**{**base, **kwargs})
