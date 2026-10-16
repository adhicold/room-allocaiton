"""Regression tests on the real competition CSVs (skipped if they are absent)."""

from pathlib import Path

import pytest

from roomfinder.data import load_campus
from roomfinder.occupancy import class_clashes

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(not (DATA / "rooms.csv").exists(), reason="dataset not present")


def test_dataset_shapes():
    c = load_campus(DATA)
    sizes = (len(c.rooms), len(c.schedule), len(c.requests), len(c.events))
    assert sizes == (120, 5000, 5000, 1800)


def test_known_clash_from_dataset_is_detected():
    c = load_campus(DATA)
    clashes = class_clashes(c.schedule)
    hit = clashes[(clashes["room_id"] == "R003") & (clashes["weekday"] == "Tue")]
    assert not hit.empty
