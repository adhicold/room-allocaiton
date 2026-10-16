import pandas as pd
import pytest

from roomfinder.data import validate


def test_validate_rejects_missing_columns(rooms):
    with pytest.raises(ValueError, match="missing columns"):
        validate("rooms", rooms.drop(columns=["capacity"]))


def test_validate_rejects_nulls(rooms):
    bad = rooms.copy()
    bad.loc[0, "capacity"] = None
    with pytest.raises(ValueError, match="null"):
        validate("rooms", bad)


def test_validate_accepts_good_table(rooms):
    validate("rooms", rooms)
    assert isinstance(rooms, pd.DataFrame)
