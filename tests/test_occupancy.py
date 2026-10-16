import pandas as pd

from roomfinder.occupancy import class_clashes, event_clashes, weekly_grid


def test_weekly_class_repeats_every_week(occ):
    for date in ("2025-09-01", "2025-09-08"):  # two Mondays
        slot = occ.slot(pd.Timestamp(date), 9)
        assert occ.grid[occ.room_row("R1"), slot] == 1
        assert occ.grid[occ.room_row("R1"), slot + 2] == 0  # class ends at 11:00


def test_event_crossing_midnight_blocks_next_day(occ):
    row = occ.room_row("R2")
    assert occ.grid[row, occ.slot(pd.Timestamp("2025-09-01"), 23)] == 1
    assert occ.grid[row, occ.slot(pd.Timestamp("2025-09-02"), 0)] == 1
    assert occ.grid[row, occ.slot(pd.Timestamp("2025-09-02"), 2)] == 0


def test_class_clash_detected_with_half_open_intervals(schedule):
    clashes = class_clashes(schedule)
    assert len(clashes) == 1  # S2 10-12 overlaps S3 11-12 in R3
    assert set(clashes.iloc[0][["schedule_id_a", "schedule_id_b"]]) == {"S2", "S3"}


def test_back_to_back_classes_do_not_clash(schedule):
    s = schedule.copy()
    s.loc[s["schedule_id"] == "S3", "start_hour"] = 12  # starts when S2 ends
    assert class_clashes(s).empty


def test_event_clashes(events):
    twin = events.assign(event_id="E2", start_time=pd.Timestamp("2025-09-02 00:00"))
    assert len(event_clashes(pd.concat([events, twin]))) == 1


def test_weekly_grid_counts_double_booking(rooms, schedule):
    grid = weekly_grid(rooms, schedule)
    assert grid.max() == 2  # R3 Tuesday 11:00
