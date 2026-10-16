import numpy as np

from roomfinder.occupancy import weekly_grid
from roomfinder.utilization import campus_heatmap, plan_moves, room_utilisation


def test_utilisation_counts_slots_once(rooms, schedule):
    util = room_utilisation(rooms, weekly_grid(rooms, schedule))
    r3 = util.set_index("room_id").loc["R3"]
    assert r3["double_booked_slots"] == 1
    assert np.isclose(r3["utilisation"], 2 / 60)  # slots 10,11 occupied (clash counted once)


def test_heatmap_shape(rooms, schedule):
    assert campus_heatmap(weekly_grid(rooms, schedule)).shape == (6, 10)


def test_plan_moves_fixes_clash_without_creating_new_one(rooms, schedule):
    moves, summary = plan_moves(rooms, schedule)
    assert summary["after"]["double_booked_slots"] < summary["before"]["double_booked_slots"]
    clash_moves = moves[moves["reason"] == "clash"]
    assert len(clash_moves) == 1
    assert clash_moves.iloc[0]["to_room"] != "R3"
    assert clash_moves.iloc[0]["to_capacity"] >= clash_moves.iloc[0]["attendance"]
