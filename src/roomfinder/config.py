"""Central constants so that no magic numbers live in the logic modules."""

from __future__ import annotations

from dataclasses import dataclass

WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
HOURS_PER_DAY = 24
OPEN_START = 8  # first bookable hour
OPEN_END = 20  # last bookable start hour used when suggesting alternatives
TEACHING_HOURS = range(8, 18)  # window used for utilisation statistics
UNDERUSED_THRESHOLD = 0.25  # room is "underused" below 25 % of teaching slots
RANDOM_STATE = 42

DATA_FILES = {
    "rooms": "rooms.csv",
    "schedule": "class_schedule.csv",
    "requests": "room_requests.csv",
    "events": "events.csv",
}


@dataclass(frozen=True)
class Weights:
    """Scoring weights for ranking feasible rooms (all non-negative)."""

    capacity_fit: float = 0.55  # prefer rooms just big enough
    facility_waste: float = 0.25  # do not burn projector/computer rooms needlessly
    building: float = 0.10  # preferred building bonus
    accessibility: float = 0.10  # accessible room bonus when it is not required

    def normalised(self) -> Weights:
        total = self.capacity_fit + self.facility_waste + self.building + self.accessibility
        if total <= 0:
            raise ValueError("At least one weight must be positive")
        return Weights(
            self.capacity_fit / total,
            self.facility_waste / total,
            self.building / total,
            self.accessibility / total,
        )
