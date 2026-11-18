import math

import numpy as np
import pandas as pd
import pytest

from src import ideal_lap
from views.results_tab import computer_text


def sector_laps() -> pd.DataFrame:
    rows = [
        ("AAA", 30.0, 31.0, 29.5, False),
        ("BBB", 29.9, 31.2, 29.9, False),
        # The quickest first sector, on a lap the stewards deleted.
        ("CCC", 29.0, 32.0, 30.0, True),
    ]
    return pd.DataFrame(
        [
            {
                "Driver": driver,
                "Team": f"Team {driver}",
                "LapNumber": 1.0,
                "Sector1": s1,
                "Sector2": s2,
                "Sector3": s3,
                "LapTimeSeconds": s1 + s2 + s3,
                "IsAccurate": True,
                "Deleted": deleted,
            }
            for driver, s1, s2, s3, deleted in rows
        ]
    )


def test_perfect_lap_joins_the_fastest_sectors_that_counted():
    perfect = ideal_lap.perfect_lap(sector_laps())

    assert perfect["Driver"].tolist() == ["BBB", "AAA", "AAA"]
    assert perfect["Time"].sum() == pytest.approx(29.9 + 31.0 + 29.5)


def test_a_circle_is_lapped_at_its_corner_speed():
    radius, speed = 100.0, 30.0
    step = 2 * math.pi * radius / 400
    limit = np.full(400, speed)
    accel = (np.array([0.0, 50.0]), np.array([10.0, 10.0]))
    brake = (np.array([0.0, 50.0]), np.array([40.0, 40.0]))

    simulated = ideal_lap.simulate(limit, step, accel, brake)

    assert simulated == pytest.approx(limit)
    assert ideal_lap.lap_seconds(simulated, step) == pytest.approx(2 * math.pi * radius / speed)


def test_the_car_brakes_in_time_for_a_slow_corner():
    limit = np.full(200, 80.0)
    limit[100] = 20.0
    accel = (np.array([0.0, 100.0]), np.array([10.0, 10.0]))
    brake = (np.array([0.0, 100.0]), np.array([40.0, 40.0]))

    speed = ideal_lap.simulate(limit, 5.0, accel, brake)

    assert speed[100] == pytest.approx(20.0)
    # Neither pass may ask for more than the car can do between two points.
    change = np.diff(speed**2) / (2 * 5.0)
    assert change.max() <= 10.0 + 1e-9
    assert change.min() >= -40.0 - 1e-9


def lap(slow_corners: list[float], length: float, points: int = 400) -> pd.DataFrame:
    distance = np.linspace(0.0, length, points)
    speed = np.full(points, 300.0)
    for corner in slow_corners:
        speed = np.minimum(speed, 80.0 + np.abs(distance - corner) * 1.5)
    return pd.DataFrame({"Distance": distance, "Speed": speed})


def test_laps_are_lined_up_at_the_slow_corners():
    typical = lap([1000.0, 3000.0], 5000.0)
    # This car's distance ran 200 m long by the second corner, then caught up.
    drifting = lap([1000.0, 3200.0], 5100.0)

    distance = ideal_lap.aligned(drifting, typical)
    apexes = ideal_lap.apexes(distance, drifting["Speed"].to_numpy())

    assert distance[apexes] == pytest.approx([1000.0, 3000.0], abs=15.0)
    assert distance[-1] == pytest.approx(5000.0)


def test_a_cars_acceleration_follows_everyones_shape_at_its_own_level():
    everyone = (np.array([10.0, 20.0, 30.0]), np.array([12.0, 8.0, 4.0]))
    own = (np.array([20.0, 30.0]), np.array([8.8, 4.4]))

    speeds, values = ideal_lap.scaled(everyone, own)

    assert speeds == pytest.approx(everyone[0])
    assert values == pytest.approx([13.2, 8.8, 4.4])


def levels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2021, 2022, 2023, 2024, 2025, 2026, 2024],
            "EventName": ["Here GP"] * 6 + ["Elsewhere GP"],
            "Length": [5000.0, 5000.0, 5010.0, 5005.0, 5000.0, 4990.0, 4000.0],
            "Wet": [False, False, True, False, False, False, False],
            "Level": [1.00, 1.02, 0.90, 1.01, 1.20, 1.015, 0.98],
            "Lap": [90.0, 91.0, 99.0, 90.5, 90.0, 90.0, 70.0],
        }
    )


def test_a_circuit_is_tuned_on_its_nearest_dry_years():
    # 2025 is the session itself and 2023 was wet, so 2024, 2026 and 2022 count.
    level = ideal_lap.track_level(levels(), "Here GP", 2025, 5000.0, simulated=89.0)

    assert level == pytest.approx(1.015)


def test_a_lap_far_from_the_circuits_usual_times_means_no_computer_lap():
    # Tuned, 95 s would be over 6% off the 90 s laps of 2022, 2024 and 2026.
    assert ideal_lap.track_level(levels(), "Here GP", 2025, 5000.0, simulated=95.0) is None


def test_years_that_disagree_mean_no_computer_lap():
    disagreeing = levels().assign(Level=[1.00, 1.08, 0.90, 1.01, 1.20, 1.015, 0.98])

    assert ideal_lap.track_level(disagreeing, "Here GP", 2025, 5000.0, simulated=89.0) is None


def test_a_new_layout_has_no_computer_lap():
    assert ideal_lap.track_level(levels(), "Here GP", 2025, 5400.0, simulated=89.0) is None
    assert ideal_lap.track_level(levels(), "Elsewhere GP", 2025, 4000.0, simulated=70.0) is None


def test_no_track_outline_means_no_computer_lap():
    simulated = ideal_lap.simulated_laps(
        sector_laps(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )

    assert simulated.empty


@pytest.mark.parametrize(
    ("gaps", "expected"),
    [
        ([0.3, -0.2, 0.5], "**BBB** beat theirs by 0.200 s."),
        ([-0.1, -0.4, 0.5], "2 drivers beat theirs, **BBB** by the most: 0.400 s."),
        ([0.3, 0.1, 0.5], "nobody beat theirs. **BBB** came closest, 0.100 s off."),
    ],
)
def test_computer_text_names_who_beat_their_cars_lap(gaps, expected):
    drivers = pd.Series(["AAA", "BBB", "CCC"])

    assert computer_text(drivers, pd.Series(gaps)).endswith(expected)
