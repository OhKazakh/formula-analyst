import numpy as np
import pandas as pd
import pytest

from src import sectors


def laps() -> pd.DataFrame:
    rows = [
        ("AAA", 1, 30.0, 31.0, 29.5, 320.0),
        ("AAA", 2, 30.4, 30.6, 29.8, 322.0),
        ("BBB", 1, 29.9, 31.2, 29.9, 330.0),
    ]
    return pd.DataFrame(
        [
            {
                "Driver": driver,
                "Team": f"Team {driver}",
                "LapNumber": float(number),
                "Sector1": s1,
                "Sector2": s2,
                "Sector3": s3,
                "LapTimeSeconds": s1 + s2 + s3,
                "SpeedST": trap,
                "IsAccurate": True,
            }
            for driver, number, s1, s2, s3, trap in rows
        ]
    )


def test_theoretical_best_adds_each_drivers_best_sectors():
    best = sectors.theoretical_best(laps()).set_index("Driver")

    assert best.loc["AAA", "Theoretical"] == pytest.approx(30.0 + 30.6 + 29.5)
    assert best.loc["AAA", "Gain"] == pytest.approx(90.5 - 90.1)
    assert best.loc["BBB", "Gain"] == pytest.approx(0.0)


def test_sector_leaders_pick_the_quickest_time_in_each_sector():
    leaders = sectors.sector_leaders(laps())

    assert leaders[["Sector", "Driver", "Lap"]].values.tolist() == [
        [1, "BBB", 1],
        [2, "AAA", 2],
        [3, "AAA", 1],
    ]


def test_speed_profile_uses_the_speed_trap_and_apex_speeds():
    distance = np.linspace(0.0, 1000.0, 101)
    trace = pd.DataFrame({"Distance": distance, "Speed": np.where(distance == 500, 90.0, 250.0)})
    corners = pd.DataFrame({"Number": [1], "Letter": [""], "Distance": [500.0]})

    profile = sectors.speed_profile(laps(), {"AAA": trace}, corners)

    assert profile[["Driver", "TopSpeed", "CornerSpeed"]].values.tolist() == [["AAA", 322.0, 90.0]]
