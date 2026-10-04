import pandas as pd
import pytest

from src import strategy


def lap(driver: str, number: int, seconds: float, **overrides) -> dict:
    row = {
        "Driver": driver,
        "Team": f"Team {driver}",
        "LapNumber": float(number),
        "LapTimeSeconds": seconds,
        "Compound": "MEDIUM",
        "TyreLife": float(number),
        "Stint": 1.0,
        "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT,
        "TrackStatus": "1",
        "IsAccurate": True,
        "Time": pd.Timedelta(seconds=90 * number),
    }
    return row | overrides


def race_with_stops() -> pd.DataFrame:
    rows = []
    for driver, stop, offset in (("AAA", 10, 0.0), ("BBB", 12, 1.0)):
        for number in range(1, 21):
            extra = {"Time": pd.Timedelta(seconds=90 * number + offset)}
            if number == stop:
                extra |= {
                    "PitInTime": pd.Timedelta(seconds=90 * number - 5),
                    "LapTimeSeconds": 101.0,
                }
            if number == stop + 1:
                extra |= {
                    "PitOutTime": pd.Timedelta(seconds=90 * number - 70),
                    "LapTimeSeconds": 104.0,
                    "Compound": "HARD",
                    "Stint": 2.0,
                }
            if number > stop + 1:
                extra |= {"Compound": "HARD", "Stint": 2.0}
            if number > stop:
                extra["Time"] = pd.Timedelta(
                    seconds=90 * number + offset + (2 if driver == "BBB" else 0)
                )
            rows.append(lap(driver, number, 90.0, **extra))
    return pd.DataFrame(rows)


def test_pit_stops_measure_pit_lane_time_and_time_lost():
    stops = strategy.pit_stops(race_with_stops())

    assert stops[["Driver", "Lap"]].values.tolist() == [["AAA", 10], ["BBB", 12]]
    assert stops["PitLane"].tolist() == pytest.approx([25.0, 25.0])
    assert stops["TimeLost"].tolist() == pytest.approx([25.0, 25.0])
    assert stops[["From", "To"]].values.tolist() == [["MEDIUM", "HARD"]] * 2
    assert strategy.pit_loss(stops) == pytest.approx(25.0)


def test_undercut_works_when_pitting_first_gets_the_driver_ahead():
    rows = []
    for number in range(1, 16):
        aaa = 90 * number + (22 if number >= 13 else 0)
        bbb = 90 * number + (1 if number <= 10 else 16)
        rows += [
            {"Driver": "AAA", "LapNumber": float(number), "Time": pd.Timedelta(seconds=aaa)},
            {"Driver": "BBB", "LapNumber": float(number), "Time": pd.Timedelta(seconds=bbb)},
        ]
    stops = pd.DataFrame({"Driver": ["BBB", "AAA"], "Lap": [10, 12]})

    attempts = strategy.undercuts(pd.DataFrame(rows), stops)

    assert attempts[["Lap", "Attacker", "Defender", "Worked"]].values.tolist() == [
        [10, "BBB", "AAA", True]
    ]
    assert attempts["Gap"].tolist() == pytest.approx([1.0])
