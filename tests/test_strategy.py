import numpy as np
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


def test_compound_offsets_separate_pace_from_driver_and_wear():
    rows = []
    for driver, pace, stop in (("AAA", 0.0, 9), ("BBB", 0.4, 12), ("CCC", 0.8, 15)):
        stints = (("SOFT", 0.0, range(1, stop + 1)), ("HARD", 0.6, range(stop + 1, 31)))
        for compound, offset, laps in stints:
            for age, number in enumerate(laps, start=1):
                seconds = 90 + pace + offset + 0.05 * age + 0.055 * (30 - number)
                rows.append(lap(driver, number, seconds, Compound=compound, TyreLife=float(age)))

    offsets = strategy.compound_offsets(pd.DataFrame(rows), 30, ["SOFT", "MEDIUM", "HARD"])

    assert set(offsets) == {"SOFT", "HARD"}
    assert offsets["HARD"] == pytest.approx(0.6, abs=0.05)


def curves() -> pd.DataFrame:
    ages = np.arange(2.0, 41.0)
    return pd.concat(
        [
            pd.DataFrame(
                {"Compound": "SOFT", "TyreLife": ages[:20], "Predicted": 0.1 * (ages[:20] - 2)}
            ),
            pd.DataFrame({"Compound": "HARD", "TyreLife": ages, "Predicted": 0.02 * (ages - 2)}),
        ]
    )


def test_plans_rank_strategies_and_drop_mirror_images():
    plans = strategy.plans(curves(), {"SOFT": 0.0, "HARD": 0.5}, total_laps=40, pit_loss=20.0)

    assert plans[0].compounds == ("SOFT", "HARD")
    assert len(plans[0].pit_laps) == 1
    stints = [
        tuple(sorted(zip(plan.compounds, np.diff((0, *plan.pit_laps, 40)), strict=True)))
        for plan in plans
    ]
    assert len(stints) == len(set(stints))
    assert plans == sorted(plans, key=lambda plan: plan.time)


def test_plan_time_matches_the_best_plan():
    plans = strategy.plans(curves(), {"SOFT": 0.0, "HARD": 0.5}, total_laps=40, pit_loss=20.0)
    best = plans[0]

    time = strategy.plan_time(
        (best.compounds, best.pit_laps), curves(), {"SOFT": 0.0, "HARD": 0.5}, 40, 20.0
    )

    assert time == pytest.approx(best.time)
