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


def test_rejoin_places_a_stop_among_the_gaps_at_that_lap():
    laps = race_with_stops()
    stops = strategy.pit_stops(laps)

    result = strategy.rejoin(laps, stops, "AAA", 5)

    assert result.loss == pytest.approx(25.0)
    assert (result.position_before, result.position_after) == (1, 2)
    assert result.ahead[0] == "BBB"
    assert result.ahead[1] == pytest.approx(24.0)
    assert result.behind is None
    assert result.actual_position is None


def test_rejoin_reports_where_a_real_stop_came_out():
    laps = race_with_stops()
    result = strategy.rejoin(laps, strategy.pit_stops(laps), "AAA", 10)

    # The made-up laps don't add the stop to AAA's race time, so it stays in front.
    assert result.actual_position == 1


def test_rejoin_needs_a_timed_lap_and_a_pit_loss():
    laps = race_with_stops()
    stops = strategy.pit_stops(laps)

    assert strategy.rejoin(laps, stops, "AAA", 99) is None
    assert strategy.rejoin(laps, stops.iloc[0:0], "AAA", 5) is None


def test_rejoin_figure_draws_the_ring_and_the_arc():
    laps = race_with_stops()
    result = strategy.rejoin(laps, strategy.pit_stops(laps), "AAA", 5)

    styles = {"AAA": {"color": "#E10600"}, "BBB": {"color": "#3671C6"}}
    figure = strategy.rejoin_figure(result, styles, 90.0)

    assert figure.layout.polar.angularaxis.direction == "clockwise"
    assert len(figure.data) == 6
