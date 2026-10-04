import numpy as np
import pandas as pd
import pytest

from src import timing

LAP_TIMES = {
    "AAA": [89.0] * 5,
    "BBB": [92.0, 92.0, 85.0, 92.0, 92.0],
    "CCC": [120.0] * 4,
    "DDD": [95.0, 95.0],
}


def race() -> pd.DataFrame:
    rows = []
    for driver, times in LAP_TIMES.items():
        elapsed = np.cumsum(times)
        for number, (seconds, end) in enumerate(zip(times, elapsed, strict=True), start=1):
            stint = 1 if driver != "AAA" or number <= 2 else 2
            rows.append(
                {
                    "Driver": driver,
                    "Team": f"Team {driver}",
                    "LapNumber": float(number),
                    "LapTimeSeconds": seconds,
                    "Time": pd.Timedelta(seconds=float(end)),
                    "Stint": float(stint),
                    "Compound": "SOFT" if stint == 1 else "HARD",
                    "TyreLife": float(number if stint == 1 else number - 2),
                    "PitInTime": pd.Timedelta(seconds=170)
                    if (driver, number) == ("AAA", 2)
                    else pd.NaT,
                    "PitOutTime": pd.Timedelta(seconds=200)
                    if (driver, number) == ("AAA", 3)
                    else pd.NaT,
                    "IsPersonalBest": number == 1 or (driver, number) == ("BBB", 3),
                }
            )
    return pd.DataFrame(rows)


def test_finishers_include_lapped_cars_but_not_retirements():
    assert timing.finishers(race()) == {"AAA", "BBB", "CCC"}


def test_tower_puts_retirements_last_and_counts_laps_down():
    standing = timing.tower(race(), 5).set_index("Driver")

    assert standing["Position"].to_dict() == {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}
    assert standing.loc["BBB", ["Gap", "Interval"]].tolist() == pytest.approx([8.0, 8.0])
    assert standing.loc["CCC", ["LapsDown", "IntervalLaps"]].tolist() == [1, 1]
    assert np.isnan(standing.loc["CCC", "Gap"])
    assert standing.loc["DDD", "Out"]


def test_tower_is_taken_at_the_requested_lap():
    standing = timing.tower(race(), 2).set_index("Driver")

    assert standing["Position"].to_dict() == {"AAA": 1, "BBB": 2, "DDD": 3, "CCC": 4}
    assert not standing["Out"].any()
    assert standing.loc["CCC", "Gap"] == pytest.approx(240.0 - 178.0)
    assert standing.loc["AAA", "InPit"]
    assert standing.loc["AAA", "Stints"] == [("SOFT", 2)]


def test_tower_marks_personal_and_overall_bests():
    standing = timing.tower(race(), 3).set_index("Driver")

    assert standing.loc["BBB", ["LastPersonalBest", "LastFastest", "BestFastest"]].tolist() == [
        True,
        True,
        True,
    ]
    assert standing.loc["AAA", "BestLap"] == 89.0
    assert not standing.loc["AAA", "BestFastest"]
    assert standing.loc["AAA", "Stints"] == [("SOFT", 2), ("HARD", 1)]


def test_lap_times_flag_the_best_laps():
    times = timing.lap_times(race(), ["AAA", "BBB"])

    assert set(times["Driver"]) == {"AAA", "BBB"}
    assert times.loc[times["Fastest"], ["Driver", "LapNumber"]].values.tolist() == [["BBB", 3.0]]
    assert times.loc[times["PersonalBest"], "Driver"].tolist() == ["AAA", "BBB"]
    assert times.loc[times["Pit"], "LapNumber"].tolist() == [2.0, 3.0]


def test_spread_figure_only_plots_cars_on_the_lead_lap():
    figure = timing.spread_figure(timing.tower(race(), 5), {})

    assert list(figure.data[0].text) == ["AAA", "BBB"]
