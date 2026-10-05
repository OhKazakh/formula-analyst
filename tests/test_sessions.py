import numpy as np
import pandas as pd
import pytest

from src import sessions


def qualifying_laps() -> pd.DataFrame:
    rows = [
        ("AAA", 1, 81.0, False),
        ("AAA", 2, 80.5, False),
        ("AAA", 3, 80.0, False),
        ("BBB", 1, 80.9, False),
        ("BBB", 2, 80.6, False),
        ("BBB", 3, 79.0, True),
        ("BBB", 3, 80.2, False),
        ("CCC", 1, 81.5, False),
        ("CCC", 2, 81.0, False),
        ("DDD", 1, 82.0, False),
    ]
    return pd.DataFrame(
        [
            {"Driver": d, "Team": f"Team {d}", "Part": p, "LapTimeSeconds": t, "Deleted": deleted}
            for d, p, t, deleted in rows
        ]
    )


def test_qualifying_results_use_the_last_part_each_driver_reached():
    table = sessions.qualifying_results(
        qualifying_laps(), ["AAA", "BBB", "CCC", "DDD"], "Qualifying"
    )

    assert table["Q3"].tolist()[:2] == [80.0, 80.2]
    assert table["Best"].tolist() == [80.0, 80.2, 81.0, 82.0]
    assert table["Gap"].tolist() == pytest.approx([0.0, 0.2, 1.0, 2.0])
    assert table["Out"].tolist() == ["Q3", "Q3", "Q2", "Q1"]


def test_sprint_qualifying_parts_are_labelled_sq():
    assert sessions.part_labels("Sprint Qualifying") == ["SQ1", "SQ2", "SQ3"]
    assert sessions.part_labels("Sprint Shootout") == ["SQ1", "SQ2", "SQ3"]


def practice_laps() -> pd.DataFrame:
    rows = []
    for driver, base in (("AAA", 90.0), ("BBB", 90.5)):
        for number in range(1, 16):
            rows.append(
                {
                    "Driver": driver,
                    "Team": f"Team {driver}",
                    "LapNumber": float(number),
                    "LapTimeSeconds": base + 0.05 * number if number != 8 else base + 20,
                    "Stint": 1.0 if number < 8 else 2.0,
                    "Compound": "MEDIUM" if number < 8 else "HARD",
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "TrackStatus": "1",
                    "IsAccurate": True,
                    "IsPersonalBest": number == 1,
                }
            )
    return pd.DataFrame(rows)


def test_practice_results_rank_personal_bests():
    table = sessions.practice_results(practice_laps())

    assert table["Driver"].tolist() == ["AAA", "BBB"]
    assert table["Gap"].tolist() == pytest.approx([0.0, 0.5])
    assert table["Laps"].tolist() == [15, 15]


def test_long_runs_are_unbroken_runs_of_race_pace_laps():
    runs = sessions.long_runs(practice_laps())

    assert sorted(runs[["Driver", "Stint", "Laps"]].values.tolist()) == [
        ["AAA", 1, 7],
        ["AAA", 2, 7],
        ["BBB", 1, 7],
        ["BBB", 2, 7],
    ]
    assert runs["Average"].is_monotonic_increasing
    assert runs["Degradation"].to_numpy() == pytest.approx(np.full(4, 0.05))


def test_gap_figure_draws_one_bar_per_timed_driver():
    table = sessions.qualifying_results(
        qualifying_laps(), ["AAA", "BBB", "CCC", "DDD"], "Qualifying"
    )

    figure = sessions.gap_figure(table, {}, cutoffs=(2,))

    assert list(figure.data[0].y) == ["AAA", "BBB", "CCC", "DDD"]
    assert len(figure.layout.shapes) == 1


def test_long_runs_keep_laps_fastf1_marks_inaccurate_but_drop_cool_downs():
    laps = practice_laps().assign(IsAccurate=False)
    cool_down = (laps["Driver"] == "AAA") & (laps["LapNumber"] == 4.0)
    laps.loc[cool_down, "LapTimeSeconds"] += 4.0

    runs = sessions.long_runs(laps)

    assert sorted(runs.loc[runs["Driver"] == "AAA", "Laps"].tolist()) == [7]
    assert len(runs[runs["Driver"] == "BBB"]) == 2
