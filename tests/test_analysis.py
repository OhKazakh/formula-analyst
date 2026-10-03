import numpy as np
import pandas as pd
import pytest

from src import analysis


@pytest.fixture
def laps() -> pd.DataFrame:
    rows = []
    for driver, stints in {"AAA": [("MEDIUM", 10), ("HARD", 12)], "BBB": [("HARD", 22)]}.items():
        lap_number = 1
        for stint_number, (compound, length) in enumerate(stints, start=1):
            for tyre_life in range(1, length + 1):
                rows.append(
                    {
                        "Driver": driver,
                        "LapNumber": float(lap_number),
                        "Stint": float(stint_number),
                        "Compound": compound,
                        "TyreLife": float(tyre_life),
                        "LapTimeSeconds": 90.0 + 0.1 * tyre_life,
                    }
                )
                lap_number += 1
    return pd.DataFrame(rows)


def lap(**overrides) -> dict:
    defaults = {
        "Driver": "AAA",
        "LapNumber": 1.0,
        "LapTime": pd.Timedelta(milliseconds=90000),
        "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT,
        "TrackStatus": "1",
        "IsAccurate": True,
        "IsPersonalBest": True,
    }
    return defaults | overrides


def test_representative_laps_drops_pit_flagged_and_inaccurate_laps():
    laps = pd.DataFrame(
        [
            lap(LapNumber=1.0),
            lap(LapNumber=2.0, PitInTime=pd.Timedelta(minutes=30)),
            lap(LapNumber=3.0, PitOutTime=pd.Timedelta(minutes=31)),
            lap(LapNumber=4.0, TrackStatus="4"),
            lap(LapNumber=5.0, IsAccurate=False),
        ]
    )

    assert analysis.representative_laps(laps)["LapNumber"].tolist() == [1.0]


def test_fastest_lap_ignores_laps_not_marked_personal_best():
    laps = pd.DataFrame(
        [
            lap(LapNumber=1.0, LapTime=pd.Timedelta(milliseconds=91000)),
            lap(LapNumber=2.0, LapTime=pd.Timedelta(milliseconds=89000), IsPersonalBest=False),
            lap(LapNumber=3.0, LapTime=pd.Timedelta(milliseconds=90000)),
        ]
    )

    assert analysis.fastest_lap(laps, "AAA")["LapNumber"] == 3.0


def test_fastest_lap_is_none_without_timed_laps():
    laps = pd.DataFrame([lap(IsPersonalBest=False), lap(Driver="BBB", LapTime=pd.NaT)])

    assert analysis.fastest_lap(laps, "AAA") is None
    assert analysis.fastest_lap(laps, "BBB") is None
    assert analysis.fastest_lap(laps, "CCC") is None


def test_stint_summary_one_row_per_stint(laps):
    stints = analysis.stint_summary(laps)

    assert stints[["Driver", "Stint"]].values.tolist() == [["AAA", 1], ["AAA", 2], ["BBB", 1]]
    assert stints["Laps"].tolist() == [10, 12, 22]
    assert stints.loc[1, ["StartLap", "EndLap"]].tolist() == [11, 22]


def test_degradation_recovers_slope(laps):
    deg = analysis.degradation(laps, min_laps=5)

    assert len(deg) == 3
    assert deg["DegPerLap"].tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_degradation_skips_short_stints(laps):
    deg = analysis.degradation(laps, min_laps=11)

    assert set(zip(deg["Driver"], deg["Stint"], strict=True)) == {("AAA", 2), ("BBB", 1)}


def test_degradation_empty_when_nothing_qualifies(laps):
    deg = analysis.degradation(laps, min_laps=100)

    assert deg.empty
    assert list(deg.columns) == ["Driver", "Stint", "Compound", "Laps", "DegPerLap"]


def test_fuel_correction_removes_linear_fuel_gain():
    laps = pd.DataFrame({"LapNumber": [1.0, 2.0, 3.0], "LapTimeSeconds": [91.0, 90.95, 90.9]})

    corrected = analysis.fuel_corrected(laps, total_laps=3, fuel_effect=0.05)

    assert corrected.tolist() == pytest.approx([90.9, 90.9, 90.9])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (pd.Timedelta(milliseconds=83226), "1:23.226"),
        (pd.Timedelta(milliseconds=59500), "0:59.500"),
        (pd.NaT, ""),
    ],
)
def test_format_lap_time(value, expected):
    assert analysis.format_lap_time(value) == expected


TEAMMATES = {
    "AAA": {"color": "#ff8000", "linestyle": "solid"},
    "BBB": {"color": "#ff8000", "linestyle": "dashed"},
}


def test_pace_figure_distinguishes_teammates():
    laps = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "BBB", "BBB"],
            "LapNumber": [1.0, 2.0, 1.0, 2.0],
            "LapTimeSeconds": [90.0, 89.0, 90.5, 89.5],
        }
    )

    first, second = analysis.pace_figure(laps, ["AAA", "BBB"], TEAMMATES).axes[0].get_lines()

    assert first.get_color() == second.get_color() == "#ff8000"
    assert (first.get_linestyle(), first.get_marker()) == ("-", "o")
    assert (second.get_linestyle(), second.get_marker(), second.get_markerfacecolor()) == (
        "--",
        "s",
        "white",
    )


def test_speed_trace_distinguishes_teammates_by_line_style():
    trace = pd.DataFrame({"Distance": [0.0, 10.0], "Speed": [300.0, 305.0]})
    corners = pd.DataFrame(columns=["Number", "Letter", "Distance"])

    fig = analysis.speed_trace_figure({"AAA": trace, "BBB": trace}, TEAMMATES, corners)
    first, second = fig.axes[0].get_lines()

    assert (first.get_linestyle(), second.get_linestyle()) == ("-", "--")
    assert first.get_marker() == second.get_marker() == "None"


def test_stint_compound_ignores_missing_values_and_uses_majority():
    assert analysis.stint_compound(pd.Series([None, "SOFT", "MEDIUM", "SOFT"])) == "SOFT"
    assert analysis.stint_compound(pd.Series([None, None])) == analysis.UNKNOWN_COMPOUND


def test_stint_summary_labels_stint_with_known_compound(laps):
    laps.loc[laps["LapNumber"] == 1.0, "Compound"] = None

    stints = analysis.stint_summary(laps)

    assert stints["Compound"].tolist() == ["MEDIUM", "HARD", "HARD"]


def replay_laps() -> pd.DataFrame:
    laps = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "BBB", "BBB"],
            "LapNumber": [1.0, 2.0, 1.0, 2.0],
            "LapStartTime": pd.to_timedelta([0, 100, 0, 120], unit="s"),
            "Time": pd.to_timedelta([100, 200, 120, 240], unit="s"),
        }
    )
    return laps


SQUARE = pd.DataFrame({"X": [0.0, 100.0, 100.0, 0.0, 0.0], "Y": [0.0, 0.0, 100.0, 100.0, 0.0]})


def test_lap_profile_is_linear_at_constant_speed():
    trace = pd.DataFrame({"Distance": np.arange(0.0, 1001.0, 100.0), "Speed": 200.0})

    time_fraction, distance_fraction = analysis.lap_profile(trace)

    np.testing.assert_allclose(time_fraction, distance_fraction, atol=1e-9)


def test_lap_profile_spends_longer_in_slow_sections():
    trace = pd.DataFrame(
        {"Distance": np.arange(0.0, 1001.0, 100.0), "Speed": [100.0] * 6 + [300.0] * 5}
    )

    assert np.interp(0.5, *analysis.lap_profile(trace)) < 0.4


def test_replay_times_span_the_race():
    times = analysis.replay_times(replay_laps(), total_laps=2)

    assert (times[0], times[-1], len(times)) == (0.0, 240.0, 2 * analysis.REPLAY_FRAMES_PER_LAP)


def test_race_positions_interpolate_within_laps():
    positions = analysis.race_positions(replay_laps(), np.array([50.0, 150.0, 210.0]))
    progress = positions.pivot(index="Frame", columns="Driver", values="Progress")
    running = positions.pivot(index="Frame", columns="Driver", values="Running")

    np.testing.assert_allclose(progress["AAA"], [0.5, 1.5, 2.0])
    np.testing.assert_allclose(progress["BBB"], [50 / 120, 1.25, 1.75])
    assert running["AAA"].tolist() == [True, True, False]
    assert running["BBB"].tolist() == [True, True, True]


def test_track_points_follow_the_outline():
    x, y = analysis.track_points(SQUARE, np.array([0.0, 0.25, 0.5, 1.125]))

    np.testing.assert_allclose(x, [0.0, 100.0, 100.0, 50.0])
    np.testing.assert_allclose(y, [0.0, 0.0, 100.0, 0.0])


def test_replay_figure_frames_slider_and_finish_order():
    times = np.array([50.0, 150.0, 250.0])
    positions = analysis.race_positions(replay_laps(), times)
    corners = pd.DataFrame({"Label": ["1"], "X": [100.0], "Y": [0.0]})

    figure = analysis.replay_figure(SQUARE, corners, positions, ["AAA", "BBB"], {}, total_laps=2)

    assert [step.label for step in figure.layout.sliders[0].steps] == ["1", "2"]
    last = figure.frames[-1]
    assert np.isnan(last.data[0].x).all()
    assert last.data[1].text[0].endswith(" 1  AAA<br> 2  BBB")


def test_lap_positions_rank_drivers_by_crossing_time():
    laps = pd.DataFrame(
        {
            "Driver": ["AAA", "BBB", "CCC", "AAA", "BBB"],
            "LapNumber": [1.0, 1.0, 1.0, 2.0, 2.0],
            "Time": pd.to_timedelta([100, 101, 105, 199, 198], unit="s"),
        }
    )

    positions = analysis.lap_positions(laps).set_index(["Driver", "LapNumber"])["Position"]

    assert positions.to_dict() == {
        ("AAA", 1.0): 1,
        ("AAA", 2.0): 2,
        ("BBB", 1.0): 2,
        ("BBB", 2.0): 1,
        ("CCC", 1.0): 3,
    }


def test_position_figure_mutes_drivers_not_highlighted():
    positions = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "BBB", "BBB"],
            "LapNumber": [1, 2, 1, 2],
            "Position": [1, 2, 2, 1],
        }
    )

    figure = analysis.position_figure(positions, ["AAA", "BBB"], TEAMMATES, highlight=["AAA"])
    first, second = figure.axes[0].get_lines()

    assert first.get_color() == "#ff8000"
    assert second.get_color() == analysis.MUTED_COLOR


def racing_laps(times: dict[str, list[float]], teams: dict[str, str]) -> pd.DataFrame:
    rows = [
        lap(Driver=driver, LapNumber=float(n), LapTimeSeconds=seconds, Team=teams[driver])
        for driver, laps in times.items()
        for n, seconds in enumerate(laps, start=1)
    ]
    return pd.DataFrame(rows)


def test_quick_laps_drop_laps_slower_than_threshold():
    laps = racing_laps({"AAA": [90.0, 95.0, 100.0]}, {"AAA": "Alpha"})

    assert analysis.quick_laps(laps)["LapTimeSeconds"].tolist() == [90.0, 95.0]


def test_team_pace_orders_teams_by_median():
    laps = racing_laps(
        {"AAA": [91.0, 92.0], "BBB": [90.0, 90.5], "CCC": [93.0, 93.5]},
        {"AAA": "Alpha", "BBB": "Beta", "CCC": "Alpha"},
    )

    pace = analysis.team_pace(laps)

    assert pace["Team"].tolist() == ["Beta", "Alpha"]
    assert pace["Gap"].tolist() == pytest.approx([0.0, 2.25])


def test_mini_sector_times_add_up_to_the_lap():
    trace = pd.DataFrame({"Distance": np.linspace(0.0, 1000.0, 101), "Speed": 180.0})

    times = analysis.mini_sector_times(trace, sectors=4)

    assert times.sum() == pytest.approx(1000.0 / 50.0, rel=1e-2)
    np.testing.assert_allclose(times, times[0], rtol=1e-2)


def test_faster_by_sector_picks_the_quicker_driver():
    distance = np.linspace(0.0, 1000.0, 101)
    early = pd.DataFrame({"Distance": distance, "Speed": np.where(distance < 500, 300.0, 100.0)})
    late = pd.DataFrame({"Distance": distance, "Speed": np.where(distance < 500, 100.0, 300.0)})

    winners = analysis.faster_by_sector({"AAA": early, "BBB": late}, sectors=4)

    assert winners.tolist() == ["AAA", "AAA", "BBB", "BBB"]


def test_distinct_colors_shade_the_second_teammate():
    colors = analysis.distinct_colors(["AAA", "BBB"], TEAMMATES)

    assert colors["AAA"] == "#ff8000"
    assert colors["BBB"] != colors["AAA"]


def test_format_seconds():
    assert analysis.format_seconds(83.226) == "1:23.226"
