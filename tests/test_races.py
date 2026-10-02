from pathlib import Path

import pandas as pd
import pytest

from src import races


def make_race(round_number: int = 16, event: str = "Italian Grand Prix") -> races.Race:
    no_time = pd.Series([pd.NaT] * 3, dtype="timedelta64[ns]")
    laps = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "BBB"],
            "Team": ["Alpha", "Alpha", "Beta"],
            "LapNumber": [1.0, 2.0, 1.0],
            "LapTime": pd.to_timedelta([91000, 90000, 92000], unit="ms"),
            "LapTimeSeconds": [91.0, 90.0, 92.0],
            "Stint": [1.0, 1.0, 1.0],
            "Compound": ["SOFT", "SOFT", "HARD"],
            "TyreLife": [1.0, 2.0, 1.0],
            "PitInTime": no_time,
            "PitOutTime": no_time,
            "TrackStatus": ["1", "1", "1"],
            "IsAccurate": [True, True, True],
            "IsPersonalBest": [True, True, True],
        }
    )
    telemetry = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "BBB"],
            "Distance": [0.0, 10.0, 0.0],
            "Speed": [300.0, 305.0, 290.0],
        }
    )
    return races.Race(
        year=2024,
        round_number=round_number,
        event=event,
        total_laps=2,
        order=["AAA", "BBB"],
        laps=laps,
        telemetry=telemetry,
        corners=pd.DataFrame({"Number": [1], "Letter": [""], "Distance": [850.0]}),
        driver_styles={"AAA": {"color": "#ff0000", "linestyle": "solid"}},
        compound_colors={"SOFT": "#da291c", "HARD": "#f0f0ec"},
    )


def test_save_and_read_round_trip(tmp_path):
    race = make_race()

    loaded = races.read_race(races.save_race(race, tmp_path))

    pd.testing.assert_frame_equal(loaded.laps, race.laps)
    pd.testing.assert_frame_equal(loaded.telemetry, race.telemetry)
    pd.testing.assert_frame_equal(loaded.corners, race.corners)
    assert (loaded.year, loaded.round_number, loaded.event, loaded.total_laps) == (
        2024,
        16,
        "Italian Grand Prix",
        2,
    )
    assert loaded.order == race.order
    assert loaded.driver_styles == race.driver_styles
    assert loaded.compound_colors == race.compound_colors


def test_saved_races_sorted_by_round(tmp_path):
    races.save_race(make_race(16, "Italian Grand Prix"), tmp_path)
    races.save_race(make_race(1, "Bahrain Grand Prix"), tmp_path)

    saved = races.saved_races(tmp_path)

    assert saved["EventName"].tolist() == ["Bahrain Grand Prix", "Italian Grand Prix"]
    assert saved["Round"].tolist() == [1, 16]


def test_saved_races_empty_without_data(tmp_path):
    saved = races.saved_races(tmp_path / "missing")

    assert saved.empty
    assert list(saved.columns) == races.SAVED_COLUMNS


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("Italian Grand Prix", "16-italian-grand-prix"),
        ("São Paulo Grand Prix", "16-sao-paulo-grand-prix"),
    ],
)
def test_race_dir_slug(event, expected):
    assert races.race_dir(Path("data"), 2024, 16, event) == Path("data", "2024", expected)


def test_load_race_prefers_saved_copy(tmp_path, monkeypatch):
    races.save_race(make_race(), tmp_path)
    monkeypatch.setattr(races, "load_live_race", lambda *_: pytest.fail("downloaded a saved race"))

    race = races.load_race(2024, "Italian Grand Prix", tmp_path)

    assert race.event == "Italian Grand Prix"


def test_load_race_downloads_unsaved_race(tmp_path, monkeypatch):
    downloaded = make_race(1, "Bahrain Grand Prix")
    monkeypatch.setattr(races, "load_live_race", lambda year, event: downloaded)

    assert races.load_race(2024, "Bahrain Grand Prix", tmp_path) is downloaded


def test_trace_returns_one_drivers_telemetry():
    trace = make_race().trace("AAA")

    assert trace["Speed"].tolist() == [300.0, 305.0]


def test_live_timing_unavailable_when_forced_offline(monkeypatch):
    monkeypatch.setenv(races.OFFLINE_ENV, "1")

    assert races.live_timing_available() is False
