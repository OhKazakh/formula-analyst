from pathlib import Path

import numpy as np
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
            "LapStartTime": pd.to_timedelta([3600, 3691, 3600], unit="s"),
            "Time": pd.to_timedelta([3691, 3781, 3692], unit="s"),
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
        track=pd.DataFrame({"X": [0.0, 100.0, 100.0, 0.0], "Y": [0.0, 0.0, 50.0, 0.0]}),
        map_corners=pd.DataFrame({"Label": ["1"], "X": [105.0], "Y": [0.0]}),
    )


def square_circuit(rotation: float = 0.0) -> dict:
    side = np.arange(0, 1000, 250)
    return {
        "x": np.concatenate([side, np.full(4, 1000), 1000 - side, np.zeros(4), [0]]).tolist(),
        "y": np.concatenate([np.zeros(4), side, np.full(4, 1000), 1000 - side, [0]]).tolist(),
        "rotation": rotation,
        "corners": [
            {"number": 1, "length": 1000, "angle": 0.0, "trackPosition": {"x": 1000, "y": 0}},
            {"number": 2, "length": 2000, "angle": 90.0, "trackPosition": {"x": 1000, "y": 1000}},
        ],
    }


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
    pd.testing.assert_frame_equal(loaded.track, race.track, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.map_corners, race.map_corners)


def test_read_race_without_track_data(tmp_path):
    path = races.save_race(make_race(), tmp_path)
    (path / "track.parquet").unlink()

    loaded = races.read_race(path)

    assert loaded.track.empty
    assert list(loaded.track.columns) == races.TRACK_COLUMNS


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


def test_clean_compounds_turns_placeholders_into_missing_values():
    cleaned = races.clean_compounds(pd.Series(["SOFT", "nan", "None", "", "UNKNOWN"]))

    assert cleaned.isna().tolist() == [False, True, True, True, False]


def test_drivers_excludes_entrants_without_laps():
    race = make_race()
    race.order.append("CCC")

    assert race.drivers == ["AAA", "BBB"]


def test_bundle_version_changes_when_a_race_is_added_or_rewritten(tmp_path):
    empty = races.bundle_version(tmp_path)
    races.save_race(make_race(), tmp_path)
    saved = races.bundle_version(tmp_path)

    race = make_race()
    race.laps.loc[0, "Compound"] = None
    races.save_race(race, tmp_path)

    assert len({empty, saved, races.bundle_version(tmp_path)}) == 3


def test_track_map_starts_at_timing_line():
    corners = pd.DataFrame({"Number": [1, 2], "Letter": ["", ""], "Distance": [50.0, 150.0]})

    track, _ = races.track_map(square_circuit(), corners)

    assert track.iloc[0].tolist() == [50.0, 0.0]
    assert track.iloc[0].tolist() == track.iloc[-1].tolist()
    assert len(track) == 17


def test_track_map_without_corner_distances_keeps_outline_start():
    corners = pd.DataFrame(columns=races.CORNER_COLUMNS)

    track, map_corners = races.track_map(square_circuit(), corners)

    assert track.iloc[0].tolist() == [0.0, 0.0]
    assert map_corners["Label"].tolist() == ["1", "2"]


def test_track_map_applies_rotation():
    corners = pd.DataFrame(columns=races.CORNER_COLUMNS)

    track, map_corners = races.track_map(square_circuit(rotation=90.0), corners)

    np.testing.assert_allclose(track.iloc[1], [0.0, 25.0], atol=1e-9)
    np.testing.assert_allclose(
        map_corners.iloc[0][["X", "Y"]].astype(float), [0.0, 150.0], atol=1e-9
    )
