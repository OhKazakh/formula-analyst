import numpy as np
import pandas as pd
import pytest

from src import telemetry

CORNERS = pd.DataFrame({"Number": [1, 2], "Letter": ["", "A"], "Distance": [300.0, 700.0]})


def trace(speed: float = 180.0, time_offset: float = 0.0, length: float = 1000.0) -> pd.DataFrame:
    distance = np.linspace(0.0, length, 201)
    speeds = np.full(distance.size, speed)
    speeds[(distance > 250) & (distance < 350)] = 80.0
    speeds[(distance > 650) & (distance < 750)] = 120.0
    brake = ((distance > 200) & (distance < 300)) | ((distance > 620) & (distance < 700))
    return pd.DataFrame(
        {
            "Distance": distance,
            "Speed": speeds,
            "Time": distance / (speed / 3.6) + time_offset,
            "Throttle": np.where(brake, 0, 100),
            "Brake": brake,
            "Gear": np.where(speeds < 100, 3, 7),
            "RPM": np.full(distance.size, 11000),
            "DRS": np.zeros(distance.size, dtype=bool),
        }
    )


def test_lap_delta_grows_with_the_slower_car():
    delta = telemetry.lap_delta(trace(200.0), trace(180.0))

    assert delta["Delta"].iloc[0] == pytest.approx(0.0)
    assert delta["Delta"].iloc[-1] == pytest.approx(1000 / 50 - 1000 / (200 / 3.6), rel=1e-3)
    assert delta["Delta"].is_monotonic_increasing


def test_elapsed_is_stretched_to_the_official_lap_time():
    assert telemetry.elapsed(trace(), lap_time=25.0)[-1] == pytest.approx(25.0)


def test_channels_only_lists_data_every_trace_has():
    full, speed_only = trace(), trace()[["Distance", "Speed"]]

    assert telemetry.channels({"A": full, "B": full}) == list(telemetry.CHANNELS)
    assert telemetry.channels({"A": full, "B": speed_only}) == []


def test_corner_speeds_find_apex_and_braking_point():
    corners = telemetry.corner_speeds(trace(), CORNERS)

    assert corners["Corner"].tolist() == ["1", "2A"]
    assert corners["MinSpeed"].tolist() == [80.0, 120.0]
    assert corners["BrakingPoint"].tolist() == pytest.approx([95.0, 75.0])


def test_corner_comparison_adds_the_speed_difference():
    slow = trace()
    fast = trace().assign(Speed=lambda rows: rows["Speed"] + 5)

    table = telemetry.corner_comparison({"AAA": slow, "BBB": fast}, CORNERS)

    assert table["Difference"].tolist() == [5.0, 5.0]
    assert list(table.columns[:3]) == ["Corner", "AAA min", "AAA brakes"]


def test_telemetry_figure_rows_follow_the_available_channels():
    styles = {"AAA": {"color": "#ff8000", "linestyle": "solid"}}
    both = telemetry.telemetry_figure({"AAA": trace(), "BBB": trace(170)}, styles, CORNERS)
    speed_only = telemetry.telemetry_figure(
        {"AAA": trace()[["Distance", "Speed"]]}, styles, CORNERS
    )

    assert len([key for key in both.layout if key.startswith("yaxis")]) == 7
    assert len([key for key in speed_only.layout if key.startswith("yaxis")]) == 1


def test_gear_runs_split_the_lap_where_the_gear_changes():
    track = pd.DataFrame({"X": [0.0, 1000.0, 1000.0, 0.0, 0.0], "Y": [0.0, 0.0, 10.0, 10.0, 0.0]})

    runs = telemetry.gear_runs(trace(), track)

    assert [gear for gear, _, _ in runs] == [7, 3, 7]
