import pandas as pd
import pytest

from src import conditions


def laps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "LapNumber": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "LapStartTime": pd.to_timedelta([0, 100, 200, 0, 101, 202], unit="s"),
            "Time": pd.to_timedelta([100, 200, 300, 101, 202, 303], unit="s"),
        }
    )


def test_weather_by_lap_takes_the_last_reading_before_each_lap_ends():
    weather = pd.DataFrame(
        {
            "Time": [0.0, 150.0, 250.0],
            "AirTemp": [30.0, 29.0, 28.0],
            "TrackTemp": [50.0, 48.0, 46.0],
            "Rainfall": [False, False, True],
        }
    )

    by_lap = conditions.weather_by_lap(weather, laps())
    summary = conditions.weather_summary(by_lap)

    assert by_lap["TrackTemp"].tolist() == [50.0, 48.0, 46.0]
    assert summary == {"track_start": 50.0, "track_end": 46.0, "air_start": 30.0, "rain_laps": [3]}


@pytest.mark.parametrize(
    ("category", "message", "expected"),
    [
        (
            "Other",
            "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 3 (RIC) - CAUSING A COLLISION",
            "Penalty",
        ),
        (
            "Other",
            "TURN 1 INCIDENT INVOLVING CARS 4 (NOR) AND 1 (VER) NOTED - LEAVING THE TRACK",
            "Investigation",
        ),
        (
            "Other",
            "CAR 31 (OCO) TIME 1:23.456 DELETED - TRACK LIMITS AT TURN 11 LAP 33",
            "Track limits",
        ),
        ("SafetyCar", "SAFETY CAR DEPLOYED", "Safety car"),
        ("Flag", "YELLOW IN TRACK SECTOR 4", "Flag"),
        ("Drs", "DRS ENABLED", "DRS"),
        ("Other", "RISK OF RAIN FOR F1 RACE IS 40 %", "Other"),
    ],
)
def test_message_type(category, message, expected):
    assert conditions.message_type(category, message) == expected


def test_race_control_leaves_out_blue_flags():
    messages = pd.DataFrame(
        {
            "Lap": [1, 2],
            "Category": ["Flag", "Other"],
            "Message": ["WAVED BLUE FLAG FOR CAR 24 (ZHO)", "DRS DISABLED"],
        }
    )

    assert conditions.race_control(messages)["Message"].tolist() == ["DRS DISABLED"]


def test_radio_clips_are_placed_on_the_drivers_lap():
    radio = pd.DataFrame(
        {"Time": [-30.0, 150.0, 250.0], "Driver": ["AAA", "AAA", "BBB"], "Url": ["a", "b", "c"]}
    )

    clips = conditions.radio_by_lap(radio, laps())

    assert clips[["Lap", "Driver", "Url"]].values.tolist() == [
        [0, "AAA", "a"],
        [2, "AAA", "b"],
        [3, "BBB", "c"],
    ]
