import pandas as pd

from src import openf1

START = pd.Timestamp("2026-10-11 12:00:00")


def at(seconds: float) -> pd.Timestamp:
    return START + pd.Timedelta(seconds=seconds)


def messages(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": at(seconds),
                "message": message,
                "flag": flag,
                "scope": scope,
                "sector": sector,
                "qualifying_phase": phase,
            }
            for seconds, message, flag, scope, sector, phase in rows
        ]
    )


def test_track_status_follows_flags_and_safety_cars():
    feed = messages(
        (10, "YELLOW IN TRACK SECTOR 4", "YELLOW", "Sector", 4, None),
        (20, "CLEAR IN TRACK SECTOR 4", "CLEAR", "Sector", 4, None),
        (30, "SAFETY CAR DEPLOYED", None, None, None, None),
        (90, "TRACK CLEAR", "CLEAR", "Track", None, None),
        (120, "VSC DEPLOYED", None, None, None, None),
        (150, "VSC ENDING", None, None, None, None),
    )

    changes = openf1.status_changes(feed)

    assert changes["Status"].tolist() == ["2", "1", "4", "1", "6", "7", "1"]
    assert changes["Date"].iloc[-1] == at(165)


def test_each_lap_lists_the_statuses_it_ran_under():
    changes = pd.DataFrame(
        {"Date": [at(0), at(30), at(90)], "Status": ["1", "4", "1"]},
    )
    starts = pd.Series([at(0), at(40), at(80), at(100)])
    ends = pd.Series([at(40), at(80), at(100), at(140)])

    assert openf1.lap_statuses(starts, ends, changes) == ["14", "4", "41", "1"]


def test_deleted_laps_skip_reinstated_times():
    feed = pd.DataFrame(
        {
            "message": [
                "CAR 1 (NOR) TIME 1:31.249 DELETED - TRACK LIMITS AT TURN 4 LAP 7 15:02:11",
                "CAR 44 (HAM) TIME 1:30.500 DELETED - TRACK LIMITS AT TURN 1 LAP 3 15:01:00",
                "CAR 44 (HAM) TIME 1:30.500 LAP 3 REINSTATED",
                "CAR 16 (LEC) LAP DELETED - TRACK LIMITS AT TURN 2 LAP 5 (PIT)",
            ]
        }
    )

    assert openf1.deleted_laps(feed) == {(1, 91.249)}


def test_qualifying_parts_start_with_each_session_start():
    feed = messages(
        (0, "SESSION STARTED", None, None, None, 1),
        (1200, "SESSION STARTED", None, None, None, 2),
        (2100, "SESSION STARTED", None, None, None, 3),
    )
    starts = pd.Series([at(60), at(1300), pd.NaT, at(2200)])
    drivers = pd.Series(["AAA"] * 4)

    assert openf1.qualifying_parts(starts, drivers, feed).tolist() == [1, 2, 3, 3]


def make_laps() -> pd.DataFrame:
    durations = [None, 80.0, 79.0, 95.0, None, 78.5]
    rows, clock = [], 0.0
    for number, duration in enumerate(durations, start=1):
        rows.append(
            {
                "driver_number": 1,
                "lap_number": number,
                "date_start": at(clock).isoformat() + "+00:00",
                "lap_duration": duration,
                "duration_sector_1": None if duration is None else duration / 3,
                "duration_sector_2": None if duration is None else duration / 3,
                "duration_sector_3": None if duration is None else duration / 3,
                "i1_speed": 300,
                "i2_speed": 280,
                "st_speed": 320,
                "is_pit_out_lap": number in (1, 5),
            }
        )
        clock += duration or 100.0
    return pd.DataFrame(rows)


def test_lap_table_builds_stints_pit_laps_and_personal_bests():
    drivers = pd.DataFrame(
        {"driver_number": [1], "name_acronym": ["NOR"], "team_name": ["McLaren"]}
    )
    stints = pd.DataFrame(
        {
            "driver_number": [1, 1],
            "stint_number": [1, 2],
            "lap_start": [1, 5],
            "lap_end": [4, 6],
            "compound": ["MEDIUM", "SOFT"],
            "tyre_age_at_start": [2, 0],
        }
    )
    empty = messages().reindex(columns=["Date", "message", "flag", "scope", "sector"])

    laps = openf1.lap_table(make_laps(), drivers, stints, pd.DataFrame(), empty, START, race=False)

    assert laps["Compound"].tolist() == ["MEDIUM"] * 4 + ["SOFT"] * 2
    assert laps["TyreLife"].tolist() == [3.0, 4.0, 5.0, 6.0, 1.0, 2.0]
    assert laps["PitOutTime"].notna().tolist() == [True, False, False, False, True, False]
    assert laps["PitInTime"].notna().tolist() == [False, False, False, True, False, True]
    assert laps["IsPersonalBest"].tolist() == [False, True, True, False, False, False]
    assert laps["LapTimeSeconds"].isna().tolist() == [True, False, False, False, True, False]


def test_classification_prefers_the_official_order():
    names = pd.Series(["NOR", "VER", "LEC"], index=[1, 3, 16])
    results = pd.DataFrame({"driver_number": [3, 1, 16], "position": [1, 2, None]})
    laps = pd.DataFrame({"Driver": ["NOR", "VER", "LEC", "HAD"]})

    assert openf1.classification([], results, names, laps) == ["VER", "NOR", "LEC", "HAD"]
    assert openf1.classification(["NOR", "VER"], results, names, laps) == [
        "NOR",
        "VER",
        "LEC",
        "HAD",
    ]


def test_driver_styles_dash_the_second_driver_of_a_team():
    drivers = pd.DataFrame(
        {
            "name_acronym": ["PIA", "NOR", "COL"],
            "team_name": ["McLaren", "McLaren", "Alpine"],
            "team_colour": ["F47600", "F47600", None],
        }
    )

    styles = openf1.driver_styles(drivers, ["NOR", "PIA", "COL"])

    assert styles == {
        "NOR": {"color": "#f47600", "linestyle": "solid"},
        "PIA": {"color": "#f47600", "linestyle": "dashed"},
    }
