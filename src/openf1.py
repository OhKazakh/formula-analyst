from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests
from fastf1.ergast import Ergast
from fastf1.mvapi.api import get_circuit

from src import races
from src.analysis import GREEN_FLAG, RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR, fastest_lap

API = "https://api.openf1.org/v1"
PAUSE = 0.4
RETRIES = 6
YELLOW = "2"
VSC, VSC_ENDING = VIRTUAL_SAFETY_CAR
# Race control doesn't announce the end of a VSC, so the ending phase is given a typical length.
VSC_ENDING_LENGTH = pd.Timedelta(seconds=15)
COMPOUND_COLORS = {
    "SOFT": "#da291c",
    "MEDIUM": "#ffd12e",
    "HARD": "#f0f0ec",
    "INTERMEDIATE": "#43b02a",
    "WET": "#0067ad",
    "UNKNOWN": "#00ffff",
    "TEST-UNKNOWN": "#434649",
}
OFFICIAL_RESULTS = {
    races.RACE: "get_race_results",
    "Sprint": "get_sprint_results",
    "Qualifying": "get_qualifying_results",
}
# FastF1's rule: a lap is deleted when race control quotes its time, unless it is reinstated.
QUOTED_TIME = r"^CAR (?P<number>\d{1,2}) .* TIME (?P<time>\d:\d\d\.\d\d\d) "
DELETED_TIME = re.compile(QUOTED_TIME + "DELETED - ")
REINSTATED_TIME = re.compile(QUOTED_TIME + ".*REINSTATED")


def fetch(path: str, **params) -> pd.DataFrame:
    for attempt in range(RETRIES):
        time.sleep(PAUSE)
        response = requests.get(f"{API}/{path}", params=params, timeout=60)
        if response.status_code == 404:
            return pd.DataFrame()
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(float(response.headers.get("Retry-After", 5 * 2**attempt)))
            continue
        response.raise_for_status()
        return pd.DataFrame(response.json())
    raise races.RaceDataUnavailable(f"OpenF1 kept refusing requests for {path}.")


def utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="ISO8601").dt.tz_convert(None)


def find_session(year: int, event: str, session: str) -> pd.Series:
    schedule = races.event_schedule(year)
    weekend = schedule[schedule["EventName"] == event].iloc[0]
    when = next(
        weekend[f"Session{number}DateUtc"]
        for number in range(1, 6)
        if weekend[f"Session{number}"] == session
    )
    found = fetch("sessions", year=year, session_name=session)
    if found.empty:
        raise races.RaceDataUnavailable(f"OpenF1 has no {year} {event} {session.lower()}.")
    found = found.assign(Start=utc(found["date_start"]))
    closest = found.loc[(found["Start"] - when).abs().idxmin()].copy()
    if abs(closest["Start"] - when) > pd.Timedelta(days=1):
        raise races.RaceDataUnavailable(f"OpenF1 has no {year} {event} {session.lower()}.")
    closest["RoundNumber"] = int(weekend["RoundNumber"])
    return closest


# FastF1's track status codes, rebuilt from race control: one change per row.
def status_changes(messages: pd.DataFrame) -> pd.DataFrame:
    events = [
        (date, message.upper(), flag, scope, sector)
        for date, message, flag, scope, sector in zip(
            messages["Date"],
            messages["message"].fillna(""),
            messages["flag"],
            messages["scope"],
            messages["sector"],
            strict=True,
        )
    ]
    events += [
        (date + VSC_ENDING_LENGTH, "VSC ENDED", None, None, None)
        for date, message, *_ in events
        if "SAFETY CAR ENDING" in message or message.startswith("VSC ENDING")
    ]
    yellow: set = set()
    safety_car = virtual = ending = red = False
    changes, current = [], None
    for date, message, flag, scope, sector in sorted(events, key=lambda event: event[0]):
        if "VIRTUAL SAFETY CAR DEPLOYED" in message or message.startswith("VSC DEPLOYED"):
            virtual, ending = True, False
        elif "SAFETY CAR ENDING" in message or message.startswith("VSC ENDING"):
            virtual, ending = False, True
        elif message == "VSC ENDED":
            ending = False
        elif message.startswith("SAFETY CAR DEPLOYED"):
            safety_car, virtual, ending = True, False, False
        elif flag == "RED":
            red = True
        elif flag == "CLEAR" and scope == "Track":
            safety_car = virtual = ending = red = False
            yellow.clear()
        elif flag == "GREEN" and scope == "Track":
            red = False
        elif flag in ("YELLOW", "DOUBLE YELLOW"):
            yellow.add(sector if scope == "Sector" else "track")
        elif flag == "CLEAR" and scope == "Sector":
            yellow.discard(sector)
        status = (
            RED_FLAG
            if red
            else SAFETY_CAR
            if safety_car
            else VSC
            if virtual
            else VSC_ENDING
            if ending
            else YELLOW
            if yellow
            else GREEN_FLAG
        )
        if status != current:
            changes.append((date, status))
            current = status
    return pd.DataFrame(changes, columns=["Date", "Status"])


def lap_statuses(starts: pd.Series, ends: pd.Series, changes: pd.DataFrame) -> list[str]:
    statuses = []
    for start, end in zip(starts, ends, strict=True):
        if pd.isna(start) or changes.empty:
            statuses.append(GREEN_FLAG)
            continue
        before = changes[changes["Date"] <= start]
        seen = [before["Status"].iloc[-1] if not before.empty else GREEN_FLAG]
        if pd.notna(end):
            during = changes[(changes["Date"] > start) & (changes["Date"] < end)]
            seen += [status for status in during["Status"] if status not in seen]
        statuses.append("".join(dict.fromkeys(seen)))
    return statuses


def deleted_laps(messages: pd.DataFrame) -> set[tuple[int, float]]:
    def quoted(pattern: re.Pattern) -> set[tuple[int, float]]:
        found = set()
        for message in messages.get("message", pd.Series(dtype=str)).fillna(""):
            if match := pattern.match(message.strip()):
                minutes, seconds = match["time"].split(":")
                found.add((int(match["number"]), round(int(minutes) * 60 + float(seconds), 3)))
        return found

    return quoted(DELETED_TIME) - quoted(REINSTATED_TIME)


def qualifying_parts(starts: pd.Series, drivers: pd.Series, messages: pd.DataFrame) -> pd.Series:
    began = messages[
        (messages["message"] == "SESSION STARTED") & messages["qualifying_phase"].notna()
    ]
    first = began.sort_values("Date").groupby("qualifying_phase")["Date"].first().sort_index()
    if first.empty:
        return pd.Series(1, index=starts.index)
    parts = pd.Series(
        np.clip(np.searchsorted(first.to_numpy(), starts.to_numpy(), side="right"), 1, len(first)),
        index=starts.index,
    ).where(starts.notna())
    # A lap without a start time belongs with the driver's neighbouring laps.
    parts = parts.groupby(drivers).transform(lambda part: part.bfill().ffill())
    return parts.fillna(1).astype(int)


def lap_table(
    raw: pd.DataFrame,
    drivers: pd.DataFrame,
    stints: pd.DataFrame,
    pits: pd.DataFrame,
    messages: pd.DataFrame,
    t0: pd.Timestamp,
    qualifying: bool = False,
    race: bool = True,
) -> pd.DataFrame:
    names = drivers.set_index("driver_number")["name_acronym"]
    teams = drivers.set_index("driver_number")["team_name"]
    laps = raw.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    start = utc(laps["date_start"])
    seconds = pd.to_numeric(laps["lap_duration"], errors="coerce")
    if not race:
        # An out-lap from the garage is timed from the previous lap's end, garage time included.
        seconds = seconds.mask(laps["is_pit_out_lap"].fillna(False).astype(bool))
    lap_time = pd.to_timedelta(seconds, unit="s")
    # A lap without a time ends when the driver's next lap starts.
    next_start = start.groupby(laps["driver_number"]).shift(-1)
    end = (start + lap_time).fillna(next_start)
    table = pd.DataFrame(
        {
            "Driver": laps["driver_number"].map(names),
            "Team": laps["driver_number"].map(teams),
            "LapNumber": laps["lap_number"].astype(float),
            "LapTime": lap_time,
            "LapTimeSeconds": seconds,
            "LapStartTime": start - t0,
            "Time": end - t0,
            "Sector1": pd.to_numeric(laps["duration_sector_1"], errors="coerce"),
            "Sector2": pd.to_numeric(laps["duration_sector_2"], errors="coerce"),
            "Sector3": pd.to_numeric(laps["duration_sector_3"], errors="coerce"),
            "SpeedI1": pd.to_numeric(laps["i1_speed"], errors="coerce"),
            "SpeedI2": pd.to_numeric(laps["i2_speed"], errors="coerce"),
            "SpeedFL": np.nan,
            "SpeedST": pd.to_numeric(laps["st_speed"], errors="coerce"),
        }
    )
    table = table.join(_stint_columns(laps, stints)).join(_pit_columns(laps, pits, start, end, t0))
    if not race:
        # Outside a race every run ends in the pit lane, and the last one has no exit to record.
        last = table["LapNumber"] == table.groupby("Driver")["LapNumber"].transform("max")
        unrecorded = last & table["PitInTime"].isna()
        table.loc[unrecorded, "PitInTime"] = table.loc[unrecorded, "Time"]
    changes = status_changes(messages)
    table["TrackStatus"] = lap_statuses(start, end, changes)
    deleted = deleted_laps(messages)
    table["Deleted"] = [
        pd.notna(lap) and (int(number), round(float(lap), 3)) in deleted
        for number, lap in zip(laps["driver_number"], table["LapTimeSeconds"], strict=True)
    ]
    extra = []
    groups = [table["Driver"]]
    if qualifying:
        table["Part"] = qualifying_parts(start, table["Driver"], messages)
        extra = ["Part", "Deleted"]
        # Personal bests start again in each part of qualifying.
        groups.append(table["Part"])
    # As in F1's timing feed, pit laps and a race's first lap are never personal bests.
    excluded = table["PitInTime"].notna() | table["PitOutTime"].notna()
    if race:
        excluded |= table["LapNumber"] == 1
    valid = table["LapTimeSeconds"].where(~table["Deleted"] & ~excluded)
    table["IsPersonalBest"] = valid.notna() & valid.le(valid.groupby(groups).cummin())
    # As in FastF1, laps behind a safety car or whose sectors don't add up aren't accurate.
    sectors = table[["Sector1", "Sector2", "Sector3"]].sum(axis=1, min_count=3)
    table["IsAccurate"] = (
        table["LapTime"].notna()
        & table["PitInTime"].isna()
        & table["PitOutTime"].isna()
        & (table["LapNumber"] > 1)
        & ((sectors - table["LapTimeSeconds"]).abs() < 0.1)
        & ~table["TrackStatus"].str.contains(f"[{SAFETY_CAR}{VSC}{VSC_ENDING}]")
    )
    table["Compound"] = races.clean_compounds(table["Compound"])
    return races.tyre_stints(table[races.LAP_COLUMNS + extra])


def _stint_columns(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    columns = pd.DataFrame(index=laps.index, columns=["Stint", "Compound", "TyreLife"])
    for stint in stints.itertuples():
        rows = (
            (laps["driver_number"] == stint.driver_number)
            & (laps["lap_number"] >= stint.lap_start)
            & (laps["lap_number"] <= stint.lap_end)
        )
        age = stint.tyre_age_at_start if pd.notna(stint.tyre_age_at_start) else 0
        columns.loc[rows, "Stint"] = float(stint.stint_number)
        columns.loc[rows, "Compound"] = str(stint.compound).upper()
        columns.loc[rows, "TyreLife"] = age + laps.loc[rows, "lap_number"] - stint.lap_start + 1
    return columns.astype({"Stint": float, "TyreLife": float})


# OpenF1 stamps a pit visit at the exit and gives the time since the entry. Its lap number
# isn't always the in-lap, so each end goes on the lap that was running at that moment.
def _pit_columns(
    laps: pd.DataFrame, pits: pd.DataFrame, start: pd.Series, end: pd.Series, t0: pd.Timestamp
) -> pd.DataFrame:
    pit_in = pd.Series(pd.NaT, index=laps.index, dtype="timedelta64[ns]")
    pit_out = pd.Series(pd.NaT, index=laps.index, dtype="timedelta64[ns]")
    for pit in pits.itertuples():
        own = laps.index[(laps["driver_number"] == pit.driver_number) & start.notna()]
        starts = start[own].to_numpy()
        exit_ = utc(pd.Series([pit.date])).iloc[0]
        lane = pd.Timedelta(seconds=pit.lane_duration) if pd.notna(pit.lane_duration) else None
        for when, column in ((exit_, pit_out), (exit_ - lane if lane else None, pit_in)):
            if when is None or len(own) == 0:
                continue
            index = np.searchsorted(starts, np.datetime64(when + pd.Timedelta(milliseconds=1)))
            if index:
                column[own[index - 1]] = when - t0
    # Not every visit has a pit row, but an out-lap always follows an in-lap.
    flagged = laps["is_pit_out_lap"].fillna(False).astype(bool)
    pit_out[flagged & pit_out.isna()] = start[flagged & pit_out.isna()] - t0
    out_next = (
        pit_out.notna().groupby(laps["driver_number"]).shift(-1, fill_value=False).astype(bool)
    )
    pit_in[out_next & pit_in.isna()] = end[out_next & pit_in.isna()] - t0
    return pd.DataFrame({"PitInTime": pit_in, "PitOutTime": pit_out})


# OpenF1's classification is the order on track, so penalties come from Jolpica where it has them.
def official_order(year: int, round_number: int, session: str) -> list[str]:
    method = OFFICIAL_RESULTS.get(session)
    if method is None:
        return []
    try:
        response = getattr(Ergast(result_type="pandas", auto_cast=True), method)(
            season=year, round=round_number
        )
    except Exception:
        return []
    if not response.content:
        return []
    return response.content[0].sort_values("position")["driverCode"].tolist()


def classification(
    official: list[str], results: pd.DataFrame, names: pd.Series, laps: pd.DataFrame
) -> list[str]:
    order = list(official)
    if not order and not results.empty:
        ranked = results.sort_values("position", na_position="last")
        order = [names[number] for number in ranked["driver_number"] if number in names.index]
    return order + [driver for driver in laps["Driver"].dropna().unique() if driver not in order]


def telemetry(
    key: int, laps: pd.DataFrame, numbers: dict[str, int], t0: pd.Timestamp
) -> pd.DataFrame:
    traces = []
    for driver in laps["Driver"].dropna().unique():
        lap = fastest_lap(laps, driver)
        if lap is None or pd.isna(lap["LapStartTime"]):
            continue
        start = t0 + lap["LapStartTime"]
        car = fetch(
            "car_data",
            session_key=key,
            driver_number=numbers[driver],
            **{"date>": start.isoformat(), "date<": (start + lap["LapTime"]).isoformat()},
        )
        if car.empty:
            continue
        seconds = (utc(car["date"]) - start).dt.total_seconds().to_numpy()
        speed = pd.to_numeric(car["speed"], errors="coerce").fillna(0).to_numpy()
        steps = np.diff(seconds, prepend=0.0)
        traces.append(
            pd.DataFrame(
                {
                    "Driver": driver,
                    "Distance": np.cumsum(speed / 3.6 * steps).astype("float32"),
                    "Speed": speed.astype("int16"),
                    "Time": seconds.astype("float32"),
                    "RPM": pd.to_numeric(car["rpm"], errors="coerce").fillna(0).to_numpy("int16"),
                    "Gear": pd.to_numeric(car["n_gear"], errors="coerce")
                    .fillna(0)
                    .to_numpy("int8"),
                    "Throttle": pd.to_numeric(car["throttle"], errors="coerce")
                    .fillna(0)
                    .clip(0, 100)
                    .to_numpy("int8"),
                    "Brake": pd.to_numeric(car["brake"], errors="coerce").fillna(0).to_numpy() > 0,
                    "DRS": pd.to_numeric(car["drs"], errors="coerce").fillna(0).to_numpy()
                    >= races.DRS_OPEN,
                }
            )
        )
    if not traces:
        return pd.DataFrame(columns=races.TELEMETRY_COLUMNS)
    return pd.concat(traces, ignore_index=True)


# Corners sit at the distance where the session's fastest lap passed closest to them.
def corners_and_map(
    year: int,
    circuit_key: int,
    key: int,
    laps: pd.DataFrame,
    traces: pd.DataFrame,
    numbers: dict[str, int],
    t0: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty = (
        pd.DataFrame(columns=races.CORNER_COLUMNS),
        pd.DataFrame(columns=races.TRACK_COLUMNS),
        pd.DataFrame(columns=races.MAP_CORNER_COLUMNS),
    )
    circuit = get_circuit(year=year, circuit_key=int(circuit_key))
    timed = laps[laps["IsPersonalBest"]].dropna(subset=["LapTimeSeconds"])
    if not circuit or not circuit.get("x") or timed.empty or traces.empty:
        return empty
    lap = timed.loc[timed["LapTimeSeconds"].idxmin()]
    trace = traces[traces["Driver"] == lap["Driver"]]
    start = t0 + lap["LapStartTime"]
    where = fetch(
        "location",
        session_key=key,
        driver_number=numbers[lap["Driver"]],
        **{"date>": start.isoformat(), "date<": (start + lap["LapTime"]).isoformat()},
    )
    if where.empty or trace.empty:
        return empty
    seconds = (utc(where["date"]) - start).dt.total_seconds().to_numpy()
    distance = np.interp(seconds, trace["Time"].to_numpy(), trace["Distance"].to_numpy())
    points = where[["x", "y"]].to_numpy(dtype=float)
    rows = []
    for corner in circuit.get("corners", []):
        marker = np.array([corner["trackPosition"]["x"], corner["trackPosition"]["y"]])
        nearest = int(np.argmin(np.hypot(*(points - marker).T)))
        rows.append((int(corner["number"]), corner.get("letter") or "", float(distance[nearest])))
    corners = pd.DataFrame(rows, columns=races.CORNER_COLUMNS)
    track, map_corners = races.track_map(circuit, corners)
    return corners, track, map_corners


def driver_styles(drivers: pd.DataFrame, order: list[str]) -> races.Styles:
    styles, seen = {}, {}
    rank = {driver: index for index, driver in enumerate(order)}
    ranked = drivers.assign(Rank=drivers["name_acronym"].map(rank)).sort_values("Rank")
    for driver, team, colour in zip(
        ranked["name_acronym"], ranked["team_name"], ranked["team_colour"], strict=True
    ):
        if not colour or pd.isna(colour):
            continue
        styles[driver] = {
            "color": f"#{str(colour).lower()}",
            "linestyle": "solid" if not seen.get(team) else "dashed",
        }
        seen[team] = True
    return styles


def load_session(year: int, event: str, session: str = races.RACE) -> races.Race:
    found = find_session(year, event, session)
    key, t0 = int(found["session_key"]), found["Start"]
    drivers = fetch("drivers", session_key=key).drop_duplicates("driver_number")
    raw = fetch("laps", session_key=key)
    name = f"{year} {event}" if session == races.RACE else f"{year} {event} {session.lower()}"
    if raw.empty or drivers.empty:
        raise races.RaceDataUnavailable(f"OpenF1 has no laps for the {name} yet.")
    messages = fetch("race_control", session_key=key)
    if not messages.empty:
        messages = messages.assign(Date=utc(messages["date"]))
    else:
        messages = pd.DataFrame(
            columns=["Date", "message", "flag", "scope", "sector", "qualifying_phase"]
        )
    laps = lap_table(
        raw,
        drivers,
        fetch("stints", session_key=key),
        fetch("pit", session_key=key),
        messages,
        t0,
        qualifying=session in races.QUALIFYING,
        race=session in races.RACES,
    )
    names = drivers.set_index("driver_number")["name_acronym"]
    numbers = {acronym: int(number) for number, acronym in names.items()}
    results = fetch("session_result", session_key=key)
    official = official_order(year, int(found["RoundNumber"]), session)
    order = classification(official, results, names, laps)
    if session in races.PRACTICE:
        order = races.fastest_first(order, laps)
        traces = pd.DataFrame(columns=races.TELEMETRY_COLUMNS)
    else:
        traces = telemetry(key, laps, numbers, t0)
    corners, track, map_corners = (
        corners_and_map(year, found["circuit_key"], key, laps, traces, numbers, t0)
        if session == races.RACE
        else (
            pd.DataFrame(columns=races.CORNER_COLUMNS),
            pd.DataFrame(columns=races.TRACK_COLUMNS),
            pd.DataFrame(columns=races.MAP_CORNER_COLUMNS),
        )
    )
    total = results["number_of_laps"].max() if "number_of_laps" in results else np.nan
    return races.Race(
        year=year,
        round_number=int(found["RoundNumber"]),
        event=event,
        total_laps=int(total if pd.notna(total) else laps["LapNumber"].max()),
        order=order,
        laps=laps,
        telemetry=traces,
        corners=corners,
        driver_styles=driver_styles(drivers, order),
        compound_colors=COMPOUND_COLORS,
        track=track,
        map_corners=map_corners,
        weather=_weather(key, t0),
        messages=_messages(messages, t0),
        radio=_radio(key, names, t0),
        session=session,
    )


def _weather(key: int, t0: pd.Timestamp) -> pd.DataFrame:
    weather = fetch("weather", session_key=key)
    if weather.empty:
        return pd.DataFrame(columns=races.WEATHER_COLUMNS)
    return pd.DataFrame(
        {
            "Time": (utc(weather["date"]) - t0).dt.total_seconds().to_numpy(),
            "AirTemp": weather["air_temperature"].to_numpy(dtype=float),
            "TrackTemp": weather["track_temperature"].to_numpy(dtype=float),
            "Humidity": weather["humidity"].to_numpy(dtype=float),
            "Rainfall": weather["rainfall"].fillna(0).astype(bool).to_numpy(),
            "WindSpeed": weather["wind_speed"].to_numpy(dtype=float),
        }
    )


def _messages(messages: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    if messages.empty:
        return pd.DataFrame(columns=races.MESSAGE_COLUMNS)
    return pd.DataFrame(
        {
            "Time": (messages["Date"] - t0).dt.total_seconds().to_numpy(),
            "Lap": pd.to_numeric(messages["lap_number"], errors="coerce").to_numpy(),
            "Category": messages["category"].astype(str).to_numpy(),
            "Flag": messages["flag"].fillna("").astype(str).to_numpy(),
            "Message": messages["message"].astype(str).to_numpy(),
        }
    )


def _radio(key: int, names: pd.Series, t0: pd.Timestamp) -> pd.DataFrame:
    radio = fetch("team_radio", session_key=key)
    if radio.empty:
        return pd.DataFrame(columns=races.RADIO_COLUMNS)
    return pd.DataFrame(
        {
            "Time": (utc(radio["date"]) - t0).dt.total_seconds().to_numpy(),
            "Driver": radio["driver_number"].map(names).fillna(radio["driver_number"].astype(str)),
            "Url": radio["recording_url"].to_numpy(),
        }
    ).sort_values("Time", ignore_index=True)
