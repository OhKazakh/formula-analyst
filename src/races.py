from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import requests
from fastf1 import _api as live_timing
from fastf1.core import Session
from fastf1.exceptions import DataNotLoadedError
from fastf1.mvapi.api import get_circuit

from src.analysis import fastest_lap
from src.charts import Styles

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data"

OFFLINE_ENV = "FORMULA_ANALYST_OFFLINE"
LIVE_TIMING = "https://livetiming.formula1.com"
LIVE_TIMING_PROBE = f"{LIVE_TIMING}/static/StreamingStatus.json"
# FastF1 quietly falls back to the live timing index when its own schedule can't be fetched,
# and that index numbers rounds differently (it counts the cancelled 2023 Imola race).
SCHEDULE_BACKEND = "fastf1"

LAP_COLUMNS = [
    "Driver",
    "Team",
    "LapNumber",
    "LapTime",
    "LapTimeSeconds",
    "Stint",
    "Compound",
    "TyreLife",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    "IsAccurate",
    "IsPersonalBest",
    "LapStartTime",
    "Time",
    "Sector1",
    "Sector2",
    "Sector3",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
]
TELEMETRY_COLUMNS = [
    "Driver",
    "Distance",
    "Speed",
    "Time",
    "RPM",
    "Gear",
    "Throttle",
    "Brake",
    "DRS",
]
WEATHER_COLUMNS = ["Time", "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]
MESSAGE_COLUMNS = ["Time", "Lap", "Category", "Flag", "Message"]
RADIO_COLUMNS = ["Time", "Driver", "Url"]
CORNER_COLUMNS = ["Number", "Letter", "Distance"]
TRACK_COLUMNS = ["X", "Y"]
MAP_CORNER_COLUMNS = ["Label", "X", "Y"]
SAVED_COLUMNS = ["Year", "Round", "EventName", "Path"]
SESSION_COLUMNS = ["Year", "Round", "EventName", "Session", "Path"]
RACE = "Race"
SESSIONS = [
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Sprint Shootout",
    "Sprint Qualifying",
    "Sprint",
    "Qualifying",
    RACE,
]
PRACTICE = {"Practice 1", "Practice 2", "Practice 3"}
QUALIFYING = {"Qualifying", "Sprint Qualifying", "Sprint Shootout"}
RACES = {RACE, "Sprint"}
LAST_FRIDAY_QUALIFYING = 2023
COMPOUND_PLACEHOLDERS = ["nan", "None", ""]
CORNER_LABEL_OFFSET = 50.0
DRS_OPEN = 10
FORMAT_VERSION = 2


class RaceDataUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, eq=False)
class Race:
    year: int
    round_number: int
    event: str
    total_laps: int
    order: list[str]
    laps: pd.DataFrame
    telemetry: pd.DataFrame
    corners: pd.DataFrame
    driver_styles: Styles
    compound_colors: dict[str, str]
    track: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TRACK_COLUMNS))
    map_corners: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=MAP_CORNER_COLUMNS)
    )
    weather: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=WEATHER_COLUMNS))
    messages: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=MESSAGE_COLUMNS))
    radio: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=RADIO_COLUMNS))
    session: str = RACE

    @property
    def drivers(self) -> list[str]:
        raced = set(self.laps["Driver"])
        return [driver for driver in self.order if driver in raced]

    def trace(self, driver: str) -> pd.DataFrame:
        return self.telemetry[self.telemetry["Driver"] == driver]


def clean_compounds(compounds: pd.Series) -> pd.Series:
    return compounds.mask(compounds.isin(COMPOUND_PLACEHOLDERS))


def enable_cache(cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def live_timing_available(timeout: float = 5.0) -> bool:
    if os.environ.get(OFFLINE_ENV):
        return False
    try:
        response = requests.get(
            LIVE_TIMING_PROBE, headers={"User-Agent": "BestHTTP"}, timeout=timeout
        )
    except requests.RequestException:
        return False
    return response.ok


def event_schedule(year: int) -> pd.DataFrame:
    return fastf1.get_event_schedule(year, include_testing=False, backend=SCHEDULE_BACKEND)


def race_calendar(year: int) -> pd.DataFrame:
    schedule = event_schedule(year)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    finished = schedule[schedule["Session5DateUtc"] < now]
    return finished[["RoundNumber", "EventName", "Country", "EventDate"]].reset_index(drop=True)


def load_race(year: int, event: str, session: str = RACE, root: Path = DATA_DIR) -> Race:
    saved = saved_sessions(root)
    weekend = saved[(saved["Year"] == year) & (saved["EventName"] == event)]
    match = weekend[weekend["Session"] == session]
    saved_path = None if match.empty else match["Path"].iloc[0]
    race = read_race(saved_path) if saved_path else load_live_race(year, event, session)
    main = weekend[weekend["Session"] == RACE]
    if session == RACE or not race.track.empty or main.empty:
        return race
    # Other sessions borrow the circuit from the race, and its colours for anyone missing.
    circuit = read_race(main["Path"].iloc[0])
    return replace(
        race,
        corners=circuit.corners,
        track=circuit.track,
        map_corners=circuit.map_corners,
        driver_styles=circuit.driver_styles | race.driver_styles,
    )


def load_live_race(year: int, event: str, session: str = RACE) -> Race:
    loaded = fastf1.get_session(year, event, session, backend=SCHEDULE_BACKEND)
    loaded.load(weather=True, messages=True)
    name = f"{year} {event}" if session == RACE else f"{year} {event} {session.lower()}"
    try:
        laps = loaded.laps
    except DataNotLoadedError as exc:
        raise RaceDataUnavailable(f"Lap timing data for the {name} isn't available.") from exc
    if laps.empty:
        raise RaceDataUnavailable(f"No laps were recorded for the {name}.")
    return race_from_session(loaded, session)


def race_from_session(session: Session, name: str = RACE) -> Race:
    laps = _lap_table(session, name in QUALIFYING)
    order = _finishing_order(session, laps)
    if name in PRACTICE:
        order = fastest_first(order, laps)
    if name == RACE:
        corners = _corners(session)
        track, map_corners = _track_map(session, corners)
    else:
        corners = pd.DataFrame(columns=CORNER_COLUMNS)
        track = pd.DataFrame(columns=TRACK_COLUMNS)
        map_corners = pd.DataFrame(columns=MAP_CORNER_COLUMNS)
    return Race(
        year=int(session.event.year),
        round_number=int(session.event["RoundNumber"]),
        event=str(session.event["EventName"]),
        total_laps=int(session.total_laps or laps["LapNumber"].max()),
        order=order,
        laps=laps,
        telemetry=(
            pd.DataFrame(columns=TELEMETRY_COLUMNS)
            if name in PRACTICE
            else _fastest_lap_telemetry(session, laps)
        ),
        corners=corners,
        driver_styles=_driver_styles(session, order),
        compound_colors=dict(fastf1.plotting.get_compound_mapping(session)),
        track=track,
        map_corners=map_corners,
        weather=_weather(session),
        messages=_messages(session),
        radio=_team_radio(session),
        session=name,
    )


def _finishing_order(session: Session, laps: pd.DataFrame) -> list[str]:
    try:
        order = session.results["Abbreviation"].dropna().tolist()
    except (DataNotLoadedError, KeyError):
        order = []
    return order + [driver for driver in laps["Driver"].unique() if driver not in order]


# FastF1 has no classification for practice and lists drivers by car number instead.
def fastest_first(order: list[str], laps: pd.DataFrame) -> list[str]:
    best = laps[laps["IsPersonalBest"]].groupby("Driver")["LapTimeSeconds"].min()
    timed = best.dropna().sort_values(kind="stable").index.tolist()
    return timed + [driver for driver in order if driver not in timed]


# Driving through the pit lane without new tyres, behind the safety car or for a penalty,
# starts a new stint in the timing data. Stints here follow tyre sets instead.
def tyre_stints(laps: pd.DataFrame) -> pd.DataFrame:
    ordered = laps.sort_values(["Driver", "LapNumber"])
    by_driver = ordered.groupby("Driver")
    previous = by_driver[["Stint", "Compound", "TyreLife"]].shift()
    continued = (ordered["Compound"] == previous["Compound"]) & (
        ordered["TyreLife"] == previous["TyreLife"] + 1
    )
    new_set = previous["Stint"].isna() | ((ordered["Stint"] != previous["Stint"]) & ~continued)
    stints = new_set.astype(int).groupby(ordered["Driver"]).cumsum().astype(float)
    return laps.assign(Stint=stints.where(ordered["Stint"].notna()).reindex(laps.index))


def _lap_table(session: Session, qualifying: bool = False) -> pd.DataFrame:
    laps = pd.DataFrame(session.laps, copy=True)
    laps["Part"] = 0
    laps["Deleted"] = laps["Deleted"].fillna(False).astype(bool)
    if qualifying:
        for part, rows in enumerate(session.laps.split_qualifying_sessions(), start=1):
            if rows is not None:
                laps.loc[rows.index, "Part"] = part
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["Compound"] = clean_compounds(laps["Compound"])
    laps["IsAccurate"] = laps["IsAccurate"].astype(bool)
    laps["IsPersonalBest"] = laps["IsPersonalBest"].astype(bool)
    for number in (1, 2, 3):
        laps[f"Sector{number}"] = laps[f"Sector{number}Time"].dt.total_seconds()
    return tyre_stints(laps[LAP_COLUMNS + (["Part", "Deleted"] if qualifying else [])])


def _fastest_lap_telemetry(session: Session, laps: pd.DataFrame) -> pd.DataFrame:
    traces = []
    for driver in laps["Driver"].unique():
        lap = fastest_lap(laps, driver)
        if lap is None:
            continue
        try:
            car_data = session.laps.loc[lap.name].get_car_data().add_distance()
        except (DataNotLoadedError, KeyError, ValueError):
            continue
        traces.append(
            pd.DataFrame(
                {
                    "Driver": driver,
                    "Distance": car_data["Distance"].to_numpy(dtype="float32"),
                    "Speed": car_data["Speed"].to_numpy(dtype="int16"),
                    "Time": car_data["Time"].dt.total_seconds().to_numpy(dtype="float32"),
                    "RPM": car_data["RPM"].to_numpy(dtype="int16"),
                    "Gear": car_data["nGear"].to_numpy(dtype="int8"),
                    "Throttle": car_data["Throttle"].clip(0, 100).to_numpy(dtype="int8"),
                    "Brake": car_data["Brake"].astype(bool).to_numpy(),
                    "DRS": (car_data["DRS"] >= DRS_OPEN).to_numpy(),
                }
            )
        )
    if not traces:
        return pd.DataFrame(columns=TELEMETRY_COLUMNS)
    return pd.concat(traces, ignore_index=True)


def _weather(session: Session) -> pd.DataFrame:
    try:
        weather = session.weather_data
    except DataNotLoadedError:
        return pd.DataFrame(columns=WEATHER_COLUMNS)
    if weather is None or weather.empty:
        return pd.DataFrame(columns=WEATHER_COLUMNS)
    return pd.DataFrame(
        {
            "Time": weather["Time"].dt.total_seconds().to_numpy(),
            "AirTemp": weather["AirTemp"].to_numpy(dtype=float),
            "TrackTemp": weather["TrackTemp"].to_numpy(dtype=float),
            "Humidity": weather["Humidity"].to_numpy(dtype=float),
            "Rainfall": weather["Rainfall"].astype(bool).to_numpy(),
            "WindSpeed": weather["WindSpeed"].to_numpy(dtype=float),
        }
    )


def _session_seconds(session: Session, utc: pd.Series) -> np.ndarray:
    try:
        start = session.t0_date
    except DataNotLoadedError:
        start = None
    if start is None:
        return np.full(len(utc), np.nan)
    return (pd.Series(utc) - start).dt.total_seconds().to_numpy()


def _messages(session: Session) -> pd.DataFrame:
    try:
        messages = session.race_control_messages
    except DataNotLoadedError:
        return pd.DataFrame(columns=MESSAGE_COLUMNS)
    if messages is None or messages.empty:
        return pd.DataFrame(columns=MESSAGE_COLUMNS)
    return pd.DataFrame(
        {
            "Time": _session_seconds(session, messages["Time"]),
            "Lap": pd.to_numeric(messages["Lap"], errors="coerce").to_numpy(),
            "Category": messages["Category"].astype(str).to_numpy(),
            "Flag": messages["Flag"].fillna("").astype(str).to_numpy(),
            "Message": messages["Message"].astype(str).to_numpy(),
        }
    )


def team_radio_captures(stream: list) -> list[dict]:
    captures = []
    for _, payload in stream or []:
        items = payload.get("Captures", []) if isinstance(payload, dict) else []
        captures.extend(items.values() if isinstance(items, dict) else items)
    return captures


def _team_radio(session: Session) -> pd.DataFrame:
    try:
        captures = team_radio_captures(live_timing.fetch_page(session.api_path, "team_radio"))
        drivers = session.results.set_index("DriverNumber")["Abbreviation"]
    except Exception:
        return pd.DataFrame(columns=RADIO_COLUMNS)
    if not captures:
        return pd.DataFrame(columns=RADIO_COLUMNS)
    utc = pd.to_datetime(
        [capture["Utc"] for capture in captures], utc=True, format="ISO8601"
    ).tz_convert(None)
    radio = pd.DataFrame(
        {
            "Time": _session_seconds(session, utc),
            "Driver": [
                drivers.get(capture["RacingNumber"], capture["RacingNumber"])
                for capture in captures
            ],
            "Url": [f"{LIVE_TIMING}{session.api_path}{capture['Path']}" for capture in captures],
        }
    )
    return radio.sort_values("Time", ignore_index=True)


def _corners(session: Session) -> pd.DataFrame:
    try:
        corners = session.get_circuit_info().corners
    except Exception:
        return pd.DataFrame(columns=CORNER_COLUMNS)
    return corners[CORNER_COLUMNS].reset_index(drop=True)


def _track_map(session: Session, corners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        circuit_key = session.session_info["Meeting"]["Circuit"]["Key"]
        circuit = get_circuit(year=int(session.event.year), circuit_key=circuit_key)
        if circuit and circuit.get("x"):
            return track_map(circuit, corners)
    except Exception:
        pass
    return pd.DataFrame(columns=TRACK_COLUMNS), pd.DataFrame(columns=MAP_CORNER_COLUMNS)


def _rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def track_map(circuit: dict, corners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rotation = _rotation(float(circuit.get("rotation", 0.0)))
    outline = np.column_stack([circuit["x"], circuit["y"]]).astype(float) / 10
    if np.allclose(outline[0], outline[-1]):
        outline = outline[:-1]

    segments = np.hypot(*(np.roll(outline, -1, axis=0) - outline).T)
    starts = np.concatenate([[0.0], np.cumsum(segments)[:-1]])
    outline_lengths = {
        f"{c['number']}{c.get('letter', '')}": c["length"] / 10 for c in circuit.get("corners", [])
    }
    offsets = [
        outline_lengths[label] - distance
        for label, distance in zip(
            corners["Number"].astype(int).astype(str) + corners["Letter"].fillna(""),
            corners["Distance"],
            strict=True,
        )
        if label in outline_lengths
    ]
    if offsets:
        timing_line = float(np.median(offsets)) % segments.sum()
        outline = np.roll(outline, -int(np.searchsorted(starts, timing_line)), axis=0)

    track = pd.DataFrame(np.vstack([outline, outline[:1]]) @ rotation, columns=TRACK_COLUMNS)

    labels = []
    for corner in circuit.get("corners", []):
        position = np.array([corner["trackPosition"]["x"], corner["trackPosition"]["y"]]) / 10
        direction = np.array([1.0, 0.0]) @ _rotation(float(corner.get("angle", 0.0)))
        x, y = (position + CORNER_LABEL_OFFSET * direction) @ rotation
        labels.append((f"{corner['number']}{corner.get('letter', '')}", x, y))
    return track, pd.DataFrame(labels, columns=MAP_CORNER_COLUMNS)


def _driver_styles(session: Session, drivers: list[str]) -> Styles:
    styles = {}
    for driver in drivers:
        try:
            style = fastf1.plotting.get_driver_style(
                driver, ["color", "linestyle"], session, colormap="official"
            )
        # FastF1 trips over drivers listed without a team colour, like some first-practice
        # stand-ins, and then fails for everyone in the session.
        except (KeyError, ValueError, AttributeError):
            continue
        styles[driver] = {"color": style["color"], "linestyle": style["linestyle"]}
    return styles


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def race_dir(root: Path, year: int, round_number: int, event: str) -> Path:
    return root / str(year) / f"{round_number:02d}-{slugify(event)}"


# Until 2023, sprint weekends held qualifying on the Friday, straight after first practice.
def weekend_order(names: list[str], year: int) -> list[str]:
    order = list(SESSIONS)
    if "Sprint" in names and year <= LAST_FRIDAY_QUALIFYING:
        order.remove("Qualifying")
        order.insert(order.index("Practice 1") + 1, "Qualifying")
    return [name for name in order if name in names]


def session_dir(root: Path, year: int, round_number: int, event: str, session: str) -> Path:
    path = race_dir(root, year, round_number, event)
    return path if session == RACE else path / slugify(session)


def save_race(race: Race, root: Path = DATA_DIR) -> Path:
    path = session_dir(root, race.year, race.round_number, race.event, race.session)
    path.mkdir(parents=True, exist_ok=True)
    race.laps.to_parquet(path / "laps.parquet", index=False)
    race.telemetry.to_parquet(path / "telemetry.parquet", index=False)
    race.track.astype("float32").to_parquet(path / "track.parquet", index=False)
    race.weather.to_parquet(path / "weather.parquet", index=False)
    race.messages.to_parquet(path / "messages.parquet", index=False)
    race.radio.to_parquet(path / "radio.parquet", index=False)
    meta = {
        "format": FORMAT_VERSION,
        "year": race.year,
        "round": race.round_number,
        "event": race.event,
        "session": race.session,
        "total_laps": race.total_laps,
        "order": race.order,
        "corners": race.corners.to_dict(orient="records"),
        "map_corners": race.map_corners.round(1).to_dict(orient="records"),
        "driver_styles": race.driver_styles,
        "compound_colors": race.compound_colors,
    }
    (path / "race.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def read_optional(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)


def bundle_format(path: Path) -> int:
    return json.loads((path / "race.json").read_text()).get("format", 1)


def read_race(path: Path) -> Race:
    meta = json.loads((path / "race.json").read_text())
    track = read_optional(path / "track.parquet", TRACK_COLUMNS)
    return Race(
        year=meta["year"],
        round_number=meta["round"],
        event=meta["event"],
        total_laps=meta["total_laps"],
        order=meta["order"],
        laps=tyre_stints(pd.read_parquet(path / "laps.parquet")),
        telemetry=pd.read_parquet(path / "telemetry.parquet"),
        corners=pd.DataFrame(meta["corners"], columns=CORNER_COLUMNS),
        driver_styles=meta["driver_styles"],
        compound_colors=meta["compound_colors"],
        track=track,
        map_corners=pd.DataFrame(meta.get("map_corners", []), columns=MAP_CORNER_COLUMNS),
        weather=read_optional(path / "weather.parquet", WEATHER_COLUMNS),
        messages=read_optional(path / "messages.parquet", MESSAGE_COLUMNS),
        radio=read_optional(path / "radio.parquet", RADIO_COLUMNS),
        session=meta.get("session", RACE),
    )


# Streamlit Cloud reloads code after a push but keeps st.cache_data, so caches key on this.
def bundle_version(root: Path = DATA_DIR) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def saved_races(root: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for meta_path in root.glob("*/*/race.json"):
        meta = json.loads(meta_path.read_text())
        rows.append(
            {
                "Year": meta["year"],
                "Round": meta["round"],
                "EventName": meta["event"],
                "Path": meta_path.parent,
            }
        )
    saved = pd.DataFrame(rows, columns=SAVED_COLUMNS)
    return saved.sort_values(["Year", "Round"], ignore_index=True)


def saved_sessions(root: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for meta_path in [*root.glob("*/*/race.json"), *root.glob("*/*/*/race.json")]:
        meta = json.loads(meta_path.read_text())
        rows.append(
            {
                "Year": meta["year"],
                "Round": meta["round"],
                "EventName": meta["event"],
                "Session": meta.get("session", RACE),
                "Path": meta_path.parent,
            }
        )
    saved = pd.DataFrame(rows, columns=SESSION_COLUMNS)
    order = saved["Session"].map({name: index for index, name in enumerate(SESSIONS)})
    return (
        saved.assign(Order=order)
        .sort_values(["Year", "Round", "Order"])
        .drop(columns="Order")
        .reset_index(drop=True)
    )
