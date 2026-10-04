from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import requests
from fastf1.core import Session
from fastf1.exceptions import DataNotLoadedError
from fastf1.mvapi.api import get_circuit

from src.analysis import fastest_lap
from src.charts import Styles

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data"

OFFLINE_ENV = "FORMULA_ANALYST_OFFLINE"
LIVE_TIMING_PROBE = "https://livetiming.formula1.com/static/StreamingStatus.json"

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
]
TELEMETRY_COLUMNS = ["Driver", "Distance", "Speed"]
CORNER_COLUMNS = ["Number", "Letter", "Distance"]
TRACK_COLUMNS = ["X", "Y"]
MAP_CORNER_COLUMNS = ["Label", "X", "Y"]
SAVED_COLUMNS = ["Year", "Round", "EventName", "Path"]
COMPOUND_PLACEHOLDERS = ["nan", "None", ""]
CORNER_LABEL_OFFSET = 50.0


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


def race_calendar(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    finished = schedule[schedule["Session5DateUtc"] < now]
    return finished[["RoundNumber", "EventName", "Country", "EventDate"]].reset_index(drop=True)


def load_race(year: int, event: str, root: Path = DATA_DIR) -> Race:
    saved = saved_races(root)
    match = saved[(saved["Year"] == year) & (saved["EventName"] == event)]
    if not match.empty:
        return read_race(match["Path"].iloc[0])
    return load_live_race(year, event)


def load_live_race(year: int, event: str) -> Race:
    session = fastf1.get_session(year, event, "R")
    session.load(weather=False, messages=False)
    try:
        laps = session.laps
    except DataNotLoadedError as exc:
        raise RaceDataUnavailable(
            f"Lap timing data for the {year} {event} isn't available."
        ) from exc
    if laps.empty:
        raise RaceDataUnavailable(f"No laps were recorded for the {year} {event}.")
    return race_from_session(session)


def race_from_session(session: Session) -> Race:
    laps = _lap_table(session)
    order = _finishing_order(session, laps)
    corners = _corners(session)
    track, map_corners = _track_map(session, corners)
    return Race(
        year=int(session.event.year),
        round_number=int(session.event["RoundNumber"]),
        event=str(session.event["EventName"]),
        total_laps=int(session.total_laps or laps["LapNumber"].max()),
        order=order,
        laps=laps,
        telemetry=_fastest_lap_telemetry(session, laps),
        corners=corners,
        driver_styles=_driver_styles(session, order),
        compound_colors=dict(fastf1.plotting.get_compound_mapping(session)),
        track=track,
        map_corners=map_corners,
    )


def _finishing_order(session: Session, laps: pd.DataFrame) -> list[str]:
    try:
        order = session.results["Abbreviation"].dropna().tolist()
    except (DataNotLoadedError, KeyError):
        order = []
    return order + [driver for driver in laps["Driver"].unique() if driver not in order]


def _lap_table(session: Session) -> pd.DataFrame:
    laps = pd.DataFrame(session.laps, copy=True)
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["Compound"] = clean_compounds(laps["Compound"])
    laps["IsAccurate"] = laps["IsAccurate"].astype(bool)
    laps["IsPersonalBest"] = laps["IsPersonalBest"].astype(bool)
    return laps[LAP_COLUMNS]


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
                    "Distance": car_data["Distance"].to_numpy(),
                    "Speed": car_data["Speed"].to_numpy(),
                }
            )
        )
    if not traces:
        return pd.DataFrame(columns=TELEMETRY_COLUMNS)
    return pd.concat(traces, ignore_index=True)


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
        except (KeyError, ValueError):
            continue
        styles[driver] = {"color": style["color"], "linestyle": style["linestyle"]}
    return styles


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def race_dir(root: Path, year: int, round_number: int, event: str) -> Path:
    return root / str(year) / f"{round_number:02d}-{slugify(event)}"


def save_race(race: Race, root: Path = DATA_DIR) -> Path:
    path = race_dir(root, race.year, race.round_number, race.event)
    path.mkdir(parents=True, exist_ok=True)
    race.laps.to_parquet(path / "laps.parquet", index=False)
    race.telemetry.to_parquet(path / "telemetry.parquet", index=False)
    race.track.astype("float32").to_parquet(path / "track.parquet", index=False)
    meta = {
        "year": race.year,
        "round": race.round_number,
        "event": race.event,
        "total_laps": race.total_laps,
        "order": race.order,
        "corners": race.corners.to_dict(orient="records"),
        "map_corners": race.map_corners.round(1).to_dict(orient="records"),
        "driver_styles": race.driver_styles,
        "compound_colors": race.compound_colors,
    }
    (path / "race.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def read_race(path: Path) -> Race:
    meta = json.loads((path / "race.json").read_text())
    track_path = path / "track.parquet"
    track = (
        pd.read_parquet(track_path) if track_path.exists() else pd.DataFrame(columns=TRACK_COLUMNS)
    )
    return Race(
        year=meta["year"],
        round_number=meta["round"],
        event=meta["event"],
        total_laps=meta["total_laps"],
        order=meta["order"],
        laps=pd.read_parquet(path / "laps.parquet"),
        telemetry=pd.read_parquet(path / "telemetry.parquet"),
        corners=pd.DataFrame(meta["corners"], columns=CORNER_COLUMNS),
        driver_styles=meta["driver_styles"],
        compound_colors=meta["compound_colors"],
        track=track,
        map_corners=pd.DataFrame(meta.get("map_corners", []), columns=MAP_CORNER_COLUMNS),
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
