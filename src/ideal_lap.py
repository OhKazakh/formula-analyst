"""Two laps nobody drove: the session's fastest sectors joined up, and a computer lap per car.

The computer lap is a quasi-steady-state lap simulation. Each car's corner speeds,
acceleration and braking come from its own fastest laps in the session, the corners from the
track outline. The outline is rough, so the result is tuned on dry qualifying at the same
circuit in other years, and left out when those years disagree or the session was wet.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import races, sectors

STEP = 5.0  # metres between points of the simulated lap
WINDOW = 40.0  # metres of track over which a corner's radius is measured
CORNER_REACH = 80.0  # metres either side of a corner to look for a car's lowest speed
BEND_REACH = 60.0
APEX_DIP = 0.8  # a slow corner's apex is this much slower than anything 80 m either side
APEX_SEARCH = 200.0  # metres to look for the same slow corner on another car's lap
MIN_TRACE = 50  # samples; fewer than this isn't a lap
GRIP_LIMIT = 0.04  # cars' corner speeds rarely differ by more; beyond it a lap is misread
CORNER_RADII = (10.0, 600.0)  # metres; outside this a "corner" is noise or a straight
BIN = 20 / 3.6  # m/s, for the acceleration and braking tables
FULL_THROTTLE = 98
SAMPLE_GAP = (0.05, 0.5)  # seconds; longer gaps in the feed hide what the car did
SAME_LAYOUT = 0.02  # lap lengths this close count as the same layout of a circuit
TUNING_YEARS = (2, 3)  # at least and at most this many other years tune a circuit
TUNING_SPREAD = 0.03  # years further apart than this mean the outline can't be trusted
PLAUSIBLE = 0.05  # further than this from the circuit's usual lap times, the data was misread
WET_TYRES = {"INTERMEDIATE", "WET"}
LEVELS_FILE = "computer-laps.parquet"
LEVEL_COLUMNS = ["Year", "EventName", "Length", "Wet", "Level", "Lap"]
SIMULATED_COLUMNS = ["Team", "Simulated", "Best"]

Table = tuple[np.ndarray, np.ndarray]


def perfect_lap(laps: pd.DataFrame) -> pd.DataFrame:
    """The fastest time through each sector, from laps the stewards didn't delete."""
    if "Deleted" in laps:
        laps = laps[~laps["Deleted"].fillna(False).astype(bool)]
    leaders = sectors.sector_leaders(laps)
    return leaders if len(leaders) == len(sectors.SECTORS) else leaders.iloc[0:0]


def resample(points: np.ndarray, step: float = STEP) -> np.ndarray:
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    distance = np.arange(0.0, along[-1], step)
    return np.column_stack(
        [np.interp(distance, along, points[:, 0]), np.interp(distance, along, points[:, 1])]
    )


def radii(points: np.ndarray, step: float = STEP, window: float = WINDOW) -> np.ndarray:
    """Radius from how far the direction turns over `window` metres around each point."""
    k = max(1, round(window / step / 2))
    heading = np.unwrap(np.arctan2(*np.diff(points, axis=0, append=points[:1])[:, ::-1].T))
    # Pad a lap either side so the start and finish turn like any other point.
    turn_per_lap = heading[-1] - heading[0] + (heading[1] - heading[0])
    padded = np.concatenate([heading - turn_per_lap, heading, heading + turn_per_lap])
    middle = np.arange(len(heading)) + len(heading)
    turned = np.abs(padded[middle + k] - padded[middle - k])
    with np.errstate(divide="ignore"):
        return np.where(turned > 1e-9, 2 * k * step / turned, np.inf)


def rates(trace: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Speed (m/s) against acceleration while flat out, and against deceleration on the brakes."""
    trace = trace.dropna(subset=["Time", "Speed"]).sort_values("Time")
    speed = trace["Speed"].to_numpy(dtype=float) / 3.6
    dt = np.diff(trace["Time"].to_numpy(dtype=float))
    steady = (dt > SAMPLE_GAP[0]) & (dt < SAMPLE_GAP[1])
    rate = np.diff(speed) / np.where(steady, dt, 1.0)
    middle = (speed[1:] + speed[:-1]) / 2
    brake = trace["Brake"].astype(bool).to_numpy()
    flat = (trace["Throttle"].to_numpy() >= FULL_THROTTLE) & ~brake
    # A sample counts when the car was flat out, or braking, at both its ends.
    accelerating = flat[1:] & flat[:-1] & steady
    braking = brake[1:] & brake[:-1] & steady
    return (
        np.column_stack([middle[accelerating], rate[accelerating]]),
        np.column_stack([middle[braking], -rate[braking]]),
    )


def rate_table(samples: np.ndarray, percentile: float, minimum: int) -> Table:
    """A high percentile of the rate in each speed bin: what the car can do, not what it did."""
    speeds, values = [], []
    for low in np.arange(0.0, 100.0, BIN):
        inside = samples[(samples[:, 0] >= low) & (samples[:, 0] < low + BIN)]
        if len(inside) >= minimum:
            speeds.append(low + BIN / 2)
            values.append(max(float(np.percentile(inside[:, 1], percentile)), 0.0))
    return np.array(speeds), np.array(values)


def scaled(everyone: Table, own: Table) -> Table:
    """Everyone's table, at this car's level where the two overlap.

    One car's two laps leave gaps and noise in its own table, at low speed most of all.
    """
    shared = [
        (value, everyone[1][match[0]])
        for speed, value in zip(*own, strict=True)
        if len(match := np.flatnonzero(np.isclose(everyone[0], speed)))
        and everyone[1][match[0]] > 0
    ]
    factor = float(np.median([mine / theirs for mine, theirs in shared])) if shared else 1.0
    return everyone[0], everyone[1] * factor


def simulate(limit: np.ndarray, step: float, accel: Table, brake: Table) -> np.ndarray:
    """Speed round a closed lap, as fast as the corner limits, acceleration and braking allow.

    The classic two passes: accelerate out of every corner, then brake back from the next.
    Both start at the slowest point, which ends the lap too, so the car brakes for it.
    """
    start = int(np.argmin(limit))
    limit = np.append(np.roll(limit, -start), limit[start])
    forward = limit.copy()
    for i in range(len(limit) - 1):
        gain = 2 * np.interp(forward[i], *accel) * step
        forward[i + 1] = min(limit[i + 1], math.sqrt(forward[i] ** 2 + gain))
    backward = forward.copy()
    for i in range(len(limit) - 2, -1, -1):
        loss = 2 * np.interp(backward[i + 1], *brake) * step
        backward[i] = min(forward[i], math.sqrt(backward[i + 1] ** 2 + loss))
    return np.roll(backward[:-1], start)


def lap_seconds(speed: np.ndarray, step: float) -> float:
    return float(np.sum(step / ((speed + np.roll(speed, -1)) / 2)))


def lap_length(telemetry: pd.DataFrame) -> float:
    return float(telemetry.groupby("Driver")["Distance"].max().median())


def apexes(distance: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """The slowest point of every slow corner, which no two cars can mistake for another."""
    found = []
    for i in range(len(speed)):
        around = speed[np.abs(distance - distance[i]) <= CORNER_REACH]
        if speed[i] <= around.min() and speed[i] < APEX_DIP * around.max():
            found.append(i)
    return np.array(found, dtype=int)


def aligned(trace: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    """The trace's distance, stretched to meet the reference lap at every slow corner.

    Distance is added up from each car's own speed, so it drifts by a different amount for
    every car during a lap: at Monza in 2026 by over 150 m between two cars.
    """
    distance = trace["Distance"].to_numpy(dtype=float)
    speed = trace["Speed"].to_numpy(dtype=float)
    theirs = reference["Distance"].to_numpy(dtype=float)
    scale = theirs.max() / distance.max()
    mine = distance[apexes(distance, speed)] * scale
    targets = theirs[apexes(theirs, reference["Speed"].to_numpy(dtype=float))]
    anchors = [(0.0, 0.0)]
    for target in targets if len(mine) else []:
        nearest = mine[np.argmin(np.abs(mine - target))]
        if abs(nearest - target) <= APEX_SEARCH and nearest > anchors[-1][0]:
            anchors.append((nearest, target))
    anchors.append((distance.max() * scale, theirs.max()))
    points, matched = np.array(anchors).T
    keep = np.concatenate([[True], (np.diff(matched) > 0) & (np.diff(points) > 0)])
    return np.interp(distance * scale, points[keep], matched[keep])


def _corner_samples(
    traces: dict[str, pd.DataFrame], radius: np.ndarray, along: np.ndarray, corners: pd.Series
) -> pd.DataFrame:
    """Each corner's tightest radius on the outline and each car's lowest speed there, as logs."""
    lengths = {driver: trace["Distance"].max() for driver, trace in traces.items()}
    typical = sorted(lengths, key=lengths.get)[len(lengths) // 2]
    rows = []
    for driver, trace in traces.items():
        distance = aligned(trace, traces[typical])
        speed = trace["Speed"].to_numpy(dtype=float)
        for corner in corners:
            near = speed[np.abs(distance - corner) <= CORNER_REACH]
            bend = radius[np.abs(along - corner) <= BEND_REACH]
            if not len(near) or not len(bend) or not CORNER_RADII[0] < bend.min() < CORNER_RADII[1]:
                continue
            rows.append((driver, math.log(bend.min()), math.log(near.min())))
    return pd.DataFrame(rows, columns=["Driver", "Radius", "Speed"])


def simulated_laps(
    laps: pd.DataFrame, telemetry: pd.DataFrame, track: pd.DataFrame, corners: pd.DataFrame
) -> pd.DataFrame:
    """Each car's simulated lap next to its best real one, both in seconds."""
    empty = pd.DataFrame(columns=SIMULATED_COLUMNS)
    if track.empty or corners.empty or telemetry.empty:
        return empty
    timed = laps.dropna(subset=["LapTimeSeconds"])
    if "Deleted" in timed:
        timed = timed[~timed["Deleted"].fillna(False).astype(bool)]
    best = timed.groupby("Driver")["LapTimeSeconds"].min()
    teams = laps.groupby("Driver")["Team"].first()
    traces = {}
    for driver, trace in telemetry.groupby("Driver"):
        # A stopped car or a gap in the feed leaves zero or missing readings.
        trace = trace.dropna(subset=["Distance", "Speed", "Time"])
        trace = trace[trace["Speed"] > 0].sort_values("Distance")
        if driver in best.index and len(trace) >= MIN_TRACE:
            traces[driver] = trace
    if not traces:
        return empty
    points = resample(track[["X", "Y"]].to_numpy(dtype=float))
    radius = radii(points)
    length = lap_length(telemetry)
    # The outline and the lap differ a little in length, so they're lined up in proportion.
    step = length / len(points)
    along = np.arange(len(points)) * step

    samples = _corner_samples(traces, radius, along, corners["Distance"])
    if len(samples) < 5:
        return empty
    slope, intercept = np.polyfit(samples["Radius"], samples["Speed"], 1)
    if slope <= 0:
        return empty
    samples["Team"] = samples["Driver"].map(teams)
    grip = (samples["Speed"] - (slope * samples["Radius"] + intercept)).groupby(samples["Team"])

    accelerating, braking = zip(*(rates(trace) for trace in traces.values()), strict=True)
    everyone = rate_table(np.vstack(accelerating), 75, 8)
    brake = rate_table(np.vstack(braking), 90, 8)
    if len(everyone[0]) < 2 or len(brake[0]) < 2:
        return empty

    cars: dict[str, list[str]] = {}
    for driver in traces:
        cars.setdefault(teams[driver], []).append(driver)
    rows = []
    for team, drivers in cars.items():
        mine = [traces[driver] for driver in drivers]
        accel = scaled(everyone, rate_table(np.vstack([rates(t)[0] for t in mine]), 75, 6))
        # Flat out, the car stops gaining speed at the fastest it went.
        top = max(float(trace["Speed"].max()) for trace in mine) / 3.6
        below = accel[0] < top
        accel = (np.append(accel[0][below], top), np.append(accel[1][below], 0.0))
        offset = float(np.clip(grip.mean().get(team, 0.0), -GRIP_LIMIT, GRIP_LIMIT))
        with np.errstate(divide="ignore", over="ignore"):
            limit = np.exp(intercept + offset) * radius**slope / 3.6
        speed = simulate(np.minimum(limit, top), step, accel, brake)
        rows.append(
            {
                "Team": team,
                "Simulated": lap_seconds(speed, step),
                "Best": float(best[drivers].min()),
            }
        )
    return pd.DataFrame(rows, columns=SIMULATED_COLUMNS)


def wet(weather: pd.DataFrame, laps: pd.DataFrame) -> bool:
    """Rain, or anyone on wet-weather tyres: a track can stay wet after the rain stops."""
    rained = "Rainfall" in weather and bool(weather["Rainfall"].astype(bool).any())
    return rained or bool(laps["Compound"].isin(WET_TYRES).any())


def track_level(
    levels: pd.DataFrame, event: str, year: int, length: float, simulated: float
) -> float | None:
    """How much slower real laps were than simulated ones in other dry years at this circuit.

    None when too few years match or they disagree, which means the outline is too rough to
    trust, or when the tuned lap lands far from those years' real ones, which means this
    session's data was misread. `simulated` is the session's typical simulated lap.
    """
    fewest, most = TUNING_YEARS
    dry = ~levels["Wet"].astype(bool)
    other = levels[(levels["Year"] != year) & (levels["EventName"] == event) & dry]
    other = other[(other["Length"] / length - 1).abs() <= SAME_LAYOUT]
    # The nearest seasons first, the earlier one on a tie.
    distance = (other["Year"] - year).abs() * 2 + (other["Year"] > year)
    nearest = other.assign(Distance=distance).nsmallest(most, "Distance")
    if len(nearest) < fewest or np.ptp(nearest["Level"]) > TUNING_SPREAD:
        return None
    level = float(nearest["Level"].median())
    if abs(simulated * level / nearest["Lap"].median() - 1) > PLAUSIBLE:
        return None
    return level


def computer_laps(simulated: pd.DataFrame, level: float) -> pd.Series:
    return simulated.set_index("Team")["Simulated"] * level


def saved_levels(root: Path = races.DATA_DIR) -> pd.DataFrame:
    path = root / LEVELS_FILE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=LEVEL_COLUMNS)


def build(root: Path = races.DATA_DIR) -> Path:
    """Every bundled qualifying session's simulated laps against its real ones."""
    saved = races.saved_sessions(root)
    rows = []
    for row in saved[saved["Session"] == "Qualifying"].itertuples():
        race = races.load_race(row.Year, row.EventName, "Qualifying", root)
        simulated = simulated_laps(race.laps, race.telemetry, race.track, race.corners)
        if simulated.empty:
            continue
        rows.append(
            {
                "Year": row.Year,
                "EventName": row.EventName,
                "Length": lap_length(race.telemetry),
                "Wet": wet(race.weather, race.laps),
                "Level": float((simulated["Best"] / simulated["Simulated"]).median()),
                "Lap": float(simulated["Best"].median()),
            }
        )
    path = root / LEVELS_FILE
    pd.DataFrame(rows, columns=LEVEL_COLUMNS).to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    print(build())
