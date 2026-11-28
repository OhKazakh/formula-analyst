"""How fast could the Aston Martin AMR25 lap Inje Speedium?

Formula 1 raced at the Korea International Circuit (KIC) from 2010 to 2013. Korean racing classes
hold lap records at both KIC and Inje, so they bridge an F1 lap at KIC to an F1 lap at Inje. An era
factor then takes the 2013 car to the 2025 one.

Run from the repository root: python research/inje-amr25/predict.py
It prints every step and writes results.json and chart.png next to this file. The only network
call fetches the circuit geometry from OpenStreetMap once and keeps it in inje_osm.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Korean class lap records on the KIC full course and at Inje Speedium (full course, 3.908 km).
# KIC: https://wvzine.com/?p=3845  Inje: https://wvzine.com/?p=3835
BRIDGE = {
    "Super 6000": ("2:10.540", "1:35.239"),
    "GTA": ("2:20.564", "1:42.713"),
    "M": ("2:25.873", "1:47.062"),
    "Alpine": ("2:30.856", "1:49.514"),
    "GTB": ("2:38.060", "1:54.562"),
    "Prius PHEV": ("2:45.974", "1:58.565"),
}
RATIO_CLASS = "Super 6000"

# 2013 Korean Grand Prix at KIC, both by Sebastian Vettel.
# https://en.wikipedia.org/wiki/2013_Korean_Grand_Prix
F1_KIC_2013 = {"qualifying": "1:37.202", "race": "1:41.380"}

# Era factor from Suzuka poles, a circuit that kept its layout: Webber 2013, Verstappen 2025.
# https://en.wikipedia.org/wiki/2013_Japanese_Grand_Prix
# https://en.wikipedia.org/wiki/2025_Japanese_Grand_Prix
SUZUKA_POLE_2013 = "1:30.915"
SUZUKA_POLE_2025 = "1:26.983"

# Cross-checks. Inje race lap records, both from the 2014 3 Hours of Inje:
# https://en.wikipedia.org/wiki/Inje_Speedium
INJE_LMP2 = "1:25.148"  # Ho-Pin Tung, Morgan LMP2
INJE_GT3 = "1:32.888"  # Han-Chen Chen, BMW Z4 GT3
# GT3 also raced at KIC, so the same circuit gives the F1 to GT3 ratio. Race laps on both sides.
# https://en.wikipedia.org/wiki/Korea_International_Circuit
KIC_GT3 = "2:07.380"  # Adderly Fong, Bentley Continental GT3, 2016 GT Asia
# LMP2 never raced at KIC. Shanghai had F1 and the WEC in 2013, with LMP2 cars of the same
# generation as the 2014 Morgan. Qualifying on both sides; the WEC time averages two drivers.
# https://en.wikipedia.org/wiki/2013_Chinese_Grand_Prix
# https://en.wikipedia.org/wiki/2013_6_Hours_of_Shanghai
SHANGHAI_F1_POLE_2013 = "1:34.484"  # Lewis Hamilton
SHANGHAI_LMP2_POLE_2013 = "1:55.423"  # G-Drive Racing
DISAGREEMENT = 3.0  # seconds outside the main range that would hold up publishing

# A demo run isn't flat out. This is a judgement, not a calculation.
DEMO_SLOWER = (8.0, 15.0)

# Inje Speedium in OpenStreetMap: the Full Course raceway and the pit lane beside the main
# straight. No reliable source gives the main straight's length, so it comes from this geometry.
# https://www.openstreetmap.org/way/651693293  https://www.openstreetmap.org/way/677355827
INJE_LENGTH = 3908.0
OSM_COURSE = 651693293
OSM_PIT_LANE = 677355827
OSM_FILE = HERE / "inje_osm.json"
OVERPASS = "https://overpass-api.de/api/interpreter"
STEP = 5.0  # metres between resampled points
WINDOW = 40.0  # metres of track over which a corner's radius is measured
STRAIGHT_RADIUS = 400.0
# The car brakes for Turn 1 somewhere before the straight ends.
BRAKING_ALLOWANCE = 100.0

# The AMR25's two cars, Alonso (14) and Stroll (18), in the app's bundled 2025 data.
AMR25 = {"ALO": 14, "STR": 18}
FULL_THROTTLE = 90
BIN = 20.0  # km/h, for the acceleration and braking tables
MIN_SAMPLES = 30


def seconds(text: str) -> float:
    minutes, rest = text.split(":")
    return int(minutes) * 60 + float(rest)


def lap_time(value: float, decimals: int = 3) -> str:
    minutes, rest = divmod(round(value, decimals), 60)
    width = 3 + decimals if decimals else 2
    return f"{int(minutes)}:{rest:0{width}.{decimals}f}"


def fit_bridge(bridge: dict[str, tuple[str, str]]) -> tuple[float, float, float]:
    """Least-squares line Inje = a * KIC + b through the class records, and its correlation."""
    kic = np.array([seconds(pair[0]) for pair in bridge.values()])
    inje = np.array([seconds(pair[1]) for pair in bridge.values()])
    a, b = np.polyfit(kic, inje, 1)
    return float(a), float(b), float(np.corrcoef(kic, inje)[0, 1])


def class_ratio(bridge: dict[str, tuple[str, str]], name: str = RATIO_CLASS) -> float:
    kic, inje = bridge[name]
    return seconds(inje) / seconds(kic)


def era_factor(old: str = SUZUKA_POLE_2013, new: str = SUZUKA_POLE_2025) -> float:
    return seconds(new) / seconds(old)


def predict(kic: float, fit: tuple[float, float, float], ratio: float, era: float) -> dict:
    """The 2013 car's KIC lap, bridged to Inje, then made a 2025 car."""
    a, b, _ = fit
    return {"linear": (a * kic + b) * era, "ratio": kic * ratio * era}


def outside(value: float, low: float, high: float) -> float:
    return max(low - value, value - high, 0.0)


def cross_checks(era: float, ranges: dict[str, tuple[float, float]]) -> list[dict]:
    gt3_ratio = seconds(F1_KIC_2013["race"]) / seconds(KIC_GT3)
    lmp2_ratio = seconds(SHANGHAI_F1_POLE_2013) / seconds(SHANGHAI_LMP2_POLE_2013)
    checks = [
        {
            "name": "GT3",
            "pace": "race",
            "circuit": "Korea International Circuit, race laps",
            "ratio": gt3_ratio,
            "estimate": seconds(INJE_GT3) * gt3_ratio * era,
        },
        {
            "name": "LMP2",
            "pace": "qualifying",
            "circuit": "Shanghai 2013, qualifying",
            "ratio": lmp2_ratio,
            "estimate": seconds(INJE_LMP2) * lmp2_ratio * era,
        },
    ]
    for check in checks:
        check["off_by"] = outside(check["estimate"], *ranges[check["pace"]])
    return checks


def describe(check: dict, pace_range: tuple[float, float]) -> str:
    if not check["off_by"]:
        return "inside the range"
    side = "faster" if check["estimate"] < pace_range[0] else "slower"
    digits = 2 if check["off_by"] < 0.1 else 1
    return f"{check['off_by']:.{digits}f} s {side} than the range"


# Circuit geometry


def osm_geometry() -> dict:
    if OSM_FILE.exists():
        return json.loads(OSM_FILE.read_text())
    query = f"[out:json][timeout:60];(way({OSM_COURSE});way({OSM_PIT_LANE}););out tags geom;"
    response = requests.post(
        OVERPASS,
        data={"data": query},
        headers={"User-Agent": "formula-analyst-research"},
        timeout=90,
    )
    response.raise_for_status()
    elements = {element["id"]: element for element in response.json()["elements"]}
    geometry = {
        "source": "© OpenStreetMap contributors, ODbL 1.0",
        "course": elements[OSM_COURSE]["geometry"],
        "pit_lane": elements[OSM_PIT_LANE]["geometry"],
    }
    OSM_FILE.write_text(json.dumps(geometry, indent=1) + "\n")
    return geometry


def metres(points: list[dict], origin: dict) -> np.ndarray:
    """Latitude and longitude as metres east and north of an origin."""
    radius = 6_371_000.0
    lat0, lon0 = math.radians(origin["lat"]), math.radians(origin["lon"])
    return np.array(
        [
            [
                (math.radians(p["lon"]) - lon0) * math.cos(lat0) * radius,
                (math.radians(p["lat"]) - lat0) * radius,
            ]
            for p in points
        ]
    )


def resample(points: np.ndarray, step: float = STEP) -> tuple[np.ndarray, np.ndarray]:
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    distance = np.arange(0.0, along[-1], step)
    return distance, np.column_stack(
        [np.interp(distance, along, points[:, 0]), np.interp(distance, along, points[:, 1])]
    )


def radii(points: np.ndarray, step: float = STEP, window: float = WINDOW) -> np.ndarray:
    """Radius from how far the direction turns over `window` metres around each point.

    Measuring over a stretch, not across one vertex, treats OpenStreetMap's sparse polyline
    and the denser 2025 outlines alike.
    """
    k = max(1, round(window / step / 2))
    heading = np.unwrap(np.arctan2(*np.diff(points, axis=0, append=points[:1])[:, ::-1].T))
    # Pad a lap either side so the start and finish turn like any other point.
    turn_per_lap = heading[-1] - heading[0] + (heading[1] - heading[0])
    padded = np.concatenate([heading - turn_per_lap, heading, heading + turn_per_lap])
    middle = np.arange(len(heading)) + len(heading)
    turned = np.abs(padded[middle + k] - padded[middle - k])
    with np.errstate(divide="ignore"):
        return np.where(turned > 1e-9, 2 * k * step / turned, np.inf)


def straights(radius: np.ndarray, threshold: float = STRAIGHT_RADIUS) -> list[tuple[int, int]]:
    """Runs of a closed loop where the radius stays above the threshold, as index pairs."""
    straight = radius > threshold
    if straight.all():
        return [(0, len(radius) - 1)]
    start = int(np.argmin(straight))  # begin inside a corner so no run wraps unseen
    runs, current = [], None
    for offset in range(len(radius)):
        index = (start + offset) % len(radius)
        if straight[index] and current is None:
            current = index
        elif not straight[index] and current is not None:
            runs.append((current, (index - 1) % len(radius)))
            current = None
    if current is not None:
        runs.append((current, (start - 1) % len(radius)))
    return runs


def run_length(run: tuple[int, int], count: int, step: float = STEP) -> float:
    return ((run[1] - run[0]) % count + 1) * step


def main_straight(geometry: dict, length: float = INJE_LENGTH) -> dict:
    origin = geometry["course"][0]
    course, pit = metres(geometry["course"], origin), metres(geometry["pit_lane"], origin)
    distance, points = resample(course)
    scale = length / (distance[-1] + STEP)
    radius = radii(points)
    runs = straights(radius)
    longest = max(runs, key=lambda run: run_length(run, len(points)))
    start, end = longest
    middle = points[(start + ((end - start) % len(points)) // 2) % len(points)]
    to_pit = float(np.min(np.hypot(*(pit - middle).T)))
    # The corner before the straight: its tightest point in the 300 m leading onto it.
    window = [(start - offset) % len(points) for offset in range(1, round(300 / STEP))]
    apex = min(window, key=lambda index: radius[index])
    bend = [(start - offset) % len(points) for offset in range(1, round(120 / STEP))]
    final_bend = min(bend, key=lambda index: radius[index])
    return {
        "osm_length_m": float(distance[-1] + STEP),
        "scale": scale,
        "straight_m": run_length(longest, len(points)) * scale,
        "straight_to_pit_lane_m": to_pit,
        "corner_radius_m": float(radius[apex]),
        "final_bend_radius_m": float(radius[final_bend]),
        "corner_to_turn_1_m": run_length((apex, end), len(points)) * scale,
    }


# The AMR25 in the bundled 2025 races


def saved_sessions(year: int = 2025):
    os.environ.setdefault("FORMULA_ANALYST_OFFLINE", "1")
    sys.path.insert(0, str(ROOT))
    from src import races

    saved = races.saved_sessions(ROOT / "data")
    return saved[saved["Year"] == year]


def load_amr25(year: int = 2025) -> list[dict]:
    """Every bundled race of the season, with the AMR25's and the pole sitter's qualifying laps."""
    saved = saved_sessions(year)
    from src import races, sessions

    loaded = []
    for row in saved[saved["Session"] == races.RACE].itertuples():
        weekend = saved[(saved["Round"] == row.Round) & (saved["Session"] == "Qualifying")]
        best = pole = None
        if not weekend.empty:
            qualifying = sessions.saved_qualifying(weekend["Path"].iloc[0]).dropna(subset=["Best"])
            mine = qualifying[qualifying["Driver"].isin(AMR25)]
            if not mine.empty:
                best, pole = float(mine["Best"].min()), float(qualifying["Best"].min())
        loaded.append(
            {"event": row.EventName, "race": races.read_race(row.Path), "quali": best, "pole": pole}
        )
    return loaded


def corner_samples(race, driver: str) -> list[tuple[float, float]]:
    """Each corner's tightest radius on the track outline and the car's lowest speed there."""
    trace = race.trace(driver)
    if trace.empty or race.track.empty or race.corners.empty:
        return []
    distance, points = resample(race.track[["X", "Y"]].to_numpy(dtype=float))
    radius = radii(points)
    # The outline and the lap use slightly different lengths, so line them up in proportion.
    along = distance * trace["Distance"].max() / (distance[-1] + STEP)
    samples = []
    for corner in race.corners["Distance"]:
        near = trace[(trace["Distance"] - corner).abs() <= 80]
        bend = radius[np.abs(along - corner) <= 60]
        if near.empty or not len(bend):
            continue
        tightest = float(np.min(bend))
        if 10 < tightest < 600:
            samples.append((tightest, float(near["Speed"].min())))
    return samples


def fit_corners(samples: list[tuple[float, float]]) -> dict:
    """Corner speed from radius, as a power law fitted to the AMR25's own corners."""
    r, v = np.log([s[0] for s in samples]), np.log([s[1] for s in samples])
    slope, intercept = np.polyfit(r, v, 1)
    spread = float(np.std(v - (slope * r + intercept)))
    return {
        "corners": len(samples),
        "slope": float(slope),
        "intercept": float(intercept),
        "spread": spread,
    }


def corner_speed(fit: dict, radius: float) -> dict:
    centre = fit["slope"] * math.log(radius) + fit["intercept"]
    return {
        **fit,
        "speed": math.exp(centre),
        "low": math.exp(centre - fit["spread"]),
        "high": math.exp(centre + fit["spread"]),
    }


def throttle_runs(trace: pd.DataFrame) -> list[dict]:
    """Every stretch flat out from a corner, with the speed against distance from its apex."""
    trace = trace.sort_values("Distance").reset_index(drop=True)
    flat = (trace["Throttle"] >= FULL_THROTTLE) & ~trace["Brake"].astype(bool)
    runs, start = [], None
    for index, on in enumerate([*flat, False]):
        if on and start is None:
            start = index
        elif not on and start is not None:
            lead = trace[(trace["Distance"] >= trace["Distance"][start] - 150)].index
            lead = [i for i in lead if i <= start]
            apex = min(lead, key=lambda i: trace["Speed"][i])
            stretch = trace.iloc[apex:index]
            runs.append(
                {
                    "apex_speed": float(trace["Speed"][apex]),
                    "apex_at": float(trace["Distance"][apex]),
                    "distance": (stretch["Distance"] - trace["Distance"][apex]).to_numpy(),
                    "speed": stretch["Speed"].to_numpy(dtype=float),
                    "drs": bool(stretch["DRS"].astype(bool).any()),
                }
            )
            start = None
    return runs


def fastest_lap(race, driver: str) -> int | None:
    laps = race.laps[race.laps["Driver"] == driver].dropna(subset=["LapTimeSeconds"])
    return None if laps.empty else int(laps.loc[laps["LapTimeSeconds"].idxmin(), "LapNumber"])


def top_speed(straight: dict, loaded: list[dict], corners: dict) -> dict:
    """The speed an AMR25 reached as far from a corner like Inje's as Turn 1 is."""
    reach = straight["corner_to_turn_1_m"]
    matches = []
    for weekend in loaded:
        race = weekend["race"]
        for driver, number in AMR25.items():
            trace = race.trace(driver)
            if trace.empty:
                continue
            for run in throttle_runs(trace):
                long_enough = run["distance"][-1] >= reach
                if long_enough and not run["drs"]:
                    matches.append(
                        {
                            "event": weekend["event"],
                            "driver": driver,
                            "car": number,
                            "lap": fastest_lap(race, driver),
                            "apex_speed": run["apex_speed"],
                            "apex_at": run["apex_at"],
                            "run_m": float(run["distance"][-1]),
                            "at_braking": float(
                                np.interp(reach - BRAKING_ALLOWANCE, run["distance"], run["speed"])
                            ),
                            "at_turn_1": float(np.interp(reach, run["distance"], run["speed"])),
                        }
                    )
    corner = corner_speed(corners, straight["corner_radius_m"])
    within = [m for m in matches if corner["low"] <= m["apex_speed"] <= corner["high"]]
    closeness = lambda m: abs(m["apex_speed"] - corner["speed"])  # noqa: E731
    best = min(within or matches, key=closeness)
    # One race: every run in it out of a corner like Inje's, both cars, read at both ends of
    # the braking allowance.
    same = [m for m in within or [best] if m["event"] == best["event"]]
    speeds = [speed for m in same for speed in (m["at_braking"], m["at_turn_1"])]
    return {
        "corner": corner,
        "match": best,
        "same_straight": same,
        "range_kmh": [min(speeds), max(speeds)],
        "candidates": sorted(within, key=closeness),
    }


# The computer lap


def performance(loaded: list[dict]) -> dict:
    """How hard the AMR25 accelerates and brakes at each speed, from its 2025 laps.

    Acceleration is the 75th percentile of flat-out samples without DRS, braking the 10th, so
    both describe what the car can do rather than what it did on an average sample.
    """
    accelerating, braking = [], []
    for weekend in loaded:
        for driver in AMR25:
            trace = weekend["race"].trace(driver).sort_values("Distance")
            if trace.empty:
                continue
            speed = trace["Speed"].to_numpy(dtype=float) / 3.6
            dt = np.diff(trace["Time"].to_numpy(dtype=float))
            rate = np.where(dt > 0, np.diff(speed) / np.maximum(dt, 1e-3), np.nan)
            middle = (speed[1:] + speed[:-1]) / 2
            brake = trace["Brake"].astype(bool).to_numpy()
            flat = (trace["Throttle"].to_numpy() >= 98) & ~brake
            flat &= ~trace["DRS"].astype(bool).to_numpy()
            # A sample counts when the car was flat out, or braking, at both its ends.
            for store, state in ((accelerating, flat), (braking, brake)):
                both = state[1:] & state[:-1]
                store += list(zip(middle[both], rate[both], strict=True))
    tables = {}
    for name, samples, percentile in (("accel", accelerating, 75), ("brake", braking, 10)):
        samples = np.array(samples)
        speeds, values = [], []
        for low in np.arange(0, 100, BIN / 3.6):
            inside = samples[(samples[:, 0] >= low) & (samples[:, 0] < low + BIN / 3.6)]
            if len(inside) >= MIN_SAMPLES:
                speeds.append(low + BIN / 7.2)
                values.append(abs(float(np.percentile(inside[:, 1], percentile))))
        tables[name] = (np.array(speeds), np.array(values))
    # Flat out, the car stops gaining speed where acceleration runs out.
    speeds, values = tables["accel"]
    tables["accel"] = (np.append(speeds, speeds[-1] + BIN / 3.6), np.append(values, 0.0))
    return tables


def simulate(radius: np.ndarray, step: float, corners: dict, car: dict) -> np.ndarray:
    """Speed round a closed lap, as fast as the corners, acceleration and braking allow.

    The classic two-pass lap simulation: corner speeds cap every point, a forward pass
    accelerates out of each corner and a backward pass brakes into the next.
    """
    accel, brake = car["accel"], car["brake"]
    with np.errstate(divide="ignore"):
        limit = np.exp(corners["intercept"]) * radius ** corners["slope"] / 3.6
    limit = np.minimum(limit, accel[0][-1])
    start = int(np.argmin(limit))
    limit = np.roll(limit, -start)
    forward = limit.copy()
    for i in range(len(limit) - 1):
        gain = 2 * np.interp(forward[i], *accel) * step
        forward[i + 1] = min(limit[i + 1], math.sqrt(forward[i] ** 2 + gain))
    backward = forward.copy()
    for i in range(len(limit) - 2, -1, -1):
        loss = 2 * np.interp(backward[i + 1], *brake) * step
        backward[i] = min(forward[i], math.sqrt(backward[i + 1] ** 2 + loss))
    return np.roll(backward, start)


def lap_seconds(speed: np.ndarray, step: float) -> float:
    return float(np.sum(step / ((speed + np.roll(speed, -1)) / 2)))


def outline_radius(points: np.ndarray) -> tuple[np.ndarray, float]:
    distance, resampled = resample(points)
    return radii(resampled), distance[-1] + STEP


def computer_lap(
    geometry: dict, loaded: list[dict], corners: dict, length: float = INJE_LENGTH
) -> dict:
    """The simulated Inje lap, corrected by how the simulator did on the 2025 circuits."""
    car = performance(loaded)
    checks = []
    for weekend in loaded:
        race = weekend["race"]
        traces = [race.trace(driver) for driver in AMR25]
        laps = race.laps[race.laps["Driver"].isin(AMR25)]["LapTimeSeconds"].dropna()
        if race.track.empty or laps.empty or all(trace.empty for trace in traces):
            continue
        radius, outline = outline_radius(race.track[["X", "Y"]].to_numpy(dtype=float))
        lap_length = max(trace["Distance"].max() for trace in traces if not trace.empty)
        step = STEP * lap_length / outline
        simulated = lap_seconds(simulate(radius, step, corners, car), step)
        checks.append(
            {
                "event": weekend["event"],
                "simulated": simulated,
                "race_lap": float(laps.min()),
                "quali_lap": weekend["quali"],
                "pole": weekend["pole"],
            }
        )
    table = pd.DataFrame(checks)
    origin = geometry["course"][0]
    radius, outline = outline_radius(metres(geometry["course"], origin))
    step = STEP * length / outline
    speed = simulate(radius, step, corners, car)
    simulated = lap_seconds(speed, step)
    result = {"simulated": simulated, "max_kmh": float(speed.max() * 3.6), "checks": checks}
    for name, column in (("qualifying", "quali_lap"), ("race", "race_lap")):
        ratio = (table[column] / table["simulated"]).dropna()
        low, middle, high = np.percentile(ratio, [25, 50, 75])
        result[name] = {
            "ratio": float(middle),
            "lap": simulated * middle,
            "low": simulated * low,
            "high": simulated * high,
            "circuits": int(len(ratio)),
        }
    gap = (table["quali_lap"] / table["pole"] - 1).dropna()
    result["gap_to_pole"] = float(gap.median())
    result["speed_trace"] = (speed * 3.6).round(1).tolist()
    result["step_m"] = step
    return result


# Output


def chart(
    fit: tuple[float, float, float], amr25_kic: float, amr25: dict, gap: float, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    carbon, surface, text, muted, grid = "#15151E", "#1C1C25", "#FFFFFF", "#AAAAAA", "#303037"
    soft, medium, hard = "#DA291C", "#FFD12E", "#F0F0EC"
    a, b, r = fit
    kic = np.array([seconds(pair[0]) for pair in BRIDGE.values()])
    inje = np.array([seconds(pair[1]) for pair in BRIDGE.values()])
    f1_kic = seconds(F1_KIC_2013["qualifying"])

    figure, axes = plt.subplots(figsize=(16, 10), dpi=100)
    figure.patch.set_facecolor(carbon)
    axes.set_facecolor(surface)
    xs = np.linspace(82, 170, 50)
    axes.plot(xs, a * xs + b, color=medium, lw=2, zorder=2, label="Fit through the classes")
    axes.plot(
        xs,
        xs * class_ratio(BRIDGE),
        color=medium,
        lw=1.5,
        ls="--",
        alpha=0.55,
        zorder=2,
        label=f"{RATIO_CLASS} ratio",
    )
    axes.scatter(kic, inje, s=90, color=hard, zorder=3, label="Korean class records")
    for name, x, y in zip(BRIDGE, kic, inje, strict=True):
        axes.annotate(
            name, (x, y), xytext=(10, -14), textcoords="offset points", color=text, fontsize=13
        )
    bridged = a * f1_kic + b
    axes.scatter([f1_kic], [bridged], s=140, facecolors="none", edgecolors=soft, lw=2, zorder=4)
    axes.annotate(
        "F1 2013 pole, bridged",
        (f1_kic, bridged),
        xytext=(-12, 18),
        textcoords="offset points",
        ha="right",
        color=muted,
        fontsize=12,
    )
    # The bridge predicts a pole lap; the AMR25 qualified `gap` off pole in 2025.
    low, high = (time * (1 + gap) for time in sorted([amr25["ratio"], amr25["linear"]]))
    axes.vlines(amr25_kic, low, high, color=soft, lw=4, zorder=4)
    axes.scatter([amr25_kic], [amr25["linear"] * (1 + gap)], s=220, color=soft, zorder=5)
    axes.annotate(
        f"AMR25, predicted\n{lap_time(low, 1)} to {lap_time(high, 1)}",
        (amr25_kic, (low + high) / 2),
        xytext=(70, -60),
        textcoords="offset points",
        va="top",
        color=text,
        fontsize=15,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": soft},
    )
    axes.annotate(
        "",
        xy=(amr25_kic, amr25["linear"] * (1 + gap)),
        xytext=(f1_kic, bridged),
        arrowprops={"arrowstyle": "->", "color": muted, "lw": 1.2},
    )

    clock = FuncFormatter(lambda value, _: lap_time(value, 0))
    axes.xaxis.set_major_formatter(clock)
    axes.yaxis.set_major_formatter(clock)
    axes.set_xlim(78, 172)
    axes.set_ylim(60, 125)
    axes.set_xlabel(
        "Lap record at the Korea International Circuit (full course)", color=muted, fontsize=14
    )
    axes.set_ylabel("Lap record at Inje Speedium", color=muted, fontsize=14)
    axes.tick_params(colors=muted, labelsize=12)
    axes.grid(color=grid, lw=0.8)
    for spine in axes.spines.values():
        spine.set_color(grid)
    axes.set_title(
        "How fast will the AMR25 lap Inje Speedium?",
        color=text,
        fontsize=22,
        fontweight="bold",
        loc="left",
        pad=34,
    )
    axes.text(
        0,
        1.015,
        f"Six Korean classes bridge the two circuits: Inje = {a:.3f} × KIC + "
        f"{b:.2f} s, r = {r:.3f}. Qualifying pace, 2025 car.",
        transform=axes.transAxes,
        color=muted,
        fontsize=13,
    )
    legend = axes.legend(loc="lower right", facecolor=carbon, edgecolor=grid, fontsize=12)
    for label in legend.get_texts():
        label.set_color(text)
    figure.tight_layout()
    figure.savefig(path, facecolor=carbon)
    plt.close(figure)


def speed_chart(computer: dict, straight: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carbon, surface, text, muted, grid, soft = (
        "#15151E",
        "#1C1C25",
        "#FFFFFF",
        "#AAAAAA",
        "#303037",
        "#DA291C",
    )
    speed = np.array(computer["speed_trace"])
    distance = np.arange(len(speed)) * computer["step_m"] / 1000
    figure, axes = plt.subplots(figsize=(16, 7), dpi=100)
    figure.patch.set_facecolor(carbon)
    axes.set_facecolor(surface)
    axes.fill_between(distance, speed, color=soft, alpha=0.15)
    axes.plot(distance, speed, color=soft, lw=2.5)
    top = int(np.argmax(speed))
    axes.annotate(
        f"{speed[top]:.0f} km/h at the end of the {straight['straight_m']:.0f} m straight",
        (distance[top], speed[top]),
        xytext=(-20, 14),
        textcoords="offset points",
        ha="right",
        color=text,
        fontsize=13,
    )
    axes.set_xlim(0, distance[-1])
    axes.set_ylim(0, max(340, speed.max() + 30))
    axes.set_xlabel("Distance round the OpenStreetMap course (km)", color=muted, fontsize=13)
    axes.set_ylabel("Speed (km/h)", color=muted, fontsize=13)
    axes.tick_params(colors=muted, labelsize=12)
    axes.grid(color=grid, lw=0.8)
    for spine in axes.spines.values():
        spine.set_color(grid)
    quali = computer["qualifying"]
    axes.set_title(
        "The computer lap: an AMR25 round Inje Speedium",
        color=text,
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=30,
    )
    axes.text(
        0,
        1.02,
        f"Simulated {lap_time(computer['simulated'], 1)}; corrected by the AMR25's real "
        f"2025 qualifying laps, {lap_time(quali['lap'], 1)} ({lap_time(quali['low'], 1)} to "
        f"{lap_time(quali['high'], 1)}).",
        transform=axes.transAxes,
        color=muted,
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(path, facecolor=carbon)
    plt.close(figure)


def main() -> int:
    print("1. Bridge: Korean class lap records")
    for name, (kic, inje) in BRIDGE.items():
        print(f"   {name:<11} KIC {kic}  Inje {inje}")
    fit = fit_bridge(BRIDGE)
    ratio = class_ratio(BRIDGE)
    print(f"   Linear fit: Inje = {fit[0]:.4f} * KIC + {fit[1]:.3f} s, r = {fit[2]:.4f}")
    print(f"   {RATIO_CLASS} ratio: Inje / KIC = {ratio:.4f}")

    era = era_factor()
    print(
        f"\n2. Era factor 2013 to 2025 from Suzuka poles: {SUZUKA_POLE_2025} / "
        f"{SUZUKA_POLE_2013} = {era:.4f}"
    )

    print("\n3. F1 at KIC in 2013, bridged to Inje, then made a 2025 car")
    predictions, ranges = {}, {}
    for pace, text in F1_KIC_2013.items():
        result = predict(seconds(text), fit, ratio, era)
        predictions[pace] = result
        ranges[pace] = (min(result.values()), max(result.values()))
        print(
            f"   {pace:<10} KIC {text} -> linear {lap_time(result['linear'])}, "
            f"ratio {lap_time(result['ratio'])}"
        )
    for pace, (low, high) in ranges.items():
        print(f"   {pace.capitalize()} pace: {lap_time(low, 1)} to {lap_time(high, 1)}")

    print("\n4. Cross-checks")
    checks = cross_checks(era, ranges)
    for check in checks:
        verdict = describe(check, ranges[check["pace"]])
        print(
            f"   {check['name']} via {check['circuit']}: ratio {check['ratio']:.4f} -> "
            f"{lap_time(check['estimate'])} at {check['pace']} pace, {verdict}"
        )
    disagree = [check["name"] for check in checks if check["off_by"] > DISAGREEMENT]
    if disagree:
        print(f"   WARNING: {', '.join(disagree)} disagree by more than {DISAGREEMENT:g} s")

    print("\n5. Inje's main straight from OpenStreetMap")
    geometry = osm_geometry()
    straight = main_straight(geometry)
    print(f"   OSM full course {straight['osm_length_m']:.0f} m, scaled to {INJE_LENGTH:.0f} m")
    print(
        f"   Main straight {straight['straight_m']:.0f} m, "
        f"{straight['straight_to_pit_lane_m']:.0f} m from the pit lane"
    )
    print(
        f"   Corner before it: radius {straight['corner_radius_m']:.0f} m, then a "
        f"{straight['final_bend_radius_m']:.0f} m bend; {straight['corner_to_turn_1_m']:.0f} m "
        "from that corner to Turn 1"
    )

    print("\n6. Top speed from the AMR25's 2025 data")
    loaded = load_amr25()
    samples = [s for w in loaded for driver in AMR25 for s in corner_samples(w["race"], driver)]
    corners = fit_corners(samples)
    speed = top_speed(straight, loaded, corners)
    corner, match = speed["corner"], speed["match"]
    print(
        f"   Corner speed vs radius over {corner['corners']} AMR25 corners: "
        f"speed ~ radius^{corner['slope']:.2f}"
    )
    print(
        f"   Inje's corner: {corner['speed']:.0f} km/h ({corner['low']:.0f} to "
        f"{corner['high']:.0f})"
    )
    print(
        f"   {len(speed['candidates'])} flat-out runs from a corner that speed, without DRS, as "
        "long as Inje's:"
    )
    for m in speed["candidates"]:
        print(
            f"     {m['event']:<26} car {m['car']}  apex {m['apex_speed']:3.0f} km/h  "
            f"{m['at_braking']:3.0f} to {m['at_turn_1']:3.0f} km/h"
        )
    print(
        f"   Closest: {match['event']} 2025, car {match['car']} ({match['driver']}), lap "
        f"{match['lap']}: {match['apex_speed']:.0f} km/h at the apex, flat out for "
        f"{match['run_m']:.0f} m"
    )
    top = speed["range_kmh"]
    runs = ", ".join(
        f"car {m['car']} lap {m['lap']} from {m['apex_at']:.0f} m" for m in speed["same_straight"]
    )
    print(f"   Runs used from that race: {runs}")
    print(
        f"   Speed {straight['corner_to_turn_1_m'] - BRAKING_ALLOWANCE:.0f} to "
        f"{straight['corner_to_turn_1_m']:.0f} m from the apex: {top[0]:.0f} to {top[1]:.0f} km/h"
    )

    print("\n7. Computer lap: a lap simulation of the AMR25 round Inje")
    computer = computer_lap(geometry, loaded, corners)
    print(
        f"   Simulated from Inje's corners and the AMR25's cornering, acceleration and braking: "
        f"{lap_time(computer['simulated'])}, {computer['max_kmh']:.0f} km/h at most"
    )
    for name in ("qualifying", "race"):
        check = computer[name]
        print(
            f"   {name.capitalize()}: on {check['circuits']} circuits of 2025 the AMR25's real "
            f"laps took {check['ratio']:.3f} times the simulator's -> {lap_time(check['lap'], 1)} "
            f"({lap_time(check['low'], 1)} to {lap_time(check['high'], 1)})"
        )

    quali = ranges["qualifying"]
    gap = computer["gap_to_pole"]
    amr25_quali = (quali[0] * (1 + gap), quali[1] * (1 + gap))
    middle = sum(quali) / 2
    demo = (middle + DEMO_SLOWER[0], middle + DEMO_SLOWER[1])
    print(
        f"\n8. Qualifying lap for a pole-level car: {lap_time(quali[0], 1)} to "
        f"{lap_time(quali[1], 1)}. The AMR25 qualified a median {gap:.1%} off pole in 2025: "
        f"{lap_time(amr25_quali[0], 1)} to {lap_time(amr25_quali[1], 1)}"
    )
    print(
        f"   Realistic demo (judgement: {DEMO_SLOWER[0]:g} to {DEMO_SLOWER[1]:g} s slower than "
        f"{lap_time(middle, 1)}): {lap_time(demo[0], 1)} to {lap_time(demo[1], 1)}"
    )

    results = {
        "fit": {"a": fit[0], "b": fit[1], "r": fit[2], "ratio": ratio},
        "era_factor": era,
        "predictions": predictions,
        "ranges": {pace: list(pair) for pace, pair in ranges.items()},
        "cross_checks": checks,
        "straight": straight,
        "top_speed": {
            "range_kmh": top,
            "corner_kmh": corner,
            "match": match,
            "same_straight": speed["same_straight"],
            "candidates": speed["candidates"],
        },
        "computer_lap": computer,
        "qualifying_lap": {"pole_level": list(quali), "amr25": list(amr25_quali), "gap": gap},
        "demo": {"full_pace_middle": middle, "slower_by": list(DEMO_SLOWER), "range": list(demo)},
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    amr25_kic = seconds(F1_KIC_2013["qualifying"]) * era
    chart(fit, amr25_kic, predictions["qualifying"], gap, HERE / "chart.png")
    speed_chart(computer, straight, HERE / "computer-lap.png")
    print("\nWrote results.json, chart.png and computer-lap.png")
    return 1 if disagree else 0


if __name__ == "__main__":
    raise SystemExit(main())
