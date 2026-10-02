from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
from fastf1.core import Lap, Laps, Session
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
GREEN_FLAG = "1"
DEFAULT_FUEL_EFFECT = 0.055
FALLBACK_COMPOUND_COLOR = "#888888"


def enable_cache(cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def race_calendar(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    finished = schedule[schedule["Session5DateUtc"] < now]
    return finished[["RoundNumber", "EventName", "Country", "EventDate"]].reset_index(drop=True)


def load_race(year: int, event: str | int) -> Session:
    session = fastf1.get_session(year, event, "R")
    session.load(weather=False, messages=False)
    return session


def finishing_order(session: Session) -> list[str]:
    return session.results["Abbreviation"].dropna().tolist()


def with_seconds(laps: Laps) -> Laps:
    laps = laps.copy()
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    return laps


def representative_laps(laps: Laps) -> Laps:
    return laps.pick_wo_box().pick_track_status(GREEN_FLAG).pick_accurate()


def fuel_corrected(
    laps: pd.DataFrame, total_laps: int, fuel_effect: float = DEFAULT_FUEL_EFFECT
) -> pd.Series:
    laps_remaining = total_laps - laps["LapNumber"]
    return laps["LapTimeSeconds"] - fuel_effect * laps_remaining


def stint_summary(laps: pd.DataFrame) -> pd.DataFrame:
    return (
        laps.dropna(subset=["Stint"])
        .groupby(["Driver", "Stint"], as_index=False)
        .agg(
            Compound=("Compound", "first"),
            StartLap=("LapNumber", "min"),
            EndLap=("LapNumber", "max"),
            Laps=("LapNumber", "count"),
        )
        .astype({"Stint": int, "StartLap": int, "EndLap": int})
    )


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float


def linear_fit(x: pd.Series, y: pd.Series) -> LinearFit:
    slope, intercept = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), 1)
    return LinearFit(float(slope), float(intercept))


def degradation(
    laps: pd.DataFrame, time_column: str = "LapTimeSeconds", min_laps: int = 8
) -> pd.DataFrame:
    columns = ["Driver", "Stint", "Compound", "Laps", "DegPerLap"]
    timed = laps.dropna(subset=["Stint", "TyreLife", time_column])
    rows = []
    for (driver, stint), stint_laps in timed.groupby(["Driver", "Stint"]):
        if len(stint_laps) < min_laps:
            continue
        fit = linear_fit(stint_laps["TyreLife"], stint_laps[time_column])
        rows.append(
            {
                "Driver": driver,
                "Stint": int(stint),
                "Compound": stint_laps["Compound"].iloc[0],
                "Laps": len(stint_laps),
                "DegPerLap": fit.slope,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("DegPerLap", ignore_index=True)


def degradation_by_compound(deg: pd.DataFrame) -> pd.DataFrame:
    return deg.groupby("Compound")["DegPerLap"].agg(["count", "mean", "median"]).round(3)


def fastest_lap(laps: Laps, driver: str) -> Lap | None:
    lap = laps.pick_drivers(driver).pick_fastest()
    if lap is None or lap.empty or pd.isna(lap["LapTime"]):
        return None
    return lap


def format_lap_time(value: pd.Timedelta) -> str:
    if pd.isna(value):
        return ""
    minutes, seconds = divmod(value.total_seconds(), 60)
    return f"{int(minutes)}:{seconds:06.3f}"


def _driver_style(driver: str, session: Session) -> dict:
    try:
        return fastf1.plotting.get_driver_style(driver, ["color", "linestyle"], session)
    except (KeyError, ValueError):
        return {}


def _compound_color(compound: str, session: Session) -> str:
    try:
        return fastf1.plotting.get_compound_color(compound, session)
    except (KeyError, ValueError):
        return FALLBACK_COMPOUND_COLOR


def _new_axes(width: float, height: float) -> tuple[Figure, Axes]:
    fig = Figure(figsize=(width, height))
    ax = fig.subplots()
    ax.grid(True, alpha=0.3)
    return fig, ax


def pace_figure(
    laps: pd.DataFrame,
    drivers: list[str],
    session: Session,
    time_column: str = "LapTimeSeconds",
) -> Figure:
    fig, ax = _new_axes(12, 5)
    for driver in drivers:
        driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
        if driver_laps.empty:
            continue
        ax.plot(
            driver_laps["LapNumber"],
            driver_laps[time_column],
            marker="o",
            markersize=3,
            label=driver,
            **_driver_style(driver, session),
        )
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    if ax.has_data():
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def strategy_figure(stints: pd.DataFrame, order: list[str], session: Session) -> Figure:
    fig, ax = _new_axes(12, max(4, 0.4 * len(order)))
    ax.grid(axis="y", visible=False)
    colors: dict[str, str] = {}
    for driver in order:
        for _, stint in stints[stints["Driver"] == driver].sort_values("Stint").iterrows():
            compound = stint["Compound"] if isinstance(stint["Compound"], str) else "UNKNOWN"
            if compound not in colors:
                colors[compound] = _compound_color(compound, session)
            ax.barh(
                driver,
                stint["Laps"],
                left=stint["StartLap"] - 1,
                color=colors[compound],
                edgecolor="black",
            )
    if colors:
        handles = [Patch(facecolor=c, edgecolor="black", label=name) for name, c in colors.items()]
        ax.legend(handles=handles, title="Compound", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.invert_yaxis()
    ax.set_xlabel("Lap")
    fig.tight_layout()
    return fig


def speed_trace_figure(laps: dict[str, Lap], session: Session) -> Figure:
    fig, ax = _new_axes(14, 5)
    lowest = np.inf
    for driver, lap in laps.items():
        telemetry = lap.get_car_data().add_distance()
        lowest = min(lowest, telemetry["Speed"].min())
        ax.plot(
            telemetry["Distance"],
            telemetry["Speed"],
            label=driver,
            **_driver_style(driver, session),
        )

    if np.isfinite(lowest):
        for _, corner in _corners(session).iterrows():
            label = f"{corner['Number']}{corner['Letter']}"
            ax.axvline(corner["Distance"], color="grey", linestyle=":", alpha=0.5)
            ax.text(corner["Distance"], lowest - 20, label, ha="center", fontsize=8)
        ax.set_ylim(bottom=lowest - 30)
        ax.legend()

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    fig.tight_layout()
    return fig


def _corners(session: Session) -> pd.DataFrame:
    try:
        return session.get_circuit_info().corners
    except Exception:
        return pd.DataFrame(columns=["Number", "Letter", "Distance"])


def stint_fit_figure(stint_laps: pd.DataFrame, time_column: str = "LapTimeSeconds") -> Figure:
    fig, ax = _new_axes(10, 4.5)
    x, y = stint_laps["TyreLife"], stint_laps[time_column]
    ax.scatter(x, y, label="Laps")
    if len(stint_laps) >= 2:
        fit = linear_fit(x, y)
        ax.plot(x, fit.slope * x + fit.intercept, color="red", label=f"Fit: {fit.slope:+.3f} s/lap")
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Lap time (s)")
    ax.legend()
    fig.tight_layout()
    return fig
