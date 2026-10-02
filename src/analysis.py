from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

GREEN_FLAG = "1"
DEFAULT_FUEL_EFFECT = 0.055
FALLBACK_COLOR = "#888888"
UNKNOWN_COMPOUND = "UNKNOWN"
TEAMMATE_MARKERS = {"solid": "o", "dashed": "s", "dashdot": "^", "dotted": "D"}

Styles = dict[str, dict[str, str]]


def representative_laps(laps: pd.DataFrame) -> pd.DataFrame:
    return laps[
        laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & (laps["TrackStatus"] == GREEN_FLAG)
        & laps["IsAccurate"]
    ]


def fuel_corrected(
    laps: pd.DataFrame, total_laps: int, fuel_effect: float = DEFAULT_FUEL_EFFECT
) -> pd.Series:
    laps_remaining = total_laps - laps["LapNumber"]
    return laps["LapTimeSeconds"] - fuel_effect * laps_remaining


def stint_compound(compounds: pd.Series) -> str:
    known = compounds.dropna()
    return str(known.mode().iat[0]) if not known.empty else UNKNOWN_COMPOUND


def stint_summary(laps: pd.DataFrame) -> pd.DataFrame:
    return (
        laps.dropna(subset=["Stint"])
        .groupby(["Driver", "Stint"], as_index=False)
        .agg(
            Compound=("Compound", stint_compound),
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
                "Compound": stint_compound(stint_laps["Compound"]),
                "Laps": len(stint_laps),
                "DegPerLap": fit.slope,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("DegPerLap", ignore_index=True)


def degradation_by_compound(deg: pd.DataFrame) -> pd.DataFrame:
    return deg.groupby("Compound")["DegPerLap"].agg(["count", "mean", "median"]).round(3)


def fastest_lap(laps: pd.DataFrame, driver: str) -> pd.Series | None:
    candidates = laps[(laps["Driver"] == driver) & laps["IsPersonalBest"]]
    if candidates["LapTime"].isna().all():
        return None
    return candidates.loc[candidates["LapTime"].idxmin()]


def format_lap_time(value: pd.Timedelta) -> str:
    if pd.isna(value):
        return ""
    minutes, seconds = divmod(value.total_seconds(), 60)
    return f"{int(minutes)}:{seconds:06.3f}"


def _new_axes(width: float, height: float) -> tuple[Figure, Axes]:
    fig = Figure(figsize=(width, height))
    ax = fig.subplots()
    ax.grid(True, alpha=0.3)
    return fig, ax


def _line_style(styles: Styles, driver: str) -> dict[str, str]:
    style = styles.get(driver, {})
    return {key: style[key] for key in ("color", "linestyle") if key in style}


def _marker_style(styles: Styles, driver: str) -> dict[str, str | float]:
    linestyle = styles.get(driver, {}).get("linestyle", "solid")
    style: dict[str, str | float] = {
        "marker": TEAMMATE_MARKERS.get(linestyle, "o"),
        "markersize": 4,
    }
    if linestyle != "solid":
        style["markerfacecolor"] = "white"
    return style


def pace_figure(
    laps: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
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
            label=driver,
            **_line_style(styles, driver),
            **_marker_style(styles, driver),
        )
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    if ax.has_data():
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def strategy_figure(
    stints: pd.DataFrame, order: list[str], compound_colors: dict[str, str]
) -> Figure:
    fig, ax = _new_axes(12, max(4, 0.4 * len(order)))
    ax.grid(axis="y", visible=False)
    used: dict[str, str] = {}
    for driver in order:
        for _, stint in stints[stints["Driver"] == driver].sort_values("Stint").iterrows():
            compound = stint["Compound"]
            used[compound] = compound_colors.get(compound, FALLBACK_COLOR)
            ax.barh(
                driver,
                stint["Laps"],
                left=stint["StartLap"] - 1,
                color=used[compound],
                edgecolor="black",
            )
    if used:
        handles = [Patch(facecolor=c, edgecolor="black", label=name) for name, c in used.items()]
        ax.legend(handles=handles, title="Compound", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.invert_yaxis()
    ax.set_xlabel("Lap")
    fig.tight_layout()
    return fig


def speed_trace_figure(
    traces: dict[str, pd.DataFrame], styles: Styles, corners: pd.DataFrame
) -> Figure:
    fig, ax = _new_axes(14, 5)
    for driver, trace in traces.items():
        ax.plot(trace["Distance"], trace["Speed"], label=driver, **_line_style(styles, driver))

    if ax.has_data():
        lowest = min(trace["Speed"].min() for trace in traces.values())
        for _, corner in corners.iterrows():
            label = f"{corner['Number']}{corner['Letter']}"
            ax.axvline(corner["Distance"], color="grey", linestyle=":", alpha=0.5)
            ax.text(corner["Distance"], lowest - 20, label, ha="center", fontsize=8)
        ax.set_ylim(bottom=lowest - 30)
        ax.legend()

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    fig.tight_layout()
    return fig


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
