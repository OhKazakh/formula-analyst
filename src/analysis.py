from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

GREEN_FLAG = "1"
DEFAULT_FUEL_EFFECT = 0.055
FALLBACK_COLOR = "#888888"
UNKNOWN_COMPOUND = "UNKNOWN"
TEAMMATE_MARKERS = {"solid": "o", "dashed": "s", "dashdot": "^", "dotted": "D"}
TEAMMATE_SYMBOLS = {"solid": "circle", "dashed": "square", "dashdot": "diamond", "dotted": "x"}
REPLAY_FRAMES_PER_LAP = 20
REPLAY_MAX_FRAMES = 1500
REPLAY_FRAME_MS = 60
LINEAR_PROFILE = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

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


def lap_profile(trace: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    distance = trace["Distance"].to_numpy(dtype=float)
    if len(distance) < 2 or distance[-1] <= distance[0]:
        return LINEAR_PROFILE
    speed = np.maximum(trace["Speed"].to_numpy(dtype=float) / 3.6, 1.0)
    elapsed = np.cumsum(np.diff(distance, prepend=distance[0]) / speed)
    return elapsed / elapsed[-1], (distance - distance[0]) / (distance[-1] - distance[0])


def replay_times(laps: pd.DataFrame, total_laps: int) -> np.ndarray:
    timed = laps.dropna(subset=["LapStartTime", "Time"])
    start = timed.loc[timed["LapNumber"] == timed["LapNumber"].min(), "LapStartTime"].min()
    end = timed["Time"].max()
    frames = min(REPLAY_MAX_FRAMES, REPLAY_FRAMES_PER_LAP * total_laps)
    return np.linspace(start.total_seconds(), end.total_seconds(), frames)


def race_positions(
    laps: pd.DataFrame, times: np.ndarray, profile: tuple[np.ndarray, np.ndarray] = LINEAR_PROFILE
) -> pd.DataFrame:
    frames = []
    for driver, driver_laps in laps.dropna(subset=["LapStartTime", "Time"]).groupby("Driver"):
        driver_laps = driver_laps.sort_values("LapNumber")
        starts = driver_laps["LapStartTime"].dt.total_seconds().to_numpy()
        ends = driver_laps["Time"].dt.total_seconds().to_numpy()
        numbers = driver_laps["LapNumber"].to_numpy(dtype=float)
        index = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)
        duration = np.maximum(ends[index] - starts[index], 1e-9)
        fraction = np.clip((times - starts[index]) / duration, 0.0, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "Frame": np.arange(len(times)),
                    "Driver": driver,
                    "Lap": numbers[index],
                    "Progress": numbers[index] - 1 + np.interp(fraction, *profile),
                    "Running": (times >= starts[0]) & (times <= ends[-1]),
                    "Finish": ends[-1],
                }
            )
        )
    columns = ["Frame", "Driver", "Lap", "Progress", "Running", "Finish"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def track_points(track: pd.DataFrame, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = track[["X", "Y"]].to_numpy(dtype=float)
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    target = (np.asarray(fractions) % 1.0) * distance[-1]
    return np.interp(target, distance, xy[:, 0]), np.interp(target, distance, xy[:, 1])


def _standings(frame: pd.DataFrame, lap: int, total_laps: int) -> str:
    stopped = frame["Finish"].where(~frame["Running"].eq(True), np.inf)
    ordered = frame.assign(Stopped=stopped).sort_values(
        ["Progress", "Stopped"], ascending=[False, True], kind="stable"
    )["Driver"]
    lines = [f"{place:>2}  {driver}" for place, driver in enumerate(ordered, start=1)]
    return f"<b>Lap {lap}/{total_laps}</b><br>" + "<br>".join(lines)


def replay_figure(
    track: pd.DataFrame,
    map_corners: pd.DataFrame,
    positions: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
    total_laps: int,
) -> go.Figure:
    width = track["X"].max() - track["X"].min()
    standings_x = track["X"].max() + 0.12 * width
    standings_y = track["Y"].max()
    colors = [styles.get(driver, {}).get("color", FALLBACK_COLOR) for driver in drivers]
    symbols = [
        TEAMMATE_SYMBOLS.get(styles.get(driver, {}).get("linestyle", "solid"), "circle")
        for driver in drivers
    ]

    frames, steps, previous_lap = [], [], 0
    for number, frame in positions.groupby("Frame"):
        frame = frame.set_index("Driver").reindex(drivers)
        x, y = track_points(track, frame["Progress"].to_numpy())
        hidden = ~frame["Running"].eq(True).to_numpy()
        x[hidden] = np.nan
        y[hidden] = np.nan
        lap = int(min(frame["Lap"].max(), total_laps))
        standings = _standings(frame.dropna(subset=["Progress"]).reset_index(), lap, total_laps)
        frames.append(
            go.Frame(
                name=str(number),
                traces=[2, 3],
                data=[
                    go.Scatter(x=np.round(x), y=np.round(y)),
                    go.Scatter(text=[standings]),
                ],
            )
        )
        if lap != previous_lap:
            previous_lap = lap
            steps.append(
                {
                    "method": "animate",
                    "label": str(lap),
                    "args": [
                        [str(number)],
                        {"mode": "immediate", "frame": {"duration": 0, "redraw": False}},
                    ],
                }
            )

    start = frames[0].data
    figure = go.Figure(
        data=[
            go.Scatter(
                x=track["X"],
                y=track["Y"],
                mode="lines",
                line={"color": "#c9ced6", "width": 10},
                hoverinfo="skip",
            ),
            go.Scatter(
                x=map_corners["X"],
                y=map_corners["Y"],
                text=map_corners["Label"],
                mode="text",
                textfont={"size": 10, "color": "#8a8f98"},
                hoverinfo="skip",
            ),
            go.Scatter(
                x=start[0].x,
                y=start[0].y,
                text=drivers,
                mode="markers+text",
                textposition="top center",
                textfont={"size": 10},
                marker={
                    "size": 13,
                    "color": colors,
                    "symbol": symbols,
                    "line": {"width": 1, "color": "white"},
                },
                hovertext=drivers,
                hoverinfo="text",
            ),
            go.Scatter(
                x=[standings_x],
                y=[standings_y],
                text=start[1].text,
                mode="text",
                textposition="bottom right",
                textfont={"family": "monospace", "size": 11},
                hoverinfo="skip",
            ),
        ],
        frames=frames,
    )
    figure.update_layout(
        height=620,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "visible": False,
            "range": [track["X"].min() - 0.05 * width, standings_x + 0.3 * width],
        },
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": 0,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 10},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": REPLAY_FRAME_MS, "redraw": False},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "len": 0.88,
                "y": 0,
                "yanchor": "top",
                "pad": {"t": 10},
                "currentvalue": {"visible": False},
                "ticklen": 0,
                "minorticklen": 0,
                "steps": steps,
            }
        ],
    )
    return figure
