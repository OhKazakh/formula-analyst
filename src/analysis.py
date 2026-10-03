from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, to_rgb
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
QUICK_LAP_THRESHOLD = 1.07
MINI_SECTORS = 25
TRACK_BACKGROUND = "#d0d4da"
MUTED_COLOR = "#d6d9de"
INK = "#16181D"
SECONDARY_INK = "#5B606B"
TIME_LABELS = {"FuelCorrected": "Fuel-corrected lap time (s)"}
CHART_STYLE = {
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#C9CDD4",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "grid.color": "#E6E8EC",
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "legend.frameon": False,
    "text.color": INK,
}

matplotlib.rcParams.update(CHART_STYLE)

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


def format_seconds(value: float) -> str:
    if pd.isna(value):
        return ""
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)}:{seconds:06.3f}"


def format_lap_time(value: pd.Timedelta) -> str:
    return "" if pd.isna(value) else format_seconds(value.total_seconds())


def _new_axes(width: float, height: float) -> tuple[Figure, Axes]:
    fig = Figure(figsize=(width, height))
    ax = fig.subplots()
    ax.grid(True)
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
    ax.set_ylabel(TIME_LABELS.get(time_column, "Lap time (s)"))
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
    fig, ax = _new_axes(12, 5)
    for driver, trace in traces.items():
        ax.plot(trace["Distance"], trace["Speed"], label=driver, **_line_style(styles, driver))

    if ax.has_data():
        lowest = min(trace["Speed"].min() for trace in traces.values())
        for _, corner in corners.iterrows():
            label = f"{corner['Number']}{corner['Letter']}"
            ax.axvline(corner["Distance"], color="grey", linestyle=":", alpha=0.5)
            ax.text(corner["Distance"], lowest - 20, label, ha="center", fontsize=10)
        ax.set_ylim(bottom=lowest - 30)
        ax.legend()

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    fig.tight_layout()
    return fig


def stint_fit_figure(
    stint_laps: pd.DataFrame, time_column: str = "LapTimeSeconds", color: str = FALLBACK_COLOR
) -> Figure:
    fig, ax = _new_axes(10, 4.5)
    x, y = stint_laps["TyreLife"], stint_laps[time_column]
    ax.scatter(x, y, s=48, color=color, edgecolor=INK, linewidth=0.6, label="Laps", zorder=3)
    if len(stint_laps) >= 2:
        fit = linear_fit(x, y)
        ax.plot(x, fit.slope * x + fit.intercept, color=INK, label=f"Fit: {fit.slope:+.3f} s/lap")
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel(TIME_LABELS.get(time_column, "Lap time (s)"))
    ax.legend()
    fig.tight_layout()
    return fig


def _elapsed(trace: pd.DataFrame) -> np.ndarray:
    distance = trace["Distance"].to_numpy(dtype=float)
    speed = np.maximum(trace["Speed"].to_numpy(dtype=float) / 3.6, 1.0)
    return np.cumsum(np.diff(distance, prepend=distance[0]) / speed)


def lap_profile(trace: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    distance = trace["Distance"].to_numpy(dtype=float)
    if len(distance) < 2 or distance[-1] <= distance[0]:
        return LINEAR_PROFILE
    elapsed = _elapsed(trace)
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


def lap_positions(laps: pd.DataFrame) -> pd.DataFrame:
    timed = laps.dropna(subset=["Time"])[["Driver", "LapNumber", "Time"]]
    ranks = timed.groupby("LapNumber")["Time"].rank(method="first").astype(int)
    return (
        timed.assign(Position=ranks)[["Driver", "LapNumber", "Position"]]
        .sort_values(["Driver", "LapNumber"])
        .reset_index(drop=True)
    )


def position_figure(
    positions: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
    highlight: list[str] | None = None,
) -> Figure:
    fig, ax = _new_axes(12, 7)
    highlighted = set(highlight or [])
    for driver in drivers:
        driver_positions = positions[positions["Driver"] == driver]
        if driver_positions.empty:
            continue
        style = _line_style(styles, driver)
        emphasis = not highlighted or driver in highlighted
        if not emphasis:
            style["color"] = MUTED_COLOR
        ax.plot(
            driver_positions["LapNumber"],
            driver_positions["Position"],
            label=driver,
            linewidth=2.2 if highlighted and emphasis else 1.5,
            zorder=3 if emphasis else 2,
            **style,
        )
    places = int(positions["Position"].max()) if not positions.empty else 1
    ax.set_ylim(places + 0.5, 0.5)
    ax.set_yticks(range(1, places + 1))
    ax.set_xlabel("Lap")
    ax.set_ylabel("Position")
    if ax.has_data():
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def quick_laps(laps: pd.DataFrame, threshold: float = QUICK_LAP_THRESHOLD) -> pd.DataFrame:
    representative = representative_laps(laps)
    fastest = representative["LapTimeSeconds"].min()
    return representative[representative["LapTimeSeconds"] <= threshold * fastest]


def team_colors(laps: pd.DataFrame, styles: Styles) -> dict[str, str]:
    first_driver = laps.drop_duplicates("Team").set_index("Team")["Driver"]
    return {
        team: styles.get(driver, {}).get("color", FALLBACK_COLOR)
        for team, driver in first_driver.items()
    }


def team_pace(laps: pd.DataFrame) -> pd.DataFrame:
    median = quick_laps(laps).groupby("Team")["LapTimeSeconds"].median().sort_values()
    return pd.DataFrame(
        {"Team": median.index, "Median": median.to_numpy(), "Gap": median.to_numpy() - median.min()}
    )


def team_pace_figure(laps: pd.DataFrame, styles: Styles) -> Figure:
    quick = quick_laps(laps)
    order = team_pace(laps)["Team"].tolist()
    colors = team_colors(quick, styles)
    fig, ax = _new_axes(8, 4.8)
    ax.grid(axis="x", visible=False)
    boxes = ax.boxplot(
        [quick.loc[quick["Team"] == team, "LapTimeSeconds"] for team in order],
        tick_labels=order,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops={"color": "black"},
    )
    for box, team in zip(boxes["boxes"], order, strict=True):
        box.set_facecolor(colors[team])
        box.set_edgecolor("black")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    ax.set_ylabel("Lap time (s)")
    fig.tight_layout()
    return fig


def lap_distribution_figure(
    laps: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
    compound_colors: dict[str, str],
) -> Figure:
    quick = quick_laps(laps)
    groups = [(driver, quick[quick["Driver"] == driver]) for driver in drivers]
    groups = [(driver, rows) for driver, rows in groups if rows["LapTimeSeconds"].nunique() > 1]
    fig, ax = _new_axes(12, 5)
    ax.grid(axis="x", visible=False)
    if not groups:
        return fig

    positions = np.arange(len(groups))
    violins = ax.violinplot(
        [driver_laps["LapTimeSeconds"] for _, driver_laps in groups],
        positions=positions,
        widths=0.8,
        showextrema=False,
    )
    rng = np.random.default_rng(0)
    used: dict[str, str] = {}
    for position, body, (driver, driver_laps) in zip(
        positions, violins["bodies"], groups, strict=True
    ):
        body.set_facecolor(styles.get(driver, {}).get("color", FALLBACK_COLOR))
        body.set_edgecolor("black")
        body.set_alpha(0.3)
        compounds = driver_laps["Compound"].fillna(UNKNOWN_COMPOUND)
        colors = [compound_colors.get(compound, FALLBACK_COLOR) for compound in compounds]
        used.update(zip(compounds, colors, strict=True))
        ax.scatter(
            position + rng.uniform(-0.15, 0.15, len(driver_laps)),
            driver_laps["LapTimeSeconds"],
            c=colors,
            s=14,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )
    ax.set_xticks(positions, [driver for driver, _ in groups])
    ax.set_ylabel("Lap time (s)")
    handles = [Patch(facecolor=c, edgecolor="black", label=name) for name, c in used.items()]
    ax.legend(handles=handles, title="Compound", loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def _track_segments(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xy = track[["X", "Y"]].to_numpy(dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    midpoints = (cumulative[:-1] + cumulative[1:]) / 2 / cumulative[-1]
    return np.stack([xy[:-1], xy[1:]], axis=1), midpoints


def _map_axes(track: pd.DataFrame, map_corners: pd.DataFrame) -> tuple[Figure, Axes]:
    fig = Figure(figsize=(8, 5.5))
    ax = fig.subplots()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(track["X"], track["Y"], color=TRACK_BACKGROUND, linewidth=11, zorder=1)
    for corner in map_corners.itertuples():
        ax.text(corner.X, corner.Y, corner.Label, fontsize=10, color=SECONDARY_INK, ha="center")
    return fig, ax


def speed_map_figure(track: pd.DataFrame, map_corners: pd.DataFrame, trace: pd.DataFrame) -> Figure:
    segments, midpoints = _track_segments(track)
    distance = trace["Distance"].to_numpy(dtype=float)
    speed = np.interp(midpoints * distance[-1], distance, trace["Speed"].to_numpy(dtype=float))
    fig, ax = _map_axes(track, map_corners)
    collection = LineCollection(segments, cmap="plasma", linewidths=5, capstyle="round", zorder=2)
    collection.set_array(speed)
    ax.add_collection(collection)
    fig.colorbar(collection, ax=ax, label="Speed (km/h)", shrink=0.7)
    fig.tight_layout()
    return fig


def mini_sector_times(trace: pd.DataFrame, sectors: int = MINI_SECTORS) -> np.ndarray:
    distance = trace["Distance"].to_numpy(dtype=float)
    boundaries = np.linspace(0.0, 1.0, sectors + 1) * distance[-1]
    return np.diff(np.interp(boundaries, distance, _elapsed(trace)))


def faster_by_sector(traces: dict[str, pd.DataFrame], sectors: int = MINI_SECTORS) -> np.ndarray:
    drivers = list(traces)
    times = np.vstack([mini_sector_times(traces[driver], sectors) for driver in drivers])
    return np.array(drivers)[times.argmin(axis=0)]


def _shade(color: str, factor: float) -> str:
    return to_hex(tuple(channel * factor for channel in to_rgb(color)))


def distinct_colors(drivers: list[str], styles: Styles) -> dict[str, str]:
    colors: dict[str, str] = {}
    for driver in drivers:
        color = styles.get(driver, {}).get("color", FALLBACK_COLOR)
        colors[driver] = _shade(color, 0.55) if color in colors.values() else color
    return colors


def dominance_map_figure(
    track: pd.DataFrame,
    map_corners: pd.DataFrame,
    traces: dict[str, pd.DataFrame],
    styles: Styles,
    sectors: int = MINI_SECTORS,
) -> Figure:
    segments, midpoints = _track_segments(track)
    winners = faster_by_sector(traces, sectors)
    colors = distinct_colors(list(traces), styles)
    sector = np.minimum((midpoints * sectors).astype(int), sectors - 1)
    fig, ax = _map_axes(track, map_corners)
    ax.add_collection(
        LineCollection(
            segments,
            colors=[colors[winners[s]] for s in sector],
            linewidths=5,
            capstyle="round",
            zorder=2,
        )
    )
    handles = [
        Patch(color=colors[driver], label=f"{driver} ({(winners == driver).sum()} of {sectors})")
        for driver in traces
    ]
    ax.legend(handles=handles, title="Faster in", loc="upper left", bbox_to_anchor=(1.0, 1))
    fig.tight_layout()
    return fig
