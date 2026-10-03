from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

GREEN_FLAG = "1"
SAFETY_CAR = "4"
RED_FLAG = "5"
VIRTUAL_SAFETY_CAR = ("6", "7")
DEFAULT_FUEL_EFFECT = 0.055
FALLBACK_COLOR = "#888888"
UNKNOWN_COMPOUND = "UNKNOWN"
NEUTRAL = "rgba(128, 128, 128, 0.35)"
OUTLINE = "rgba(128, 128, 128, 0.8)"
BUTTON = "#F7F4F1"
BUTTON_TEXT = "#15151E"
ACCENT = "#E10600"
LIGHT_BACKGROUND = "#FFFFFF"
DARK_BACKGROUND = "#15151E"
LOW_CONTRAST = 1.6
LABEL_OPACITY = 0.75
PLOTLY_DASHES = {"solid": "solid", "dashed": "dash", "dashdot": "dashdot", "dotted": "dot"}
TEAMMATE_SYMBOLS = {"solid": "circle", "dashed": "square", "dashdot": "diamond", "dotted": "x"}
REPLAY_FRAMES_PER_LAP = 20
REPLAY_MAX_FRAMES = 1500
REPLAY_FRAME_MS = 60
REPLAY_MAP_WIDTH = 720
REPLAY_CONTROLS_HEIGHT = 110
REPLAY_HEIGHT_RANGE = (440, 640)
LINEAR_PROFILE = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
QUICK_LAP_THRESHOLD = 1.07
MINI_SECTORS = 25
MAP_POINTS = 1200
TIME_LABELS = {"FuelCorrected": "Fuel-corrected lap time (s)"}

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


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = _rgb(color)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _luminance(color: str) -> float:
    channels = np.array(_rgb(color)) / 255
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def contrast(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def hard_to_see(color: str) -> bool:
    if not color.startswith("#"):
        return False
    backgrounds = (LIGHT_BACKGROUND, DARK_BACKGROUND)
    return min(contrast(color, background) for background in backgrounds) < LOW_CONTRAST


def _style(styles: Styles, driver: str) -> tuple[str, str]:
    style = styles.get(driver, {})
    return style.get("color", FALLBACK_COLOR), style.get("linestyle", "solid")


def _line(styles: Styles, driver: str, width: float = 2) -> dict:
    color, linestyle = _style(styles, driver)
    return {"color": color, "dash": PLOTLY_DASHES.get(linestyle, "solid"), "width": width}


def _marker(styles: Styles, driver: str, size: int = 7) -> dict:
    color, linestyle = _style(styles, driver)
    symbol = TEAMMATE_SYMBOLS.get(linestyle, "circle")
    if linestyle == "solid":
        return {
            "symbol": symbol,
            "color": color,
            "size": size,
            "line": {"color": OUTLINE, "width": 1},
        }
    return {"symbol": f"{symbol}-open", "color": color, "size": size, "line": {"width": 2}}


def _halo(x: object, y: object, line: dict) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={**line, "color": OUTLINE, "width": line["width"] + 2.5},
        hoverinfo="skip",
        showlegend=False,
    )


def _add_line(figure: go.Figure, line: dict, **trace: object) -> None:
    if hard_to_see(line["color"]):
        figure.add_trace(_halo(trace["x"], trace["y"], line))
    figure.add_trace(go.Scatter(line=line, **trace))


def _finish(figure: go.Figure, height: int, **layout: object) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom"},
        hovermode="closest",
    )
    figure.update_layout(**layout)
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    return figure


def _tyre_labels(laps: pd.DataFrame) -> list[str]:
    labels = []
    for compound, age in zip(laps["Compound"], laps["TyreLife"], strict=True):
        name = compound if isinstance(compound, str) else UNKNOWN_COMPOUND
        labels.append(f"{name}, {int(age)} laps old" if pd.notna(age) else name)
    return labels


def pace_figure(
    laps: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
    time_column: str = "LapTimeSeconds",
) -> go.Figure:
    figure = go.Figure()
    for driver in drivers:
        driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
        if driver_laps.empty:
            continue
        _add_line(
            figure,
            _line(styles, driver),
            x=driver_laps["LapNumber"],
            y=driver_laps[time_column],
            name=driver,
            mode="lines+markers",
            marker=_marker(styles, driver),
            customdata=np.column_stack(
                [driver_laps[time_column].map(format_seconds), _tyre_labels(driver_laps)]
            ),
            hovertemplate=(
                f"<b>{driver}</b> · lap %{{x}}<br>%{{customdata[0]}} · %{{customdata[1]}}"
                "<extra></extra>"
            ),
        )
    figure.update_xaxes(title="Lap")
    figure.update_yaxes(title=TIME_LABELS.get(time_column, "Lap time (s)"))
    return _finish(figure, 460)


def _compound_order(present: list[str], compound_colors: dict[str, str]) -> list[str]:
    known = [compound for compound in compound_colors if compound in present]
    return known + [compound for compound in present if compound not in compound_colors]


def strategy_figure(
    stints: pd.DataFrame, order: list[str], compound_colors: dict[str, str]
) -> go.Figure:
    figure = go.Figure()
    for compound in _compound_order(stints["Compound"].unique().tolist(), compound_colors):
        rows = stints[stints["Compound"] == compound]
        figure.add_trace(
            go.Bar(
                orientation="h",
                y=rows["Driver"],
                x=rows["Laps"],
                base=rows["StartLap"] - 1,
                name=compound,
                marker={
                    "color": compound_colors.get(compound, FALLBACK_COLOR),
                    "line": {"color": OUTLINE, "width": 1},
                },
                customdata=rows[["Stint", "StartLap", "EndLap"]].to_numpy(),
                hovertemplate=(
                    f"<b>%{{y}}</b> · stint %{{customdata[0]}}<br>{compound} · "
                    "laps %{customdata[1]}–%{customdata[2]} (%{x} laps)<extra></extra>"
                ),
            )
        )
    figure.update_yaxes(
        type="category", categoryorder="array", categoryarray=order, autorange="reversed"
    )
    figure.update_xaxes(title="Lap")
    if not stints.empty:
        figure.update_xaxes(range=[0, int(stints["EndLap"].max()) + 1])
    return _finish(
        figure,
        max(320, 26 * len(order) + 100),
        barmode="overlay",
        bargap=0.3,
        legend_title_text="Compound",
    )


def _corner_label(number: object, letter: object) -> str:
    return f"{number}{letter if isinstance(letter, str) else ''}"


def speed_trace_figure(
    traces: dict[str, pd.DataFrame], styles: Styles, corners: pd.DataFrame
) -> go.Figure:
    figure = go.Figure()
    for driver, trace in traces.items():
        _add_line(
            figure,
            _line(styles, driver),
            x=trace["Distance"],
            y=trace["Speed"],
            name=driver,
            mode="lines",
            hovertemplate=f"{driver} %{{y:.0f}} km/h<extra></extra>",
        )
    if traces:
        lowest = min(trace["Speed"].min() for trace in traces.values())
        highest = max(trace["Speed"].max() for trace in traces.values())
        figure.update_layout(
            shapes=[
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "paper",
                    "x0": distance,
                    "x1": distance,
                    "y0": 0,
                    "y1": 1,
                    "line": {"color": NEUTRAL, "width": 1, "dash": "dot"},
                    "layer": "below",
                }
                for distance in corners["Distance"]
            ],
            annotations=[
                {
                    "x": corner.Distance,
                    "y": lowest - 20,
                    "text": _corner_label(corner.Number, corner.Letter),
                    "showarrow": False,
                    "font": {"size": 11},
                    "opacity": LABEL_OPACITY,
                }
                for corner in corners.itertuples()
            ],
        )
        figure.update_yaxes(range=[lowest - 30, highest + 15])
    figure.update_xaxes(title="Distance (m)", hoverformat=",.0f")
    figure.update_yaxes(title="Speed (km/h)")
    return _finish(figure, 440, hovermode="x unified")


def stint_fit_figure(
    stint_laps: pd.DataFrame, time_column: str = "LapTimeSeconds", color: str = FALLBACK_COLOR
) -> go.Figure:
    x, y = stint_laps["TyreLife"], stint_laps[time_column]
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            name="Laps",
            mode="markers",
            marker={"color": color, "size": 9, "line": {"color": OUTLINE, "width": 1}},
            hovertemplate="Tyre age %{x} laps<br>%{y:.3f} s<extra></extra>",
        )
    )
    if len(stint_laps) >= 2:
        fit = linear_fit(x, y)
        ends = np.array([x.min(), x.max()], dtype=float)
        _add_line(
            figure,
            {"color": color, "dash": "dash", "width": 2},
            x=ends,
            y=fit.slope * ends + fit.intercept,
            name=f"Fit: {fit.slope:+.3f} s/lap",
            mode="lines",
            hoverinfo="skip",
        )
    figure.update_xaxes(title="Tyre age (laps)")
    figure.update_yaxes(title=TIME_LABELS.get(time_column, "Lap time (s)"))
    return _finish(figure, 380)


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


def replay_height(track: pd.DataFrame) -> int:
    width, height = np.ptp(track["X"].to_numpy()), np.ptp(track["Y"].to_numpy())
    estimate = REPLAY_MAP_WIDTH * 1.12 * height / max(width, 1.0) + REPLAY_CONTROLS_HEIGHT
    return int(np.clip(estimate, *REPLAY_HEIGHT_RANGE))


def _corner_labels(map_corners: pd.DataFrame, size: int = 11) -> go.Scatter:
    return go.Scatter(
        x=map_corners["X"],
        y=map_corners["Y"],
        text=map_corners["Label"],
        mode="text",
        textfont={"size": size},
        opacity=LABEL_OPACITY,
        hoverinfo="skip",
        showlegend=False,
    )


def replay_figure(
    track: pd.DataFrame,
    map_corners: pd.DataFrame,
    positions: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
    total_laps: int,
) -> go.Figure:
    width = track["X"].max() - track["X"].min()
    colors = [_style(styles, driver)[0] for driver in drivers]
    symbols = [TEAMMATE_SYMBOLS.get(_style(styles, driver)[1], "circle") for driver in drivers]

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
                line={"color": NEUTRAL, "width": 10},
                hoverinfo="skip",
            ),
            _corner_labels(map_corners, size=10),
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
                    "line": {"width": 1, "color": OUTLINE},
                },
                hovertext=drivers,
                hoverinfo="text",
                cliponaxis=False,
            ),
            go.Scatter(
                x=[0],
                y=[1],
                text=start[1].text,
                mode="text",
                textposition="bottom right",
                textfont={"family": "monospace", "size": 11},
                hoverinfo="skip",
                xaxis="x2",
                yaxis="y2",
                cliponaxis=False,
            ),
        ],
        frames=frames,
    )
    figure.update_layout(
        height=replay_height(track),
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "visible": False,
            "domain": [0, 0.82],
            "range": [track["X"].min() - 0.04 * width, track["X"].max() + 0.04 * width],
        },
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        xaxis2={"visible": False, "domain": [0.84, 1], "range": [0, 1], "fixedrange": True},
        yaxis2={"visible": False, "anchor": "x2", "range": [0, 1], "fixedrange": True},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": 0,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 10},
                "bgcolor": BUTTON,
                "bordercolor": OUTLINE,
                "font": {"color": BUTTON_TEXT},
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
                "bgcolor": NEUTRAL,
                "bordercolor": OUTLINE,
                "activebgcolor": ACCENT,
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
) -> go.Figure:
    highlighted = set(highlight or [])
    figure = go.Figure()
    labels = []
    for driver in sorted(drivers, key=lambda driver: not highlighted or driver in highlighted):
        driver_positions = positions[positions["Driver"] == driver]
        if driver_positions.empty:
            continue
        emphasis = not highlighted or driver in highlighted
        bold = bool(highlighted) and emphasis
        line = _line(styles, driver, width=2.5 if bold else 1.5)
        if not emphasis:
            line["color"] = NEUTRAL
        _add_line(
            figure,
            line,
            x=driver_positions["LapNumber"],
            y=driver_positions["Position"],
            name=driver,
            mode="lines",
            hovertemplate=f"<b>{driver}</b> · lap %{{x}} · P%{{y}}<extra></extra>",
        )
        last = driver_positions.iloc[-1]
        labels.append(
            {
                "x": last["LapNumber"],
                "y": last["Position"],
                "text": f"<b>{driver}</b>" if bold else driver,
                "xanchor": "left",
                "xshift": 6,
                "showarrow": False,
                "font": {"size": 11},
            }
        )
    places = int(positions["Position"].max()) if not positions.empty else 1
    figure.update_yaxes(title="Position", range=[places + 0.5, 0.5], tick0=1, dtick=1)
    figure.update_xaxes(title="Lap")
    return _finish(
        figure,
        max(420, 26 * places + 80),
        showlegend=False,
        annotations=labels,
        margin={"l": 10, "r": 50, "t": 20, "b": 10},
    )


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


def _edge(color: str) -> str:
    return OUTLINE if hard_to_see(color) else color


def team_pace_figure(laps: pd.DataFrame, styles: Styles) -> go.Figure:
    quick = quick_laps(laps)
    colors = team_colors(quick, styles)
    figure = go.Figure()
    for team in team_pace(laps)["Team"]:
        figure.add_trace(
            go.Box(
                y=quick.loc[quick["Team"] == team, "LapTimeSeconds"],
                name=team,
                fillcolor=_rgba(colors[team], 0.55),
                line={"color": _edge(colors[team]), "width": 1.5},
                boxpoints=False,
                yhoverformat=".3f",
            )
        )
    figure.update_xaxes(tickangle=-30)
    figure.update_yaxes(title="Lap time (s)")
    return _finish(figure, 440, showlegend=False)


def lap_distribution_figure(
    laps: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
    compound_colors: dict[str, str],
) -> go.Figure:
    quick = quick_laps(laps)
    groups = [(driver, quick[quick["Driver"] == driver]) for driver in drivers]
    groups = [(driver, rows) for driver, rows in groups if rows["LapTimeSeconds"].nunique() > 1]
    figure = go.Figure()
    rng = np.random.default_rng(0)
    scattered = []
    for position, (driver, driver_laps) in enumerate(groups):
        color = _style(styles, driver)[0]
        figure.add_trace(
            go.Violin(
                x=np.full(len(driver_laps), position),
                y=driver_laps["LapTimeSeconds"],
                name=driver,
                width=0.8,
                points=False,
                fillcolor=_rgba(color, 0.3),
                line={"color": _edge(color), "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        jitter = position + rng.uniform(-0.15, 0.15, len(driver_laps))
        scattered.append(driver_laps.assign(Position=jitter))

    if scattered:
        points = pd.concat(scattered)
        compounds = points["Compound"].fillna(UNKNOWN_COMPOUND)
        for compound in _compound_order(compounds.unique().tolist(), compound_colors):
            rows = points[compounds == compound]
            figure.add_trace(
                go.Scatter(
                    x=rows["Position"],
                    y=rows["LapTimeSeconds"],
                    name=compound,
                    mode="markers",
                    marker={
                        "color": compound_colors.get(compound, FALLBACK_COLOR),
                        "size": 6,
                        "line": {"color": OUTLINE, "width": 0.5},
                    },
                    customdata=np.column_stack(
                        [
                            rows["Driver"],
                            rows["LapNumber"].astype(int),
                            rows["LapTimeSeconds"].map(format_seconds),
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> · lap %{customdata[1]}<br>"
                        f"%{{customdata[2]}} · {compound}<extra></extra>"
                    ),
                )
            )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(groups))),
        ticktext=[driver for driver, _ in groups],
        range=[-0.6, len(groups) - 0.4],
    )
    figure.update_yaxes(title="Lap time (s)")
    return _finish(figure, 460, violinmode="overlay", legend_title_text="Compound")


def _map_figure(track: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=track["X"],
            y=track["Y"],
            mode="lines",
            line={"color": NEUTRAL, "width": 14},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return figure


def speed_map_figure(
    track: pd.DataFrame, map_corners: pd.DataFrame, trace: pd.DataFrame
) -> go.Figure:
    fractions = np.linspace(0.0, 1.0, MAP_POINTS)
    x, y = track_points(track, fractions)
    distance = trace["Distance"].to_numpy(dtype=float)
    speed = np.interp(fractions * distance[-1], distance, trace["Speed"].to_numpy(dtype=float))
    figure = _map_figure(track)
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={
                "color": speed,
                "colorscale": "Plasma",
                "size": 7,
                "colorbar": {
                    "title": {"text": "Speed (km/h)", "side": "top"},
                    "orientation": "h",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": -0.02,
                    "yanchor": "top",
                    "len": 0.6,
                    "thickness": 12,
                },
            },
            hovertemplate="%{marker.color:.0f} km/h<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_trace(_corner_labels(map_corners))
    return _finish(figure, 520)


def mini_sector_times(trace: pd.DataFrame, sectors: int = MINI_SECTORS) -> np.ndarray:
    distance = trace["Distance"].to_numpy(dtype=float)
    boundaries = np.linspace(0.0, 1.0, sectors + 1) * distance[-1]
    return np.diff(np.interp(boundaries, distance, _elapsed(trace)))


def faster_by_sector(traces: dict[str, pd.DataFrame], sectors: int = MINI_SECTORS) -> np.ndarray:
    drivers = list(traces)
    times = np.vstack([mini_sector_times(traces[driver], sectors) for driver in drivers])
    return np.array(drivers)[times.argmin(axis=0)]


def _shade(color: str, factor: float) -> str:
    return "#" + "".join(f"{round(channel * factor):02x}" for channel in _rgb(color))


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
) -> go.Figure:
    drivers = list(traces)
    winners = faster_by_sector(traces, sectors)
    colors = distinct_colors(drivers, styles)
    fractions = np.linspace(0.0, 1.0, MAP_POINTS)
    x, y = track_points(track, fractions)
    sector = np.minimum((fractions * sectors).astype(int), sectors - 1)
    figure = _map_figure(track)
    shown: set[str] = set()
    for number in range(sectors):
        inside = np.flatnonzero(sector == number)
        points = np.append(inside, min(inside[-1] + 1, len(fractions) - 1))
        driver = str(winners[number])
        figure.add_trace(
            go.Scatter(
                x=x[points],
                y=y[points],
                mode="lines",
                line={"color": colors[driver], "width": 6},
                name=f"{driver} ({(winners == driver).sum()} of {sectors})",
                legendgroup=driver,
                legendrank=drivers.index(driver),
                showlegend=driver not in shown,
                hovertemplate=f"Mini-sector {number + 1}: {driver} faster<extra></extra>",
            )
        )
        shown.add(driver)
    figure.add_trace(_corner_labels(map_corners))
    return _finish(figure, 520, legend_title_text="Faster in")


@dataclass(frozen=True)
class FastestLap:
    driver: str
    lap: int
    time: pd.Timedelta


@dataclass(frozen=True)
class RaceSummary:
    fastest: FastestLap | None
    pit_stops: int
    safety_car_laps: int
    virtual_safety_car_laps: int
    red_flag: bool
    laps_led: dict[str, int]
    lead_changes: int


def race_summary(laps: pd.DataFrame, winner: str) -> RaceSummary:
    personal_bests = laps[laps["IsPersonalBest"]].dropna(subset=["LapTime"])
    fastest = None
    if not personal_bests.empty:
        best = personal_bests.loc[personal_bests["LapTime"].idxmin()]
        fastest = FastestLap(str(best["Driver"]), int(best["LapNumber"]), best["LapTime"])

    status = laps["TrackStatus"].fillna("").astype(str)
    red_flag = status.str.contains(RED_FLAG)
    winner_status = status[laps["Driver"] == winner]
    safety_car = winner_status.str.contains(SAFETY_CAR)
    virtual = winner_status.apply(lambda codes: any(code in codes for code in VIRTUAL_SAFETY_CAR))

    leaders = pd.Series(dtype=str)
    if "Time" in laps.columns:
        positions = lap_positions(laps)
        leaders = positions[positions["Position"] == 1].sort_values("LapNumber")["Driver"]
    led = leaders.value_counts()

    return RaceSummary(
        fastest=fastest,
        pit_stops=int((laps["PitInTime"].notna() & ~red_flag).sum()),
        safety_car_laps=int(safety_car.sum()),
        virtual_safety_car_laps=int((virtual & ~safety_car).sum()),
        red_flag=bool(red_flag.any()),
        laps_led={str(driver): int(count) for driver, count in led.items()},
        lead_changes=max(int((leaders != leaders.shift()).sum()) - 1, 0),
    )
