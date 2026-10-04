from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import colorbrewer
from plotly.subplots import make_subplots

from src.charts import (
    LABEL_OPACITY,
    NEUTRAL,
    Styles,
    add_line,
    corner_labels,
    finish,
    line_style,
    track_figure,
    track_points,
)

CHANNELS = {
    "Throttle": "Throttle (%)",
    "Brake": "Brake",
    "Gear": "Gear",
    "RPM": "RPM",
    "DRS": "DRS",
}
ROW_HEIGHTS = {"Speed": 3.0, "Delta": 1.6, "Throttle": 1.4, "Brake": 0.7, "Gear": 1.2, "RPM": 1.2}
STEPPED = {"Brake", "Gear", "DRS"}
DELTA_POINTS = 1000
FULL_THROTTLE = 98
GEAR_COLORS = colorbrewer.Paired[:8]
MAP_POINTS = 1200


def elapsed(trace: pd.DataFrame, lap_time: float | None = None) -> np.ndarray:
    if "Time" in trace and trace["Time"].notna().all():
        time = trace["Time"].to_numpy(dtype=float)
        time = time - time[0]
    else:
        distance = trace["Distance"].to_numpy(dtype=float)
        speed = np.maximum(trace["Speed"].to_numpy(dtype=float) / 3.6, 1.0)
        time = np.cumsum(np.diff(distance, prepend=distance[0]) / speed)
    # Car data drifts a little over a lap, so stretch it to end on the official lap time.
    if lap_time and time[-1] > 0:
        time = time * lap_time / time[-1]
    return time


def lap_delta(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    lap_times: tuple[float | None, float | None] = (None, None),
    points: int = DELTA_POINTS,
) -> pd.DataFrame:
    end = min(reference["Distance"].max(), other["Distance"].max())
    distance = np.linspace(0.0, end, points)
    reference_time, other_time = lap_times
    gap = np.interp(distance, other["Distance"], elapsed(other, other_time)) - np.interp(
        distance, reference["Distance"], elapsed(reference, reference_time)
    )
    return pd.DataFrame({"Distance": distance, "Delta": gap})


def channels(traces: dict[str, pd.DataFrame]) -> list[str]:
    return [
        channel
        for channel in CHANNELS
        if all(channel in trace and trace[channel].notna().any() for trace in traces.values())
    ]


def _axis(row: int, kind: str) -> str:
    return kind if row == 1 else f"{kind}{row}"


def telemetry_figure(
    traces: dict[str, pd.DataFrame],
    styles: Styles,
    corners: pd.DataFrame,
    lap_times: dict[str, float] | None = None,
) -> go.Figure:
    drivers = list(traces)
    rows = ["Speed"]
    if len(drivers) == 2:
        rows.append("Delta")
    rows += channels(traces)
    figure = make_subplots(
        rows=len(rows),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[ROW_HEIGHTS.get(row, 0.7) for row in rows],
    )
    for index, row in enumerate(rows, start=1):
        axes = {"xaxis": _axis(index, "x"), "yaxis": _axis(index, "y")}
        if row == "Delta":
            reference, other = drivers
            times = lap_times or {}
            delta = lap_delta(
                traces[reference], traces[other], (times.get(reference), times.get(other))
            )
            add_line(
                figure,
                {**line_style(styles, other), "dash": "solid"},
                x=delta["Distance"],
                y=delta["Delta"],
                name=f"{other} to {reference}",
                mode="lines",
                showlegend=False,
                hovertemplate=f"{other} %{{y:+.3f}} s<extra></extra>",
                **axes,
            )
            figure.add_hline(y=0, line={"color": NEUTRAL, "width": 1}, row=index, col=1)
            figure.update_yaxes(title=f"Gap to {reference} (s)", row=index, col=1)
            continue
        for driver, trace in traces.items():
            values = trace[row].astype(float)
            add_line(
                figure,
                {
                    **line_style(styles, driver, width=1.6),
                    "shape": "hv" if row in STEPPED else "linear",
                },
                x=trace["Distance"],
                y=values,
                name=driver,
                legendgroup=driver,
                showlegend=index == 1,
                mode="lines",
                hovertemplate=f"{driver} %{{y:.0f}}<extra></extra>",
                **axes,
            )
        title = "Speed (km/h)" if row == "Speed" else CHANNELS[row]
        figure.update_yaxes(title=title, row=index, col=1)
        if row in {"Brake", "DRS"}:
            labels = ["Off", "On"] if row == "Brake" else ["Shut", "Open"]
            figure.update_yaxes(tickvals=[0, 1], ticktext=labels, row=index, col=1)
    for corner in corners.itertuples():
        figure.add_shape(
            type="line",
            x0=corner.Distance,
            x1=corner.Distance,
            xref="x",
            y0=0,
            y1=1,
            yref="paper",
            line={"color": NEUTRAL, "width": 1, "dash": "dot"},
            layer="below",
        )
        figure.add_annotation(
            x=corner.Distance,
            y=0,
            xref="x",
            yref="y domain",
            yanchor="bottom",
            text=f"{corner.Number}{corner.Letter if isinstance(corner.Letter, str) else ''}",
            showarrow=False,
            font={"size": 10},
            opacity=LABEL_OPACITY,
        )
    figure.update_xaxes(title="Distance (m)", hoverformat=",.0f", row=len(rows), col=1)
    height = int(sum(ROW_HEIGHTS.get(row, 0.7) for row in rows) * 110 + 120)
    return finish(figure, height, hovermode="x unified")


def _windows(distances: np.ndarray, lap_length: float) -> list[tuple[float, float]]:
    bounds = np.concatenate([[0.0], (distances[1:] + distances[:-1]) / 2, [lap_length]])
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def corner_speeds(trace: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    columns = ["Corner", "MinSpeed", "BrakingPoint"]
    if trace.empty or corners.empty:
        return pd.DataFrame(columns=columns)
    distance = trace["Distance"].to_numpy(dtype=float)
    speed = trace["Speed"].to_numpy(dtype=float)
    brake = trace["Brake"].to_numpy(dtype=bool) if "Brake" in trace else None
    corner_distances = corners["Distance"].to_numpy(dtype=float)
    rows = []
    for corner, (start, end) in zip(
        corners.itertuples(), _windows(corner_distances, distance[-1]), strict=True
    ):
        inside = np.flatnonzero((distance >= start) & (distance <= end))
        if inside.size == 0:
            continue
        apex = inside[np.argmin(speed[inside])]
        braking = np.nan
        if brake is not None:
            before = inside[inside <= apex]
            onsets = before[brake[before] & ~np.roll(brake, 1)[before]]
            if onsets.size:
                braking = corner.Distance - distance[onsets[-1]]
        letter = corner.Letter if isinstance(corner.Letter, str) else ""
        rows.append(
            {
                "Corner": f"{corner.Number}{letter}",
                "MinSpeed": float(speed[apex]),
                "BrakingPoint": braking,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def corner_comparison(traces: dict[str, pd.DataFrame], corners: pd.DataFrame) -> pd.DataFrame:
    tables = {driver: corner_speeds(trace, corners) for driver, trace in traces.items()}
    drivers = list(tables)
    merged = tables[drivers[0]][["Corner"]]
    for driver, table in tables.items():
        merged = merged.merge(
            table.rename(columns={"MinSpeed": f"{driver} min", "BrakingPoint": f"{driver} brakes"}),
            on="Corner",
            how="left",
        )
    if len(drivers) == 2:
        first, second = drivers
        merged["Difference"] = merged[f"{second} min"] - merged[f"{first} min"]
    return merged


def gear_runs(trace: pd.DataFrame, track: pd.DataFrame) -> list[tuple[int, np.ndarray, np.ndarray]]:
    fractions = np.linspace(0.0, 1.0, MAP_POINTS)
    x, y = track_points(track, fractions)
    distance = trace["Distance"].to_numpy(dtype=float)
    position = np.clip(np.searchsorted(distance, fractions * distance[-1]), 0, len(distance) - 1)
    gears = trace["Gear"].to_numpy(dtype=int)[position]
    starts = np.flatnonzero(np.diff(gears, prepend=gears[0] - 1))
    ends = np.append(starts[1:], len(gears) - 1)
    return [
        (int(gears[start]), x[start : end + 1], y[start : end + 1])
        for start, end in zip(starts, ends, strict=True)
    ]


def gear_map_figure(
    track: pd.DataFrame, map_corners: pd.DataFrame, trace: pd.DataFrame
) -> go.Figure:
    figure = track_figure(track)
    shown: set[int] = set()
    for gear, x, y in gear_runs(trace, track):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": GEAR_COLORS[(gear - 1) % len(GEAR_COLORS)], "width": 6},
                name=f"Gear {gear}",
                legendgroup=str(gear),
                legendrank=gear,
                showlegend=gear not in shown,
                hovertemplate=f"Gear {gear}<extra></extra>",
            )
        )
        shown.add(gear)
    figure.add_trace(corner_labels(map_corners))
    return finish(figure, 520)
