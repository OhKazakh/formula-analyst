from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

FALLBACK_COLOR = "#888888"
ACCENT = "#E10600"
NEUTRAL = "rgba(128, 128, 128, 0.35)"
OUTLINE = "rgba(128, 128, 128, 0.8)"
LIGHT_BACKGROUND = "#FFFFFF"
DARK_BACKGROUND = "#15151E"
LOW_CONTRAST = 1.6
LABEL_OPACITY = 0.75
PLOTLY_DASHES = {"solid": "solid", "dashed": "dash", "dashdot": "dashdot", "dotted": "dot"}
TEAMMATE_SYMBOLS = {"solid": "circle", "dashed": "square", "dashdot": "diamond", "dotted": "x"}

Styles = dict[str, dict[str, str]]


def rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgba(color: str, alpha: float) -> str:
    red, green, blue = rgb(color)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _luminance(color: str) -> float:
    channels = np.array(rgb(color)) / 255
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def contrast(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


# A few official colours (white Williams, yellow Renault) vanish on one of the backgrounds.
def hard_to_see(color: str) -> bool:
    if not color.startswith("#"):
        return False
    backgrounds = (LIGHT_BACKGROUND, DARK_BACKGROUND)
    return min(contrast(color, background) for background in backgrounds) < LOW_CONTRAST


def edge(color: str) -> str:
    return OUTLINE if hard_to_see(color) else color


def driver_style(styles: Styles, driver: str) -> tuple[str, str]:
    style = styles.get(driver, {})
    return style.get("color", FALLBACK_COLOR), style.get("linestyle", "solid")


def line_style(styles: Styles, driver: str, width: float = 2) -> dict:
    color, linestyle = driver_style(styles, driver)
    return {"color": color, "dash": PLOTLY_DASHES.get(linestyle, "solid"), "width": width}


def marker_style(styles: Styles, driver: str, size: int = 7) -> dict:
    color, linestyle = driver_style(styles, driver)
    symbol = TEAMMATE_SYMBOLS.get(linestyle, "circle")
    if linestyle == "solid":
        return {
            "symbol": symbol,
            "color": color,
            "size": size,
            "line": {"color": OUTLINE, "width": 1},
        }
    return {"symbol": f"{symbol}-open", "color": color, "size": size, "line": {"width": 2}}


def halo(x: object, y: object, line: dict, **trace: object) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={**line, "color": OUTLINE, "width": line["width"] + 2.5},
        hoverinfo="skip",
        showlegend=False,
        **trace,
    )


def add_line(figure: go.Figure, line: dict, **trace: object) -> None:
    if hard_to_see(line["color"]):
        axes = {key: trace[key] for key in ("xaxis", "yaxis") if key in trace}
        figure.add_trace(halo(trace["x"], trace["y"], line, **axes))
    figure.add_trace(go.Scatter(line=line, **trace))


def finish(figure: go.Figure, height: int, **layout: object) -> go.Figure:
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


def compound_order(present: list[str], compound_colors: dict[str, str]) -> list[str]:
    known = [compound for compound in compound_colors if compound in present]
    return known + [compound for compound in present if compound not in compound_colors]


def corner_labels(map_corners: pd.DataFrame, size: int = 11) -> go.Scatter:
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


def track_figure(track: pd.DataFrame) -> go.Figure:
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


def track_points(track: pd.DataFrame, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = track[["X", "Y"]].to_numpy(dtype=float)
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    target = (np.asarray(fractions) % 1.0) * distance[-1]
    return np.interp(target, distance, xy[:, 0]), np.interp(target, distance, xy[:, 1])
