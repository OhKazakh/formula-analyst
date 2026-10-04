from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src import telemetry
from src.charts import OUTLINE, Styles, driver_style, finish

SECTORS = ["Sector1", "Sector2", "Sector3"]
TRAPS = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]


def has_sectors(laps: pd.DataFrame) -> bool:
    return all(column in laps for column in SECTORS) and laps[SECTORS].notna().any().all()


def theoretical_best(laps: pd.DataFrame) -> pd.DataFrame:
    timed = laps[laps["IsAccurate"]].dropna(subset=["LapTimeSeconds"])
    best = timed.groupby("Driver")[SECTORS].min()
    table = pd.DataFrame(
        {
            "Fastest": timed.groupby("Driver")["LapTimeSeconds"].min(),
            "Theoretical": best.sum(axis=1, min_count=3),
        }
    ).join(best)
    table["Gain"] = table["Fastest"] - table["Theoretical"]
    return table.dropna(subset=["Theoretical"]).sort_values("Theoretical").reset_index()


def sector_leaders(laps: pd.DataFrame) -> pd.DataFrame:
    timed = laps[laps["IsAccurate"]]
    rows = []
    for number, sector in enumerate(SECTORS, start=1):
        times = timed.dropna(subset=[sector])
        if times.empty:
            continue
        best = times.loc[times[sector].idxmin()]
        rows.append(
            {
                "Sector": number,
                "Driver": best["Driver"],
                "Team": best["Team"],
                "Lap": int(best["LapNumber"]),
                "Time": float(best[sector]),
            }
        )
    return pd.DataFrame(rows, columns=["Sector", "Driver", "Team", "Lap", "Time"])


def speed_profile(
    laps: pd.DataFrame, traces: dict[str, pd.DataFrame], corners: pd.DataFrame
) -> pd.DataFrame:
    top_speed = laps.groupby("Driver")["SpeedST"].max() if "SpeedST" in laps else pd.Series()
    rows = []
    for driver, trace in traces.items():
        if trace.empty:
            continue
        apexes = telemetry.corner_speeds(trace, corners)["MinSpeed"]
        trap = top_speed.get(driver)
        rows.append(
            {
                "Driver": driver,
                "TopSpeed": float(trap) if pd.notna(trap) else float(trace["Speed"].max()),
                "CornerSpeed": float(apexes.mean()) if not apexes.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=["Driver", "TopSpeed", "CornerSpeed"]).dropna()


def speed_profile_figure(profile: pd.DataFrame, styles: Styles) -> go.Figure:
    colors = [driver_style(styles, driver)[0] for driver in profile["Driver"]]
    figure = go.Figure(
        go.Scatter(
            x=profile["TopSpeed"],
            y=profile["CornerSpeed"],
            text=profile["Driver"],
            mode="markers+text",
            textposition="top center",
            textfont={"size": 10},
            marker={"color": colors, "size": 12, "line": {"color": OUTLINE, "width": 1}},
            hovertemplate=(
                "<b>%{text}</b><br>top speed %{x:.0f} km/h<br>"
                "average apex %{y:.0f} km/h<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title="Top speed at the speed trap (km/h)")
    figure.update_yaxes(title="Average minimum speed in corners (km/h)")
    return finish(figure, 440, showlegend=False)
