from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import analysis
from src.charts import OUTLINE, finish

MAX_PIT_LANE = 60.0
UNDERCUT_GAP = 3.0
UNDERCUT_WINDOW = 5
STOP_COLUMNS = ["Driver", "Team", "Lap", "PitLane", "TimeLost", "From", "To"]
UNDERCUT_COLUMNS = ["Lap", "Attacker", "Defender", "Gap", "Worked"]


def pit_stops(laps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    green = analysis.representative_laps(laps)
    typical = green.groupby("Driver")["LapTimeSeconds"].median()
    for driver, driver_laps in laps.sort_values("LapNumber").groupby("Driver"):
        driver_laps = driver_laps.reset_index(drop=True)
        for index in driver_laps.index[driver_laps["PitInTime"].notna()]:
            if index + 1 >= len(driver_laps):
                continue
            in_lap, out_lap = driver_laps.loc[index], driver_laps.loc[index + 1]
            if pd.isna(out_lap["PitOutTime"]):
                continue
            pit_lane = (out_lap["PitOutTime"] - in_lap["PitInTime"]).total_seconds()
            if not 0 < pit_lane <= MAX_PIT_LANE:
                continue
            lost = (
                in_lap["LapTimeSeconds"]
                + out_lap["LapTimeSeconds"]
                - 2 * typical.get(driver, np.nan)
            )
            rows.append(
                {
                    "Driver": driver,
                    "Team": in_lap["Team"],
                    "Lap": int(in_lap["LapNumber"]),
                    "PitLane": pit_lane,
                    "TimeLost": lost,
                    "From": in_lap["Compound"],
                    "To": out_lap["Compound"],
                }
            )
    return pd.DataFrame(rows, columns=STOP_COLUMNS)


def pit_loss(stops: pd.DataFrame) -> float | None:
    lost = stops["TimeLost"].dropna()
    lost = lost[(lost > 5) & (lost < 60)]
    return float(lost.median()) if not lost.empty else None


def pit_stop_figure(stops: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    order = stops.groupby("Team")["PitLane"].median().sort_values().index.tolist()
    figure = go.Figure()
    for team in order:
        rows = stops[stops["Team"] == team]
        figure.add_trace(
            go.Box(
                x=[team] * len(rows),
                y=rows["PitLane"],
                name=team,
                boxpoints="all",
                pointpos=0,
                jitter=0.3,
                fillcolor="rgba(0, 0, 0, 0)",
                line={"color": OUTLINE, "width": 1},
                marker={
                    "color": colors.get(team, OUTLINE),
                    "size": 8,
                    "line": {"color": OUTLINE, "width": 1},
                },
                customdata=np.column_stack([rows["Driver"], rows["Lap"]]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · lap %{customdata[1]}<br>%{y:.1f} s<extra></extra>"
                ),
            )
        )
    figure.update_yaxes(title="Time in the pit lane (s)")
    figure.update_xaxes(tickangle=-30)
    return finish(figure, 400, showlegend=False)


def undercuts(
    laps: pd.DataFrame,
    stops: pd.DataFrame,
    gap: float = UNDERCUT_GAP,
    window: int = UNDERCUT_WINDOW,
) -> pd.DataFrame:
    timed = laps.dropna(subset=["Time"]).assign(
        Seconds=lambda rows: rows["Time"].dt.total_seconds()
    )
    positions = analysis.lap_positions(laps).set_index(["LapNumber", "Driver"])["Position"]
    seconds = timed.set_index(["LapNumber", "Driver"])["Seconds"]
    pitted = {(row.Driver, row.Lap) for row in stops.itertuples()}
    rows = []
    for stop in stops.sort_values("Lap").itertuples():
        before = stop.Lap - 1
        if (before, stop.Driver) not in positions.index:
            continue
        place = positions[(before, stop.Driver)]
        ahead = positions.loc[before]
        ahead = ahead[ahead == place - 1].index.tolist()
        if not ahead:
            continue
        rival = ahead[0]
        interval = seconds.get((before, stop.Driver)) - seconds.get((before, rival))
        if not interval <= gap:
            continue
        reply = next(
            (lap for lap in range(stop.Lap + 1, stop.Lap + window + 1) if (rival, lap) in pitted),
            None,
        )
        if reply is None:
            continue
        after = reply + 1
        if (after, stop.Driver) not in positions.index or (after, rival) not in positions.index:
            continue
        rows.append(
            {
                "Lap": stop.Lap,
                "Attacker": stop.Driver,
                "Defender": rival,
                "Gap": interval,
                "Worked": bool(positions[(after, stop.Driver)] < positions[(after, rival)]),
            }
        )
    return pd.DataFrame(rows, columns=UNDERCUT_COLUMNS)
