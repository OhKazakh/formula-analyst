from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis import stint_compound
from src.charts import OUTLINE, Styles, driver_style, finish

TOWER_COLUMNS = [
    "Position",
    "Driver",
    "Team",
    "Laps",
    "Gap",
    "LapsDown",
    "Interval",
    "IntervalLaps",
    "LastLap",
    "BestLap",
    "Compound",
    "TyreAge",
    "Stints",
    "InPit",
    "Out",
    "LastPersonalBest",
    "LastFastest",
    "BestFastest",
]


def finishers(laps: pd.DataFrame) -> set[str]:
    timed = laps.dropna(subset=["Time"])
    if timed.empty:
        return set()
    flag = timed.loc[timed["LapNumber"] == timed["LapNumber"].max(), "Time"].min()
    last = timed.groupby("Driver")["Time"].max()
    # Lapped cars cross the line after the winner takes the flag; anyone who stopped before
    # that retired.
    return set(last[last >= flag].index)


def stint_history(driver_laps: pd.DataFrame) -> list[tuple[str, int]]:
    stints = (
        driver_laps.dropna(subset=["Stint"])
        .groupby("Stint", sort=True)
        .agg(Compound=("Compound", stint_compound), Laps=("LapNumber", "count"))
    )
    return list(zip(stints["Compound"], stints["Laps"].astype(int), strict=True))


def tower(laps: pd.DataFrame, lap: int) -> pd.DataFrame:
    timed = laps.dropna(subset=["Time"])
    played = timed[timed["LapNumber"] <= lap]
    if played.empty:
        return pd.DataFrame(columns=TOWER_COLUMNS)
    finished = finishers(laps)
    last_lap = timed.groupby("Driver")["LapNumber"].max()
    latest = played.sort_values("LapNumber").groupby("Driver").tail(1)
    latest = latest.assign(
        Out=[driver not in finished and last_lap[driver] < lap for driver in latest["Driver"]],
        Seconds=latest["Time"].dt.total_seconds(),
    ).sort_values(["Out", "LapNumber", "Seconds"], ascending=[True, False, True])

    bests = laps[laps["IsPersonalBest"] & (laps["LapNumber"] <= lap)]
    best = bests.groupby("Driver")["LapTimeSeconds"].min()
    fastest = best.min()
    upto = laps[laps["LapNumber"] <= lap]
    histories = {driver: stint_history(rows) for driver, rows in upto.groupby("Driver")}

    rows = []
    leader = latest.iloc[0]
    ahead = None
    for position, row in enumerate(latest.itertuples(), start=1):
        same_lap = not row.Out and row.LapNumber == leader.LapNumber
        running_ahead = ahead is not None and not row.Out
        best_lap = best.get(row.Driver, np.nan)
        rows.append(
            {
                "Position": position,
                "Driver": row.Driver,
                "Team": row.Team,
                "Laps": int(row.LapNumber),
                "Gap": row.Seconds - leader.Seconds if same_lap else np.nan,
                "LapsDown": 0 if row.Out else int(leader.LapNumber - row.LapNumber),
                "Interval": (
                    row.Seconds - ahead.Seconds
                    if running_ahead and ahead.LapNumber == row.LapNumber
                    else np.nan
                ),
                "IntervalLaps": (int(ahead.LapNumber - row.LapNumber) if running_ahead else 0),
                "LastLap": row.LapTimeSeconds,
                "BestLap": best_lap,
                "Compound": row.Compound,
                "TyreAge": row.TyreLife,
                "Stints": histories.get(row.Driver, []),
                "InPit": pd.notna(row.PitInTime),
                "Out": row.Out,
                "LastPersonalBest": bool(row.IsPersonalBest),
                "LastFastest": bool(row.IsPersonalBest) and row.LapTimeSeconds == fastest,
                "BestFastest": best_lap == fastest,
            }
        )
        if not row.Out:
            ahead = row
    return pd.DataFrame(rows, columns=TOWER_COLUMNS)


def spread_figure(
    standing: pd.DataFrame, styles: Styles, highlight: str | None = None
) -> go.Figure:
    on_lead_lap = standing.dropna(subset=["Gap"])
    colors, sizes = [], []
    for driver in on_lead_lap["Driver"]:
        colors.append(driver_style(styles, driver)[0])
        sizes.append(16 if driver == highlight else 11)
    figure = go.Figure(
        go.Scatter(
            x=on_lead_lap["Gap"],
            y=np.zeros(len(on_lead_lap)),
            text=on_lead_lap["Driver"],
            mode="markers+text",
            textposition=[
                "top center" if i % 2 == 0 else "bottom center" for i in range(len(on_lead_lap))
            ],
            textfont={"size": 10},
            marker={"color": colors, "size": sizes, "line": {"color": OUTLINE, "width": 1}},
            customdata=on_lead_lap["Position"],
            hovertemplate="P%{customdata} %{text}<br>+%{x:.1f} s<extra></extra>",
            cliponaxis=False,
        )
    )
    figure.update_yaxes(visible=False, range=[-1, 1])
    figure.update_xaxes(title="Gap to the leader (s)", rangemode="tozero", zeroline=False)
    return finish(figure, 170, showlegend=False, margin={"l": 10, "r": 10, "t": 10, "b": 10})


def lap_times(laps: pd.DataFrame, drivers: list[str]) -> pd.DataFrame:
    chosen = laps[laps["Driver"].isin(drivers)]
    personal = chosen[chosen["IsPersonalBest"]].groupby("Driver")["LapTimeSeconds"].min()
    fastest = laps.loc[laps["IsPersonalBest"], "LapTimeSeconds"].min()
    return chosen.assign(
        PersonalBest=lambda rows: (
            rows["IsPersonalBest"] & (rows["LapTimeSeconds"] == rows["Driver"].map(personal))
        ),
        Fastest=lambda rows: rows["IsPersonalBest"] & (rows["LapTimeSeconds"] == fastest),
        Pit=lambda rows: rows["PitInTime"].notna() | rows["PitOutTime"].notna(),
    )[
        [
            "Driver",
            "LapNumber",
            "LapTimeSeconds",
            "Compound",
            "TyreLife",
            "Pit",
            "PersonalBest",
            "Fastest",
        ]
    ].sort_values(["LapNumber", "Driver"], ignore_index=True)
