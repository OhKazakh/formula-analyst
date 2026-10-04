from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.charts import FALLBACK_COLOR, OUTLINE, PLOTLY_DASHES, Styles

RACE_WIN_POINTS = 25
FASTEST_LAP_SEASONS = range(2019, 2025)


def sprint_win_points(year: int) -> int:
    if year >= 2022:
        return 8
    if year == 2021:
        return 3
    return 0


def max_round_points(year: int, sprint: bool) -> int:
    race = RACE_WIN_POINTS + (1 if year in FASTEST_LAP_SEASONS else 0)
    return race + (sprint_win_points(year) if sprint else 0)


def remaining_points(schedule: pd.DataFrame, year: int, after_round: int) -> int:
    remaining = schedule[schedule["Round"] > after_round]
    return sum(max_round_points(year, bool(sprint)) for sprint in remaining["Sprint"])


def standings(results: pd.DataFrame, after_round: int) -> pd.DataFrame:
    played = results[results["Round"] <= after_round]
    finishes = (
        played[(played["Session"] == "Race") & played["Classified"]]
        .pivot_table(index="Driver", columns="Position", values="Round", aggfunc="count")
        .reindex(columns=range(1, int(played["Position"].max()) + 1))
        .fillna(0)
    )
    table = played.groupby("Driver")["Points"].sum().to_frame().join(finishes).fillna(0)
    table = table.sort_values(["Points", *finishes.columns], ascending=False)
    latest = played.sort_values("Round").groupby("Driver")[["Name", "Team"]].last()
    return pd.DataFrame(
        {
            "Position": range(1, len(table) + 1),
            "Driver": table.index,
            "Name": latest.loc[table.index, "Name"].to_numpy(),
            "Team": latest.loc[table.index, "Team"].to_numpy(),
            "Points": table["Points"].to_numpy(),
            "Wins": table.get(1, pd.Series(0, index=table.index)).astype(int).to_numpy(),
        }
    )


def title_contenders(table: pd.DataFrame, remaining: int) -> pd.DataFrame:
    leader = table["Points"].max()
    maximum = table["Points"] + remaining
    return table.assign(Maximum=maximum, CanWin=maximum >= leader)


def points_by_round(results: pd.DataFrame) -> pd.DataFrame:
    return results.pivot_table(index="Driver", columns="Round", values="Points", aggfunc="sum")


def _short_name(event: str) -> str:
    return event.removesuffix(" Grand Prix")


def points_heatmap(
    results: pd.DataFrame, schedule: pd.DataFrame, order: list[str], upto_round: int
) -> go.Figure:
    played = results[results["Round"] <= upto_round]
    points = points_by_round(played).reindex(index=order)
    rounds = list(points.columns)
    names = schedule.set_index("Round")["EventName"]
    labels = [_short_name(str(names.get(r, r))) for r in rounds]
    race = played[played["Session"] == "Race"].pivot_table(
        index="Driver", columns="Round", values="Position"
    )
    sprint = played[played["Session"] == "Sprint"].pivot_table(
        index="Driver", columns="Round", values="Position"
    )
    detail = [
        [_finish_text(race, sprint, driver, round_number) for round_number in rounds]
        for driver in order
    ]

    figure = make_subplots(
        rows=1, cols=2, column_widths=[0.9, 0.1], shared_yaxes=True, horizontal_spacing=0.01
    )
    figure.add_trace(
        go.Heatmap(
            z=points.to_numpy(),
            x=labels,
            y=order,
            text=points.fillna("").to_numpy(),
            texttemplate="%{text}",
            customdata=detail,
            hovertemplate="%{y} · %{x}<br>%{z} points<br>%{customdata}<extra></extra>",
            showscale=False,
            xgap=1,
            ygap=1,
        ),
        row=1,
        col=1,
    )
    totals = points.sum(axis=1)
    figure.add_trace(
        go.Heatmap(
            z=totals.to_numpy()[:, None],
            x=["Total"],
            y=order,
            text=totals.round(1).to_numpy()[:, None],
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{z} points<extra></extra>",
            showscale=False,
            xgap=1,
            ygap=1,
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(autorange="reversed", row=1, col=1)
    figure.update_xaxes(side="top", tickangle=-45)
    figure.update_layout(
        height=max(400, 26 * len(order) + 140),
        margin={"l": 10, "r": 10, "t": 110, "b": 10},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _finish_text(race: pd.DataFrame, sprint: pd.DataFrame, driver: str, round_number: int) -> str:
    parts = []
    for label, table in (("Race", race), ("Sprint", sprint)):
        if driver in table.index and round_number in table.columns:
            position = table.at[driver, round_number]
            if pd.notna(position):
                parts.append(f"{label} P{int(position)}")
    return " · ".join(parts)


def progression_figure(
    results: pd.DataFrame,
    schedule: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
    upto_round: int,
) -> go.Figure:
    played = results[results["Round"] <= upto_round]
    cumulative = points_by_round(played).fillna(0).cumsum(axis=1)
    names = schedule.set_index("Round")["EventName"]
    labels = [_short_name(str(names.get(r, r))) for r in cumulative.columns]
    figure = go.Figure()
    for driver in drivers:
        if driver not in cumulative.index:
            continue
        style = styles.get(driver, {})
        figure.add_trace(
            go.Scatter(
                x=labels,
                y=cumulative.loc[driver].to_numpy(),
                name=driver,
                mode="lines+markers",
                marker={"size": 6, "line": {"color": OUTLINE, "width": 1}},
                line={
                    "color": style.get("color", FALLBACK_COLOR),
                    "dash": PLOTLY_DASHES.get(style.get("linestyle", "solid"), "solid"),
                },
                hovertemplate=f"{driver} · %{{x}}<br>%{{y}} points<extra></extra>",
            )
        )
    figure.update_layout(
        height=480,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="Points",
        xaxis={"tickangle": -45},
        legend={"orientation": "v"},
    )
    return figure
