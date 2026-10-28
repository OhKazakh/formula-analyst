from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.charts import FALLBACK_COLOR, OUTLINE, PLOTLY_DASHES, Styles

RACE_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
FASTEST_LAP_SEASONS = range(2019, 2025)
TEAM_COLUMNS = ["Position", "Team", "Points", "Wins"]


def sprint_points(year: int) -> tuple[int, ...]:
    if year >= 2022:
        return 8, 7, 6, 5, 4, 3, 2, 1
    if year == 2021:
        return 3, 2, 1
    return ()


def round_points(year: int, sprint: bool, places: slice, fastest_lap: bool = False) -> int:
    race = sum(RACE_POINTS[places]) + (1 if fastest_lap and year in FASTEST_LAP_SEASONS else 0)
    return race + (sum(sprint_points(year)[places]) if sprint else 0)


# A team can score with both cars, so its best weekend is a one-two.
def max_round_points(year: int, sprint: bool, cars: int = 1) -> int:
    return round_points(year, sprint, slice(0, cars), fastest_lap=True)


def remaining_points(schedule: pd.DataFrame, year: int, after_round: int, cars: int = 1) -> int:
    remaining = schedule[schedule["Round"] > after_round]
    return sum(max_round_points(year, bool(sprint), cars) for sprint in remaining["Sprint"])


# The most a rival can still score while someone else takes the top places every round.
def runner_up_points(schedule: pd.DataFrame, year: int, after_round: int, cars: int = 1) -> int:
    remaining = schedule[schedule["Round"] > after_round]
    places = slice(cars, 2 * cars)
    return sum(round_points(year, bool(sprint), places) for sprint in remaining["Sprint"])


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


def team_standings(official: pd.DataFrame, after_round: int) -> pd.DataFrame:
    played = official[official["Round"] <= after_round]
    if played.empty:
        return pd.DataFrame(columns=TEAM_COLUMNS)
    latest = played[played["Round"] == played["Round"].max()]
    ordered = latest.sort_values(
        ["Position", "Points"], ascending=[True, False], na_position="last"
    )
    return ordered[TEAM_COLUMNS].astype({"Position": "Int64", "Wins": int}).reset_index(drop=True)


def standings_change(after: pd.DataFrame, before: pd.DataFrame, key: str) -> pd.DataFrame:
    previous = before.set_index(key)
    return after.assign(
        Gained=after["Points"] - after[key].map(previous["Points"]).fillna(0),
        Moved=after[key].map(previous["Position"]) - after["Position"],
    )


def team_colors(results: pd.DataFrame, styles: Styles) -> dict[str, str]:
    colors: dict[str, str] = {}
    latest = results.sort_values("Round").drop_duplicates("Driver", keep="last")
    for driver, team in zip(latest["Driver"], latest["Team"], strict=True):
        if color := styles.get(driver, {}).get("color"):
            colors.setdefault(team, color)
    return colors


def driver_teams(results: pd.DataFrame) -> dict[str, str]:
    latest = results.sort_values("Round").drop_duplicates("Driver", keep="last")
    return dict(zip(latest["Driver"], latest["Team"], strict=True))


def penalties(official: pd.DataFrame, results: pd.DataFrame, after_round: int) -> dict[str, float]:
    scored = results[results["Round"] <= after_round].groupby("Team")["Points"].sum()
    standings = team_standings(official, after_round).set_index("Team")["Points"]
    difference = (standings - scored.reindex(standings.index).fillna(0)).round(1)
    return {team: float(points) for team, points in difference.items() if points}


def title_contenders(table: pd.DataFrame, remaining: int) -> pd.DataFrame:
    leader = table["Points"].max()
    maximum = table["Points"] + remaining
    return table.assign(Maximum=maximum, CanWin=maximum >= leader)


def title_outlook(
    tables: dict[int, pd.DataFrame],
    schedule: pd.DataFrame,
    year: int,
    after_round: int,
    key: str,
    cars: int = 1,
) -> pd.DataFrame:
    """Standings after a round, with what each entrant needs to win the title.

    `tables` holds the standings after every round up to `after_round`, best first.
    """
    remaining = remaining_points(schedule, year, after_round, cars)
    outlook = title_contenders(tables[after_round], remaining).reset_index(drop=True)
    leader = outlook["Points"].iloc[0]
    outlook["Gap"] = leader - outlook["Points"]
    finished = schedule["Round"].max() <= after_round
    contenders = int(outlook["CanWin"].sum())

    status = ["In the fight" if can_win else "Out" for can_win in outlook["CanWin"]]
    status[0] = "Champion" if finished or contenders == 1 else "Leader"
    outlook["Status"] = ["" if finished and index else label for index, label in enumerate(status)]
    # Points are whole numbers from here on, so finishing ahead takes one more than the gap.
    outlook["Needed"] = (outlook["Gap"] // 1 + 1).where(outlook["Status"] == "In the fight")

    chasing = runner_up_points(schedule, year, after_round, cars)
    control, blockers = [], []
    for index, points in enumerate(outlook["Points"]):
        rivals = (outlook["Points"] + chasing).drop(index)
        best = rivals.idxmax() if len(rivals) else None
        blocked = best is not None and rivals[best] >= points + remaining
        control.append(not blocked)
        blockers.append(outlook.loc[best, key] if blocked else None)
    outlook["InControl"] = control
    outlook["Blocker"] = blockers
    outlook["OutSince"] = outlook[key].map(_eliminated(tables, schedule, year, key, cars))
    outlook["Clinch"] = None
    clinched = _clinched(tables, schedule, year, outlook[key].iloc[0], key, cars)
    outlook.loc[0, "Clinch"] = clinched or _earliest_clinch(
        outlook, schedule, year, after_round, cars
    )
    return outlook


def _eliminated(
    tables: dict[int, pd.DataFrame], schedule: pd.DataFrame, year: int, key: str, cars: int
) -> dict[str, int]:
    out: dict[str, int] = {}
    for round_number, table in sorted(tables.items()):
        leader = table["Points"].max()
        remaining = remaining_points(schedule, year, round_number, cars)
        for name, points in zip(table[key], table["Points"], strict=True):
            if name not in out and points + remaining < leader:
                out[name] = round_number
    return out


def _clinched(
    tables: dict[int, pd.DataFrame],
    schedule: pd.DataFrame,
    year: int,
    champion: str,
    key: str,
    cars: int,
) -> int | None:
    for round_number, table in sorted(tables.items()):
        remaining = remaining_points(schedule, year, round_number, cars)
        rivals = table[table[key] != champion]["Points"]
        points = table.loc[table[key] == champion, "Points"]
        if not points.empty and (rivals.empty or rivals.max() + remaining < points.iloc[0]):
            return round_number
    return None


# The soonest the leader could wrap it up: winning everything while the rest score nothing.
def _earliest_clinch(
    outlook: pd.DataFrame, schedule: pd.DataFrame, year: int, after_round: int, cars: int
) -> int | None:
    lead = outlook["Points"].iloc[0]
    rival = outlook["Points"].iloc[1:].max() if len(outlook) > 1 else 0
    gained = 0
    upcoming = schedule[schedule["Round"] > after_round].sort_values("Round")
    for round_number, sprint in zip(upcoming["Round"], upcoming["Sprint"], strict=True):
        gained += max_round_points(year, bool(sprint), cars)
        if lead + gained > rival + remaining_points(schedule, year, round_number, cars):
            return int(round_number)
    return None


def points_by_round(results: pd.DataFrame, by: str = "Driver") -> pd.DataFrame:
    return results.pivot_table(index=by, columns="Round", values="Points", aggfunc="sum")


def _short_name(event: str) -> str:
    return event.removesuffix(" Grand Prix")


def points_heatmap(
    results: pd.DataFrame,
    schedule: pd.DataFrame,
    order: list[str],
    upto_round: int,
    by: str = "Driver",
) -> go.Figure:
    played = results[results["Round"] <= upto_round]
    points = points_by_round(played, by).reindex(index=order)
    rounds = list(points.columns)
    names = schedule.set_index("Round")["EventName"]
    labels = [_short_name(str(names.get(r, r))) for r in rounds]
    finishes = _finish_texts(played, by)
    detail = [[finishes.get((name, round_number), "") for round_number in rounds] for name in order]

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


def _finish_texts(played: pd.DataFrame, by: str) -> dict[tuple[str, int], str]:
    texts: dict[tuple[str, int], str] = {}
    for session in ("Race", "Sprint"):
        rows = played[played["Session"] == session].sort_values("Position")
        for (name, round_number), positions in rows.groupby([by, "Round"])["Position"]:
            places = f"{session} " + ", ".join(f"P{int(position)}" for position in positions)
            key = (name, int(round_number))
            texts[key] = f"{texts[key]} · {places}" if key in texts else places
    return texts


def progression_figure(
    results: pd.DataFrame,
    schedule: pd.DataFrame,
    drivers: list[str],
    styles: Styles,
    upto_round: int,
) -> go.Figure:
    played = results[results["Round"] <= upto_round]
    cumulative = points_by_round(played).fillna(0).cumsum(axis=1)
    return _progression(
        cumulative, schedule, {driver: styles.get(driver, {}) for driver in drivers}
    )


def team_progression_figure(
    official: pd.DataFrame,
    schedule: pd.DataFrame,
    teams: list[str],
    colors: dict[str, str],
    upto_round: int,
) -> go.Figure:
    played = official[official["Round"] <= upto_round]
    cumulative = (
        played.pivot_table(index="Team", columns="Round", values="Points").ffill(axis=1).fillna(0)
    )
    lines = {team: {"color": colors.get(team, FALLBACK_COLOR)} for team in teams}
    return _progression(cumulative, schedule, lines)


def _progression(
    cumulative: pd.DataFrame, schedule: pd.DataFrame, lines: dict[str, dict[str, str]]
) -> go.Figure:
    names = schedule.set_index("Round")["EventName"]
    labels = [_short_name(str(names.get(r, r))) for r in cumulative.columns]
    figure = go.Figure()
    for name, style in lines.items():
        if name not in cumulative.index:
            continue
        figure.add_trace(
            go.Scatter(
                x=labels,
                y=cumulative.loc[name].to_numpy(),
                name=name,
                mode="lines+markers",
                marker={"size": 6, "line": {"color": OUTLINE, "width": 1}},
                line={
                    "color": style.get("color", FALLBACK_COLOR),
                    "dash": PLOTLY_DASHES.get(style.get("linestyle", "solid"), "solid"),
                },
                hovertemplate=f"{name} · %{{x}}<br>%{{y}} points<extra></extra>",
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
