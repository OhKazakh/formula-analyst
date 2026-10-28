import base64

import pandas as pd
import streamlit as st

from src import championship, races, seasons
from views import data, logos, theme
from views.links import requested, requested_int
from views.state import favourite, pick_favourite, with_favourite

PROGRESSION_DRIVERS = 6
VIEWS = ["Drivers", "Teams"]
# The timing screen's colours: purple for the best, green for still in it.
PILLS = {
    "Champion": ("#5300A6", "#FFFFFF"),
    "Leader": ("#E10600", "#FFFFFF"),
    "In the fight": ("#28973E", "#15151E"),
    "Out": ("#67676D", "#FFFFFF"),
}
PILL_WIDTH = 96


def pill(status: str) -> str | None:
    if status not in PILLS:
        return None
    fill, ink = PILLS[status]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PILL_WIDTH}" height="22" '
        f'viewBox="0 0 {PILL_WIDTH} 22"><rect width="{PILL_WIDTH}" height="22" rx="11" '
        f'fill="{fill}"/><text x="{PILL_WIDTH / 2:g}" y="15" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="700" '
        f'fill="{ink}">{status}</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def short_event(season: seasons.Season, round_number: int) -> str:
    return season.event_name(int(round_number)).removesuffix(" Grand Prix") + " GP"


def outlook_line(row, outlook: pd.DataFrame, season: seasons.Season, key: str) -> str:
    remaining = int(outlook["Maximum"].iloc[0] - outlook["Points"].iloc[0])
    leader = outlook[key].iloc[0]
    if row.Status == "Champion":
        return f"Clinched at the {short_event(season, row.Clinch)}" if pd.notna(row.Clinch) else ""
    if row.Status == "Leader":
        lead = row.Points - outlook["Points"].iloc[1]
        when = f" · could clinch at the {short_event(season, row.Clinch)}"
        return f"Leads by {lead:g}" + (when if pd.notna(row.Clinch) else "")
    if row.Status == "In the fight":
        hands = "in their own hands" if row.InControl else "needs help"
        return f"Must outscore {leader} by {row.Needed:g}+ · {hands}"
    if row.Status == "Out":
        since = (
            f"Out since the {short_event(season, row.OutSince)}: " if pd.notna(row.OutSince) else ""
        )
        return f"{since}{row.Gap:g} behind, {remaining} left"
    return f"{row.Gap:g} points behind" if row.Gap else ""


def title_paths(
    outlook: pd.DataFrame, season: seasons.Season, after: int, teams: bool = False
) -> list[str]:
    if outlook["Status"].iloc[0] != "Leader":
        return []
    name, key = ("Team", "Team") if teams else ("Name", "Driver")
    # Teams take plural verbs, as in "Mercedes lead".
    leads, is_ = ("lead", "are") if teams else ("leads", "is")
    rounds = int((season.schedule["Round"] > after).sum())
    remaining = int(outlook["Maximum"].iloc[0] - outlook["Points"].iloc[0])
    sweep = "a one-two in every remaining race" if teams else "winning every remaining race"
    behind = "taking third and fourth" if teams else "finishing second"
    even = (
        "even with a one-two in every remaining race"
        if teams
        else "even if they won every remaining race"
    )
    leader = outlook.iloc[0]
    lead = leader["Points"] - outlook["Points"].iloc[1]
    paths = [
        f"**{leader[name]}** {leads} by {lead:g} with {remaining} points left. The title is "
        "settled once no one can catch that lead"
        + (
            f", at the {short_event(season, leader['Clinch'])} at the earliest."
            if pd.notna(leader["Clinch"])
            else "."
        )
    ]
    fighting = outlook[outlook["Status"] == "In the fight"]
    for row in fighting[fighting["InControl"]].itertuples():
        paths.append(
            f"**{getattr(row, name)}** {is_} {row.Gap:g} behind and must outscore "
            f"{leader[name]} by {row.Needed:g} or more over the last {rounds} rounds. "
            f"{sweep.capitalize()} would be enough whatever the others do."
        )
    helped = fighting[~fighting["InControl"]]
    for blocker, group in helped.groupby("Blocker", sort=False):
        rival = outlook.loc[outlook[key] == blocker, name]
        rival = rival.iloc[0] if not rival.empty else blocker
        names = [f"**{getattr(row, name)}** ({row.Gap:g} behind)" for row in group.itertuples()]
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        paths.append(
            f"{who} can only win if {rival} {'drop' if teams else 'drops'} points: {even}, "
            f"{rival} would still be ahead by {behind} every time."
        )
    out = int((outlook["Status"] == "Out").sum())
    if out:
        everyone = "Every other team is" if teams else "Everyone else is"
        paths.append(f"{everyone} out: even {sweep} would leave them behind {leader[name]}.")
    return paths


def title_race(
    contenders: pd.DataFrame, remaining: int, finished: bool, teams: bool = False
) -> str:
    leader = contenders.iloc[0]
    name = leader["Team"] if teams else leader["Name"]
    leads, has = ("lead", "have") if teams else ("leads", "has")
    title = "the teams' championship" if teams else "the championship"
    if finished:
        return f"**{name}** won {title} with {leader['Points']:g} points."
    if remaining == 0:
        return f"**{name}** {leads} with {leader['Points']:g} points."
    challengers = int(contenders["CanWin"].sum()) - 1
    if challengers == 0:
        return (
            f"**{name}** {has} clinched the title: nobody can catch "
            f"{leader['Points']:g} points with {remaining} still available."
        )
    rivals = ("team" if teams else "driver") + ("" if challengers == 1 else "s")
    return (
        f"**{name}** {leads} with {leader['Points']:g} points. With {remaining} points "
        f"still available, {challengers} other {rivals} can still take the title."
    )


def outlook_table(
    outlook: pd.DataFrame, colors: dict[str, str], season: seasons.Season, key: str
) -> pd.DataFrame:
    columns = {
        "Pos": outlook["Position"],
        "Badge": [logos.badge(team, colors.get(team), season.year) for team in outlook["Team"]],
    }
    if key == "Driver":
        columns["Driver"] = outlook["Name"]
    columns |= {
        "Team": outlook["Team"],
        "Points": outlook["Points"],
        "Gap": ["–" if not gap else f"{gap:g}" for gap in outlook["Gap"]],
        "Wins": outlook["Wins"],
        "Title": [pill(status) for status in outlook["Status"]],
        "What it takes": [outlook_line(row, outlook, season, key) for row in outlook.itertuples()],
    }
    return pd.DataFrame(columns)


# Pixel widths keep the whole table, explanation included, inside a laptop-wide page.
def table_config() -> dict:
    return {
        "Pos": st.column_config.NumberColumn(width=44),
        "Badge": st.column_config.ImageColumn(" ", width=44),
        "Driver": st.column_config.TextColumn(width=160),
        "Team": st.column_config.TextColumn(width=120),
        "Points": st.column_config.Column(width=60),
        "Gap": st.column_config.Column(width=55),
        "Wins": st.column_config.Column(width=50),
        "Title": st.column_config.ImageColumn("Title", width=PILL_WIDTH + 8),
        "What it takes": st.column_config.TextColumn(width=360),
    }


def show_outlook(
    outlook: pd.DataFrame,
    colors: dict[str, str],
    season: seasons.Season,
    after: int,
    key: str,
    marked: pd.Series,
) -> None:
    table = outlook_table(outlook, colors, season, key)
    st.dataframe(
        # A styled table shows six decimals unless told otherwise; half points stay visible.
        theme.favourite_rows(table, marked).format({"Points": "{:g}"}),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config=table_config(),
    )
    paths = title_paths(outlook, season, after, teams=key == "Team")
    if paths:
        st.markdown("\n".join(f"- {path}" for path in paths))


def drivers_view(season: seasons.Season, after: int, colors: dict[str, str]) -> None:
    tables = {
        round_number: championship.standings(season.results, round_number)
        for round_number in season.completed_rounds
        if round_number <= after
    }
    outlook = championship.title_outlook(tables, season.schedule, season.year, after, "Driver")
    table = tables[after]
    remaining = championship.remaining_points(season.schedule, season.year, after)
    finished = after == season.schedule["Round"].max()
    st.markdown(title_race(outlook, remaining, finished))
    show_outlook(outlook, colors, season, after, "Driver", outlook["Driver"].eq(favourite()))
    st.caption(
        "Points left assume a win and, where it exists, the fastest-lap point at every remaining "
        "race, plus a win in every remaining sprint. A tie on points goes to whoever has more "
        "wins. In their own hands means winning every remaining race would be enough whatever "
        "the others do."
    )

    st.subheader("Points progression")
    leaders = with_favourite(
        table["Driver"].head(PROGRESSION_DRIVERS).tolist(), table["Driver"].tolist()
    )
    drivers = st.multiselect(
        "Drivers",
        table["Driver"].tolist(),
        default=leaders,
        key=f"progression-{season.year}-{favourite() or ''}",
    )
    st.plotly_chart(
        championship.progression_figure(
            season.results, season.schedule, drivers, season.driver_styles, after
        ),
        config={"displayModeBar": False},
    )

    st.subheader("Points per round")
    st.plotly_chart(
        championship.points_heatmap(
            season.results, season.schedule, table["Driver"].tolist(), after
        ),
        config={"displayModeBar": False},
    )
    st.caption("Race and sprint points combined. Hover a cell for the finishing positions.")


def penalties_text(penalties: dict[str, float]) -> str:
    notes = [
        f"{team}'s total includes a {abs(points):g}-point penalty"
        if points < 0
        else f"{team}'s total includes {points:g} points that weren't scored on track"
        for team, points in penalties.items()
    ]
    return " ".join(f"{note}." for note in notes)


def teams_view(season: seasons.Season, after: int, colors: dict[str, str]) -> None:
    table = championship.team_standings(season.team_standings, after)
    if table.empty:
        st.info("Team standings aren't available for this season.")
        return
    remaining = championship.remaining_points(season.schedule, season.year, after, cars=2)
    tables = {
        round_number: championship.team_standings(season.team_standings, round_number)
        for round_number in season.completed_rounds
        if round_number <= after
    }
    outlook = championship.title_outlook(
        tables, season.schedule, season.year, after, "Team", cars=2
    )
    finished = after == season.schedule["Round"].max()
    st.markdown(title_race(outlook, remaining, finished, teams=True))
    show_outlook(outlook, colors, season, after, "Team", pd.Series(False, index=outlook.index))
    caption = (
        "The official standings. Points left assume a one-two and, where it exists, the "
        "fastest-lap point at every remaining race, plus a one-two in every remaining sprint."
    )
    penalties = championship.penalties(season.team_standings, season.results, after)
    if penalties:
        caption += " " + penalties_text(penalties)
    st.caption(caption)

    st.subheader("Points progression")
    teams = table["Team"].tolist()
    chosen = st.multiselect("Teams", teams, default=teams, key=f"team_progression-{season.year}")
    st.plotly_chart(
        championship.team_progression_figure(
            season.team_standings, season.schedule, chosen, colors, after
        ),
        config={"displayModeBar": False},
    )

    st.subheader("Points per round")
    st.plotly_chart(
        championship.points_heatmap(season.results, season.schedule, teams, after, by="Team"),
        config={"displayModeBar": False},
    )
    st.caption(
        "Points scored by both cars, race and sprint combined, so penalties applied to the "
        "standings don't show here. Hover a cell for the finishing positions."
    )


def requested_view() -> str | None:
    return {label.lower(): label for label in VIEWS}.get(requested("view"))


def render() -> None:
    bundle_version = races.bundle_version()
    years = sorted(data.bundled_seasons(bundle_version), reverse=True)
    if not years:
        st.info("No championship data is bundled.")
        return
    requested_year = requested_int("season")
    year = st.sidebar.selectbox(
        "Season",
        years,
        index=years.index(requested_year) if requested_year in years else 0,
        key="championship_season",
    )
    season: seasons.Season = data.season(year, bundle_version)
    rounds = season.completed_rounds
    st.title(f"{year} Championship")
    if not rounds:
        st.info("No races have been run this season yet.")
        return

    requested_round = requested_int("round") if requested_year == year else None
    after = (
        st.select_slider(
            "Standings after",
            options=rounds,
            value=requested_round if requested_round in rounds else rounds[-1],
            format_func=lambda r: f"Round {r} · {season.event_name(r)}",
            key=f"standings_after-{year}",
        )
        if len(rounds) > 1
        else rounds[0]
    )
    drivers = championship.standings(season.results, after)["Driver"].tolist()
    liked = pick_favourite(drivers)
    colors = championship.team_colors(season.results, season.driver_styles)

    drivers_tab, teams_tab = st.tabs(
        VIEWS, default=requested_view(), key="championship_view", on_change="rerun"
    )
    params = {
        "season": str(year),
        "round": str(after),
        "view": st.session_state.get("championship_view", VIEWS[0]).lower(),
    }
    st.query_params.from_dict(params | ({"driver": liked} if liked else {}))
    if drivers_tab.open:
        with drivers_tab:
            drivers_view(season, after, colors)
    if teams_tab.open:
        with teams_tab:
            teams_view(season, after, colors)
