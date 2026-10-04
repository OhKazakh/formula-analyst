import pandas as pd
import streamlit as st

from src import championship, races, seasons
from views import data, logos, theme
from views.links import requested, requested_int
from views.state import favourite, pick_favourite, with_favourite

PROGRESSION_DRIVERS = 6
VIEWS = ["Drivers", "Teams"]
BADGE = st.column_config.ImageColumn(" ", width="small")
SMALL = st.column_config.Column(width="small")


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


def standings_table(contenders: pd.DataFrame, colors: dict[str, str], year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pos": contenders["Position"],
            "Badge": [logos.badge(team, colors.get(team), year) for team in contenders["Team"]],
            "Driver": contenders["Name"],
            "Team": contenders["Team"],
            "Points": contenders["Points"],
            "Wins": contenders["Wins"],
            "Max": contenders["Maximum"],
            "Can win": contenders["CanWin"].map({True: "✓", False: ""}),
        }
    )


def team_table(contenders: pd.DataFrame, colors: dict[str, str], year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pos": contenders["Position"],
            "Badge": [logos.badge(team, colors.get(team), year) for team in contenders["Team"]],
            "Team": contenders["Team"],
            "Points": contenders["Points"],
            "Wins": contenders["Wins"],
            "Max": contenders["Maximum"],
            "Can win": contenders["CanWin"].map({True: "✓", False: ""}),
        }
    )


def table_config() -> dict:
    return {
        "Badge": BADGE,
        "Driver": st.column_config.TextColumn(width=220),
        "Team": st.column_config.TextColumn(width=220),
        **{column: SMALL for column in ("Pos", "Points", "Wins", "Max", "Can win")},
    }


def drivers_view(season: seasons.Season, after: int, colors: dict[str, str]) -> None:
    table = championship.standings(season.results, after)
    remaining = championship.remaining_points(season.schedule, season.year, after)
    contenders = championship.title_contenders(table, remaining)
    finished = after == season.schedule["Round"].max()
    st.markdown(title_race(contenders, remaining, finished))

    standings = standings_table(contenders, colors, season.year)
    st.dataframe(
        theme.favourite_rows(standings, contenders["Driver"].eq(favourite())),
        hide_index=True,
        width="stretch",
        height=35 * (len(standings) + 1) + 3,
        column_config=table_config(),
    )
    st.caption(
        "Max is the most points a driver could still finish with: a win and, where it exists, "
        "the fastest-lap point at every remaining race, plus a win in every remaining sprint."
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
    contenders = championship.title_contenders(table, remaining)
    finished = after == season.schedule["Round"].max()
    st.markdown(title_race(contenders, remaining, finished, teams=True))

    standings = team_table(contenders, colors, season.year)
    st.dataframe(
        standings,
        hide_index=True,
        width="stretch",
        height=35 * (len(standings) + 1) + 3,
        column_config=table_config(),
    )
    caption = (
        "The official standings. Max assumes a one-two and, where it exists, the fastest-lap "
        "point at every remaining race, plus a one-two in every remaining sprint."
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
