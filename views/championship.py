import pandas as pd
import streamlit as st

from src import championship, races, seasons
from views import data
from views.links import requested_int

PROGRESSION_DRIVERS = 6


def title_race(contenders: pd.DataFrame, remaining: int, finished: bool) -> str:
    leader = contenders.iloc[0]
    if finished:
        return f"**{leader['Name']}** won the championship with {leader['Points']:g} points."
    if remaining == 0:
        return f"**{leader['Name']}** leads with {leader['Points']:g} points."
    challengers = int(contenders["CanWin"].sum()) - 1
    if challengers == 0:
        return (
            f"**{leader['Name']}** has clinched the title: nobody can catch "
            f"{leader['Points']:g} points with {remaining} still available."
        )
    plural = "driver" if challengers == 1 else "drivers"
    return (
        f"**{leader['Name']}** leads with {leader['Points']:g} points. With {remaining} points "
        f"still available, {challengers} other {plural} can still take the title."
    )


def standings_table(contenders: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pos": contenders["Position"],
            "Driver": contenders["Name"],
            "Team": contenders["Team"],
            "Points": contenders["Points"],
            "Wins": contenders["Wins"],
            "Max": contenders["Maximum"],
            "Can win": contenders["CanWin"].map({True: "✓", False: ""}),
        }
    )


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
    st.title(f"{year} Drivers' Championship")
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
    st.query_params.from_dict({"season": str(year), "round": str(after)})
    table = championship.standings(season.results, after)
    remaining = championship.remaining_points(season.schedule, year, after)
    contenders = championship.title_contenders(table, remaining)
    finished = after == season.schedule["Round"].max()
    st.markdown(title_race(contenders, remaining, finished))

    standings = standings_table(contenders)
    small = st.column_config.Column(width="small")
    st.dataframe(
        standings,
        hide_index=True,
        width="stretch",
        height=35 * (len(standings) + 1) + 3,
        column_config={column: small for column in ("Pos", "Points", "Wins", "Max", "Can win")},
    )
    st.caption(
        "Max is the most points a driver could still finish with: a win and, where it exists, "
        "the fastest-lap point at every remaining race, plus a win in every remaining sprint."
    )

    st.subheader("Points progression")
    leaders = table["Driver"].head(PROGRESSION_DRIVERS).tolist()
    drivers = st.multiselect(
        "Drivers", table["Driver"].tolist(), default=leaders, key=f"progression-{year}"
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
