import pandas as pd
import streamlit as st

from src import analysis, races
from views.data import show
from views.state import race_key


def driver_pace(race: races.Race, clean: pd.DataFrame) -> None:
    st.subheader("How did each driver's pace change over the race?")
    drivers = st.multiselect(
        "Drivers", race.drivers, default=race.drivers[:3], key=race_key(race, "pace_drivers")
    )
    with st.container(horizontal=True):
        representative = st.toggle(
            "Representative laps only",
            value=True,
            key=race_key(race, "representative"),
            help="Excludes pit in/out laps, laps under safety car or flags, "
            "and laps FastF1 marks as inaccurate.",
        )
        fuel = st.toggle(
            "Fuel-corrected",
            value=False,
            key=race_key(race, "fuel_corrected"),
            help="Normalises every lap to an empty tank so tyre wear isn't hidden "
            "by the car getting lighter.",
        )

    if not drivers:
        st.info("Pick at least one driver.")
        return

    source = clean if representative else race.laps
    laps = source[source["Driver"].isin(drivers)].copy()
    if laps.empty:
        st.info(
            "None of these drivers have representative laps in this race. "
            "Turn off “Representative laps only” to see every lap."
            if representative
            else "No laps to show for these drivers."
        )
        return

    column = "LapTimeSeconds"
    if fuel:
        laps["FuelCorrected"] = analysis.fuel_corrected(laps, race.total_laps)
        column = "FuelCorrected"
    show(analysis.pace_figure(laps, drivers, race.driver_styles, column))


def team_pace(race: races.Race) -> None:
    st.subheader("Which teams were quickest?")
    pace = analysis.team_pace(race.laps)
    if pace.empty:
        st.info("This race has no representative laps to compare.")
        return
    if len(pace) > 1:
        st.markdown(
            f"**{pace['Team'].iloc[0]}** were quickest, {pace['Gap'].iloc[1]:.3f} s a lap "
            f"faster than {pace['Team'].iloc[1]}."
        )
    chart, table = st.columns([2, 1])
    with chart:
        show(analysis.team_pace_figure(race.laps, race.driver_styles))
    table.dataframe(
        pd.DataFrame(
            {
                "Team": pace["Team"],
                "Median lap": pace["Median"].map(analysis.format_seconds),
                "Gap": pace["Gap"].map(lambda gap: f"+{gap:.3f}s" if gap else "–"),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Racing laps within 107% of the fastest, without pit laps, safety car periods or "
        "inaccurate laps. Teams are ordered by their median lap time."
    )


def lap_distribution(race: races.Race) -> None:
    st.subheader("How consistent was each driver?")
    drivers = st.multiselect(
        "Drivers",
        race.drivers,
        default=race.drivers[:10],
        key=race_key(race, "distribution_drivers"),
    )
    if not drivers:
        st.info("Pick at least one driver.")
        return
    show(
        analysis.lap_distribution_figure(
            race.laps, drivers, race.driver_styles, race.compound_colors
        )
    )
    st.caption(
        "Each dot is one lap, coloured by tyre compound; the shape shows how lap times spread."
    )


def pace_tab(race: races.Race, clean: pd.DataFrame) -> None:
    driver_pace(race, clean)
    st.divider()
    team_pace(race)
    st.divider()
    lap_distribution(race)
