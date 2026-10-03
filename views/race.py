from datetime import date

import pandas as pd
import streamlit as st

from src import analysis, races
from views import data
from views.data import show

FIRST_SEASON = 2018
DEFAULT_YEAR = 2024
DEFAULT_EVENT = "Italian Grand Prix"
FASTER_WHERE = "Who's faster where"


def race_key(race: races.Race, name: str) -> str:
    return f"{name}-{race.year}-{race.round_number}"


def season_events(year: int, bundled: pd.DataFrame, online: bool) -> list[str]:
    offline_events = bundled.loc[bundled["Year"] == year, "EventName"].tolist()
    if not online:
        return offline_events
    try:
        return data.calendar(year)["EventName"].tolist()
    except Exception:
        return offline_events


def sidebar(bundle_version: str) -> tuple[int, str]:
    bundled = data.bundled_races(bundle_version)
    online = data.live_timing()

    seasons = set(bundled["Year"])
    if online:
        seasons |= set(range(FIRST_SEASON, date.today().year + 1))
    if not seasons:
        st.sidebar.error("No race data is available.")
        st.stop()
    years = sorted(seasons, reverse=True)
    default_year = years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years else 0
    year = st.sidebar.selectbox("Season", years, index=default_year, key="race_season")

    events = season_events(year, bundled, online)
    if not events:
        st.sidebar.warning("No completed races for this season yet.")
        st.stop()
    default = events.index(DEFAULT_EVENT) if DEFAULT_EVENT in events else len(events) - 1
    event = st.sidebar.selectbox("Grand Prix", events, index=default, key="race_event")

    if not online:
        st.sidebar.info("Live timing data is unavailable here, so only bundled races are listed.")
    return year, event


def pace_tab(race: races.Race, clean: pd.DataFrame) -> None:
    st.subheader("How did each driver's pace change over the race?")
    drivers = st.multiselect(
        "Drivers", race.drivers, default=race.drivers[:3], key=race_key(race, "pace_drivers")
    )
    with st.container(horizontal=True):
        representative = st.toggle(
            "Representative laps only",
            value=True,
            help="Excludes pit in/out laps, laps under safety car or flags, "
            "and laps FastF1 marks as inaccurate.",
        )
        fuel = st.toggle(
            "Fuel-corrected",
            value=False,
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


def positions_tab(race: races.Race) -> None:
    st.subheader("How did the running order change?")
    if "Time" not in race.laps.columns:
        st.info("Lap timing isn't available for this race.")
        return
    highlight = st.multiselect(
        "Highlight drivers",
        race.drivers,
        default=race.drivers[:3],
        key=race_key(race, "position_highlight"),
    )
    positions = analysis.lap_positions(race.laps)
    show(analysis.position_figure(positions, race.drivers, race.driver_styles, highlight))
    st.caption("Position at the end of each lap, from the order drivers crossed the timing line.")


def replay_tab(race: races.Race, bundle_version: str) -> None:
    st.subheader("How did the race unfold on track?")
    if race.track.empty or not {"LapStartTime", "Time"} <= set(race.laps.columns):
        st.info("A track map isn't available for this race.")
        return
    figure = data.replay_chart(race.year, race.event, bundle_version, race)
    st.plotly_chart(figure, config={"displayModeBar": False})
    st.caption(
        "Press Play or drag the slider. Positions within a lap are estimated from lap times "
        "and the winner's fastest-lap speed profile, and the order is the order on track, "
        "before any penalties."
    )


def comparison_tab(race: races.Race) -> None:
    st.subheader("Which teams were quickest?")
    pace = analysis.team_pace(race.laps)
    if pace.empty:
        st.info("This race has no representative laps to compare.")
        return
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


def strategy_tab(race: races.Race) -> None:
    st.subheader("What tyre strategy did everyone run?")
    stints = analysis.stint_summary(race.laps)
    show(analysis.strategy_figure(stints, race.drivers, race.compound_colors))
    with st.expander("Stint table"):
        st.dataframe(stints, hide_index=True, width="stretch")


def fastest_lap_tab(race: races.Race) -> None:
    st.subheader("How do two drivers compare on their fastest lap?")
    left, right = st.columns(2)
    first = left.selectbox("Driver A", race.drivers, index=0)
    second = right.selectbox("Driver B", race.drivers, index=min(1, len(race.drivers) - 1))

    drivers = list(dict.fromkeys([first, second]))
    laps = {driver: analysis.fastest_lap(race.laps, driver) for driver in drivers}
    timed = {driver: lap for driver, lap in laps.items() if lap is not None}
    if missing := [driver for driver in drivers if driver not in timed]:
        st.warning(f"No timed lap for {', '.join(missing)}.")
    if not timed:
        return

    summary = pd.DataFrame(
        [
            {
                "Driver": driver,
                "Lap": int(lap["LapNumber"]),
                "Time": analysis.format_lap_time(lap["LapTime"]),
                "Compound": (
                    lap["Compound"] if pd.notna(lap["Compound"]) else analysis.UNKNOWN_COMPOUND
                ),
                "Tyre age": int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
            }
            for driver, lap in timed.items()
        ]
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    if summary["Lap"].nunique() > 1:
        st.caption(
            "These laps were set at different points in the race, so fuel load and tyre age differ."
        )

    traces = {driver: race.trace(driver) for driver in timed}
    traces = {driver: trace for driver, trace in traces.items() if not trace.empty}
    if not traces:
        st.warning("Telemetry isn't available for these laps.")
        return
    show(analysis.speed_trace_figure(traces, race.driver_styles, race.corners))

    if race.track.empty:
        return
    st.subheader("Where on the lap?")
    options = [f"{driver} speed" for driver in traces]
    if len(traces) == 2:
        options.append(FASTER_WHERE)
    view = st.radio(
        "Colour the track by",
        options,
        horizontal=True,
        key=race_key(race, f"track_map_view-{first}-{second}"),
    )
    chart, _ = st.columns([3, 1])
    with chart:
        if view == FASTER_WHERE:
            show(
                analysis.dominance_map_figure(
                    race.track, race.map_corners, traces, race.driver_styles
                )
            )
            st.caption(
                f"The lap is split into {analysis.MINI_SECTORS} equal mini-sectors, each coloured "
                "by the driver who was quicker through it."
            )
        else:
            driver = view.removesuffix(" speed")
            show(analysis.speed_map_figure(race.track, race.map_corners, traces[driver]))


def degradation_tab(race: races.Race, clean: pd.DataFrame) -> None:
    st.subheader("How fast did the tyres degrade?")
    left, right = st.columns(2)
    min_laps = left.slider("Minimum laps per stint", 3, 20, 8)
    fuel_effect = right.slider(
        "Fuel effect (s per lap of fuel)", 0.0, 0.1, analysis.DEFAULT_FUEL_EFFECT, 0.005
    )

    laps = clean.copy()
    laps["FuelCorrected"] = analysis.fuel_corrected(laps, race.total_laps, fuel_effect)
    deg = analysis.degradation(laps, "FuelCorrected", min_laps)
    if deg.empty:
        st.info("No stints long enough to fit.")
        return

    summary, table = st.columns([1, 2])
    summary.markdown("**By compound** (s/lap)")
    summary.dataframe(analysis.degradation_by_compound(deg), width="stretch")
    table.markdown("**By stint**")
    table.dataframe(deg.round({"DegPerLap": 3}), hide_index=True, width="stretch")

    row = st.selectbox(
        "Inspect stint",
        deg.index,
        format_func=lambda i: (
            f"{deg.at[i, 'Driver']} · stint {deg.at[i, 'Stint']} · {deg.at[i, 'Compound']}"
        ),
    )
    driver, stint = deg.at[row, "Driver"], deg.at[row, "Stint"]
    stint_laps = laps[(laps["Driver"] == driver) & (laps["Stint"] == stint)]
    show(analysis.stint_fit_figure(stint_laps, "FuelCorrected"))


def render() -> None:
    bundle_version = races.bundle_version()
    year, event = sidebar(bundle_version)
    st.title(f"{year} {event}")

    with st.spinner("Loading timing data. The first load of a race takes about a minute."):
        try:
            race = data.race(year, event, bundle_version)
        except races.RaceDataUnavailable as exc:
            st.error(str(exc))
            st.button("Try again")
            st.stop()
        except Exception as exc:
            st.error(f"Couldn't load this race: {exc}")
            st.button("Try again")
            st.stop()

    clean = analysis.representative_laps(race.laps)
    pace, positions, replay, comparison, strategy, fastest, deg = st.tabs(
        [
            "Race pace",
            "Positions",
            "Race replay",
            "Pace comparison",
            "Tyre strategy",
            "Fastest lap",
            "Degradation",
        ]
    )
    with pace:
        pace_tab(race, clean)
    with positions:
        positions_tab(race)
    with replay:
        replay_tab(race, bundle_version)
    with comparison:
        comparison_tab(race)
    with strategy:
        strategy_tab(race)
    with fastest:
        fastest_lap_tab(race)
    with deg:
        degradation_tab(race, clean)
