from datetime import date

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src import analysis, races

FIRST_SEASON = 2018
DEFAULT_YEAR = 2024
DEFAULT_EVENT = "Italian Grand Prix"

st.set_page_config(page_title="Formula Analyst", page_icon="🏁", layout="wide")
races.enable_cache()


@st.cache_data(ttl="1h", show_spinner=False)
def live_timing() -> bool:
    return races.live_timing_available()


@st.cache_data(show_spinner=False)
def bundled_races(bundle_version: str) -> pd.DataFrame:
    return races.saved_races()


@st.cache_data(ttl="1d", show_spinner=False)
def calendar(year: int) -> pd.DataFrame:
    return races.race_calendar(year)


@st.cache_data(show_spinner=False, max_entries=16)
def load(year: int, event: str, bundle_version: str) -> races.Race:
    return races.load_race(year, event)


def show(fig: Figure) -> None:
    st.pyplot(fig)


def season_events(year: int, bundled: pd.DataFrame, online: bool) -> list[str]:
    offline_events = bundled.loc[bundled["Year"] == year, "EventName"].tolist()
    if not online:
        return offline_events
    try:
        return calendar(year)["EventName"].tolist()
    except Exception:
        return offline_events


def sidebar(bundle_version: str) -> tuple[int, str]:
    st.sidebar.header("Race")
    bundled = bundled_races(bundle_version)
    online = live_timing()

    seasons = set(bundled["Year"])
    if online:
        seasons |= set(range(FIRST_SEASON, date.today().year + 1))
    if not seasons:
        st.sidebar.error("No race data is available.")
        st.stop()
    years = sorted(seasons, reverse=True)
    default_year = years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years else 0
    year = st.sidebar.selectbox("Season", years, index=default_year)

    events = season_events(year, bundled, online)
    if not events:
        st.sidebar.warning("No completed races for this season yet.")
        st.stop()
    default = events.index(DEFAULT_EVENT) if DEFAULT_EVENT in events else len(events) - 1
    event = st.sidebar.selectbox("Grand Prix", events, index=default)

    if not online:
        st.sidebar.info("Live timing data is unavailable here, so only bundled races are listed.")
    st.sidebar.caption(
        "Timing data via [FastF1](https://github.com/theOehrly/Fast-F1). "
        "Unofficial and non-commercial."
    )
    return year, event


def pace_tab(race: races.Race, clean: pd.DataFrame) -> None:
    st.subheader("How did each driver's pace change over the race?")
    drivers = st.multiselect("Drivers", race.drivers, default=race.drivers[:3])
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
    data = source[source["Driver"].isin(drivers)].copy()
    if data.empty:
        st.info(
            "None of these drivers have representative laps in this race. "
            "Turn off “Representative laps only” to see every lap."
            if representative
            else "No laps to show for these drivers."
        )
        return

    column = "LapTimeSeconds"
    if fuel:
        data["FuelCorrected"] = analysis.fuel_corrected(data, race.total_laps)
        column = "FuelCorrected"
    show(analysis.pace_figure(data, drivers, race.driver_styles, column))


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


def degradation_tab(race: races.Race, clean: pd.DataFrame) -> None:
    st.subheader("How fast did the tyres degrade?")
    left, right = st.columns(2)
    min_laps = left.slider("Minimum laps per stint", 3, 20, 8)
    fuel_effect = right.slider(
        "Fuel effect (s per lap of fuel)", 0.0, 0.1, analysis.DEFAULT_FUEL_EFFECT, 0.005
    )

    data = clean.copy()
    data["FuelCorrected"] = analysis.fuel_corrected(data, race.total_laps, fuel_effect)
    deg = analysis.degradation(data, "FuelCorrected", min_laps)
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
    stint_laps = data[(data["Driver"] == driver) & (data["Stint"] == stint)]
    show(analysis.stint_fit_figure(stint_laps, "FuelCorrected"))


def main() -> None:
    bundle_version = races.bundle_version()
    year, event = sidebar(bundle_version)
    st.title(f"{year} {event}")

    with st.spinner("Loading timing data. The first load of a race takes about a minute."):
        try:
            race = load(year, event, bundle_version)
        except races.RaceDataUnavailable as exc:
            st.error(str(exc))
            st.button("Try again")
            st.stop()
        except Exception as exc:
            st.error(f"Couldn't load this race: {exc}")
            st.button("Try again")
            st.stop()

    clean = analysis.representative_laps(race.laps)
    pace, strategy, fastest, deg = st.tabs(
        ["Race pace", "Tyre strategy", "Fastest lap", "Degradation"]
    )
    with pace:
        pace_tab(race, clean)
    with strategy:
        strategy_tab(race)
    with fastest:
        fastest_lap_tab(race)
    with deg:
        degradation_tab(race, clean)


main()
