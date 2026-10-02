from datetime import date

import pandas as pd
import streamlit as st
from fastf1.core import Session
from matplotlib.figure import Figure

from src import analysis

FIRST_SEASON = 2018
DEFAULT_YEAR = 2024
DEFAULT_EVENT = "Italian Grand Prix"

st.set_page_config(page_title="Formula Analyst", page_icon="🏁", layout="wide")
analysis.enable_cache()


@st.cache_data(ttl="1d", show_spinner=False)
def calendar(year: int) -> pd.DataFrame:
    return analysis.race_calendar(year)


@st.cache_resource(show_spinner=False, max_entries=2)
def race(year: int, event: str) -> Session:
    return analysis.load_race(year, event)


def show(fig: Figure) -> None:
    st.pyplot(fig)


def sidebar() -> tuple[int, str]:
    st.sidebar.header("Race")
    years = list(range(date.today().year, FIRST_SEASON - 1, -1))
    year = st.sidebar.selectbox("Season", years, index=years.index(DEFAULT_YEAR))
    try:
        events = calendar(year)["EventName"].tolist()
    except Exception as exc:
        st.sidebar.error(f"Couldn't load the {year} calendar: {exc}")
        st.stop()
    if not events:
        st.sidebar.warning("No completed races for this season yet.")
        st.stop()
    default = events.index(DEFAULT_EVENT) if DEFAULT_EVENT in events else len(events) - 1
    event = st.sidebar.selectbox("Grand Prix", events, index=default)
    st.sidebar.caption(
        "Timing data via [FastF1](https://github.com/theOehrly/Fast-F1). "
        "Unofficial and non-commercial."
    )
    return year, event


def pace_tab(session: Session, laps: pd.DataFrame, clean: pd.DataFrame, order: list[str]) -> None:
    st.subheader("How did each driver's pace change over the race?")
    drivers = st.multiselect("Drivers", order, default=order[:3])
    left, right, _ = st.columns([1, 1, 2])
    representative = left.toggle(
        "Representative laps only",
        value=True,
        help="Excludes pit in/out laps, laps under safety car or flags, "
        "and laps FastF1 marks as inaccurate.",
    )
    fuel = right.toggle(
        "Fuel-corrected",
        value=False,
        help="Normalises every lap to an empty tank so tyre wear isn't hidden "
        "by the car getting lighter.",
    )

    if not drivers:
        st.info("Pick at least one driver.")
        return

    data = (clean if representative else laps).copy()
    column = "LapTimeSeconds"
    if fuel:
        data["FuelCorrected"] = analysis.fuel_corrected(data, session.total_laps)
        column = "FuelCorrected"
    show(analysis.pace_figure(data, drivers, session, column))


def strategy_tab(session: Session, laps: pd.DataFrame, order: list[str]) -> None:
    st.subheader("What tyre strategy did everyone run?")
    stints = analysis.stint_summary(laps)
    show(analysis.strategy_figure(stints, order, session))
    with st.expander("Stint table"):
        st.dataframe(stints, hide_index=True, width="stretch")


def fastest_lap_tab(session: Session, laps: pd.DataFrame, order: list[str]) -> None:
    st.subheader("How do two drivers compare on their fastest lap?")
    left, right = st.columns(2)
    first = left.selectbox("Driver A", order, index=0)
    second = right.selectbox("Driver B", order, index=min(1, len(order) - 1))

    candidates = {
        driver: analysis.fastest_lap(laps, driver) for driver in dict.fromkeys([first, second])
    }
    missing = [driver for driver, lap in candidates.items() if lap is None]
    if missing:
        st.warning(f"No timed lap for {', '.join(missing)}.")
    selected = {driver: lap for driver, lap in candidates.items() if lap is not None}
    if not selected:
        return

    summary = pd.DataFrame(
        [
            {
                "Driver": driver,
                "Lap": int(lap["LapNumber"]),
                "Time": analysis.format_lap_time(lap["LapTime"]),
                "Compound": lap["Compound"],
                "Tyre age": int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
            }
            for driver, lap in selected.items()
        ]
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    if summary["Lap"].nunique() > 1:
        st.caption(
            "These laps were set at different points in the race, so fuel load and tyre age differ."
        )

    try:
        fig = analysis.speed_trace_figure(selected, session)
    except Exception as exc:
        st.warning(f"Telemetry isn't available for these laps: {exc}")
        return
    show(fig)


def degradation_tab(session: Session, clean: pd.DataFrame) -> None:
    st.subheader("How fast did the tyres degrade?")
    left, right = st.columns(2)
    min_laps = left.slider("Minimum laps per stint", 3, 20, 8)
    fuel_effect = right.slider(
        "Fuel effect (s per lap of fuel)", 0.0, 0.1, analysis.DEFAULT_FUEL_EFFECT, 0.005
    )

    data = clean.copy()
    data["FuelCorrected"] = analysis.fuel_corrected(data, session.total_laps, fuel_effect)
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
    year, event = sidebar()
    st.title(f"{year} {event}")

    with st.spinner("Loading timing data. The first load of a race takes about a minute."):
        try:
            session = race(year, event)
        except Exception as exc:
            st.error(f"Couldn't load this race: {exc}")
            st.stop()

    laps = analysis.with_seconds(session.laps)
    clean = analysis.representative_laps(laps)
    order = analysis.finishing_order(session)

    pace, strategy, fastest, deg = st.tabs(
        ["Race pace", "Tyre strategy", "Fastest lap", "Degradation"]
    )
    with pace:
        pace_tab(session, laps, clean, order)
    with strategy:
        strategy_tab(session, laps, order)
    with fastest:
        fastest_lap_tab(session, laps, order)
    with deg:
        degradation_tab(session, clean)


main()
