import pandas as pd
import streamlit as st

from src import analysis, races
from views.data import show
from views.state import race_key

FASTER_WHERE = "Who's faster where"


def telemetry_tab(race: races.Race) -> None:
    st.subheader("How do two drivers compare on their fastest lap?")
    left, right = st.columns(2)
    first = left.selectbox("Driver A", race.drivers, index=0, key=race_key(race, "driver_a"))
    second = right.selectbox(
        "Driver B",
        race.drivers,
        index=min(1, len(race.drivers) - 1),
        key=race_key(race, "driver_b"),
    )

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
    if view == FASTER_WHERE:
        show(
            analysis.dominance_map_figure(race.track, race.map_corners, traces, race.driver_styles)
        )
        st.caption(
            f"The lap is split into {analysis.MINI_SECTORS} equal mini-sectors, each coloured "
            "by the driver who was quicker through it."
        )
    else:
        driver = view.removesuffix(" speed")
        show(analysis.speed_map_figure(race.track, race.map_corners, traces[driver]))
