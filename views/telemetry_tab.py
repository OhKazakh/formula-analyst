import pandas as pd
import streamlit as st

from src import analysis, races, telemetry
from views import tyres
from views.data import show
from views.state import race_key

FASTER_WHERE = "Who's faster where"


def lap_summary(race: races.Race, timed: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for driver, lap in timed.items():
        compound = lap["Compound"] if pd.notna(lap["Compound"]) else analysis.UNKNOWN_COMPOUND
        rows.append(
            {
                "Driver": driver,
                "Lap": int(lap["LapNumber"]),
                "Time": analysis.format_lap_time(lap["LapTime"]),
                "Tyre": tyres.badge(compound, race.compound_colors),
                "Compound": compound,
                "Tyre age": int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
            }
        )
    return pd.DataFrame(rows)


def corners_section(race: races.Race, traces: dict[str, pd.DataFrame]) -> None:
    if race.corners.empty or len(traces) != 2:
        return
    st.subheader("Corner by corner")
    table = telemetry.corner_comparison(traces, race.corners)
    config = {"Difference": st.column_config.NumberColumn(format="%+.0f km/h")}
    for driver in traces:
        config[f"{driver} min"] = st.column_config.NumberColumn(format="%.0f km/h")
        braking = table[f"{driver} brakes"]
        table[f"{driver} brakes"] = braking.map(
            lambda metres: "–" if pd.isna(metres) else f"{metres:.0f} m"
        )
    st.dataframe(table, hide_index=True, width="stretch", column_config=config)
    st.caption(
        "Minimum speed through each corner, and where braking started in metres before the "
        "corner marker. A dash means the driver didn't brake for it."
    )


def track_map_section(
    race: races.Race, traces: dict[str, pd.DataFrame], first: str, second: str
) -> None:
    if race.track.empty:
        return
    st.subheader("Where on the lap?")
    options = [f"{driver} speed" for driver in traces]
    if "Gear" in telemetry.channels(traces):
        options += [f"{driver} gears" for driver in traces]
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
    elif view.endswith(" gears"):
        driver = view.removesuffix(" gears")
        show(telemetry.gear_map_figure(race.track, race.map_corners, traces[driver]))
    else:
        driver = view.removesuffix(" speed")
        show(analysis.speed_map_figure(race.track, race.map_corners, traces[driver]))


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

    summary = lap_summary(race, timed)
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={"Tyre": st.column_config.ImageColumn(" ", width="small")},
    )
    if summary["Lap"].nunique() > 1:
        st.caption(
            "These laps were set at different points in the race, so fuel load and tyre age differ."
        )

    traces = {driver: race.trace(driver).reset_index(drop=True) for driver in timed}
    traces = {driver: trace for driver, trace in traces.items() if not trace.empty}
    if not traces:
        st.warning("Telemetry isn't available for these laps.")
        return
    lap_times = {driver: lap["LapTime"].total_seconds() for driver, lap in timed.items()}
    if len(traces) == 2:
        reference, other = traces
        difference = lap_times[other] - lap_times[reference]
        quicker = other if difference < 0 else reference
        st.markdown(f"**{quicker}**'s lap was {abs(difference):.3f} s quicker.")
    show(telemetry.telemetry_figure(traces, race.driver_styles, race.corners, lap_times))
    if len(traces) == 2:
        st.caption(
            f"The gap line shows how far {drivers[1]} was behind {drivers[0]} (above zero) or "
            "ahead (below zero) at each point of the lap. Car data is sampled about four times "
            "a second, so short brake taps can be missed."
        )

    corners_section(race, traces)
    track_map_section(race, traces, first, second)
