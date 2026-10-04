import streamlit as st

from src import analysis, races
from views.data import show
from views.state import race_key
from views.text import listing


def laps_led_text(summary: analysis.RaceSummary) -> str:
    leaders = list(summary.laps_led.items())
    if not leaders:
        return ""
    first, laps = leaders[0]
    if len(leaders) == 1:
        return f"**{first}** led every lap."
    others = listing([f"{driver} {count}" for driver, count in leaders[1:5]])
    changes = "once" if summary.lead_changes == 1 else f"{summary.lead_changes} times"
    return f"**{first}** led {laps} laps, {others}. The lead changed hands {changes}."


def overview_tab(race: races.Race, summary: analysis.RaceSummary) -> None:
    st.subheader("How did the running order change?")
    if "Time" not in race.laps.columns:
        st.info("Lap timing isn't available for this race.")
        return
    if text := laps_led_text(summary):
        st.markdown(text)
    highlight = st.multiselect(
        "Highlight drivers",
        race.drivers,
        default=race.drivers[:3],
        key=race_key(race, "position_highlight"),
    )
    positions = analysis.lap_positions(race.laps)
    show(analysis.position_figure(positions, race.drivers, race.driver_styles, highlight))
    st.caption("Position at the end of each lap, from the order drivers crossed the timing line.")
