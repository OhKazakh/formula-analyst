import pandas as pd
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


def running_order(race: races.Race, summary: analysis.RaceSummary) -> list[str]:
    st.subheader("How did the running order change?")
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
    return highlight


def winning_margin(gaps: pd.DataFrame, order: list[str]) -> str:
    final = gaps[gaps["LapNumber"] == gaps["LapNumber"].max()].set_index("Driver")["Gap"]
    if len(order) < 2 or order[1] not in final.index:
        return ""
    return f"**{order[0]}** won by {final[order[1]]:.1f} s from {order[1]}."


def gaps_section(race: races.Race, highlight: list[str]) -> None:
    st.subheader("How far behind the leader was everyone?")
    gaps = analysis.gap_to_leader(race.laps)
    if text := winning_margin(gaps, race.order):
        st.markdown(text)
    neutralised = analysis.neutralised_laps(race.laps)
    show(analysis.gap_figure(gaps, race.drivers, race.driver_styles, highlight, neutralised))
    caption = "The gap to the leader each time a driver crossed the timing line."
    if any(neutralised.values()):
        caption += " Shaded laps ran behind the safety car (SC) or the virtual safety car (VSC)."
    st.caption(caption)


def overview_tab(race: races.Race, summary: analysis.RaceSummary) -> None:
    if "Time" not in race.laps.columns:
        st.info("Lap timing isn't available for this race.")
        return
    highlight = running_order(race, summary)
    st.divider()
    gaps_section(race, highlight)
