import pandas as pd
import streamlit as st

from src import analysis, conditions, races
from views.data import show
from views.state import favourite, race_key, with_favourite
from views.text import lap_ranges, listing, plural

ALL_DRIVERS = "All drivers"
DECISIONS = {
    "Penalty": ("penalty", "penalties"),
    "Investigation": ("investigation", "investigations"),
    "Track limits": ("lap time deleted for track limits", "lap times deleted for track limits"),
}


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
        default=with_favourite(race.drivers[:3], race.drivers),
        key=race_key(race, f"position_highlight-{favourite() or ''}"),
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


def weather_text(summary: dict) -> str:
    text = (
        f"Track {summary['track_start']:.0f} °C at the start and {summary['track_end']:.0f} °C "
        f"at the finish, air {summary['air_start']:.0f} °C."
    )
    rain = summary["rain_laps"]
    return f"{text} Rain on {lap_ranges(rain)}." if rain else f"{text} No rain."


def conditions_section(race: races.Race) -> None:
    st.subheader("What were the conditions?")
    by_lap = conditions.weather_by_lap(race.weather, race.laps)
    if by_lap.empty:
        st.info("Weather data isn't available for this race.")
        return
    st.markdown(weather_text(conditions.weather_summary(by_lap)))
    show(conditions.weather_figure(by_lap))


def decisions_text(counts: pd.Series) -> str:
    parts = [
        plural(int(counts[kind]), one, many)
        for kind, (one, many) in DECISIONS.items()
        if counts.get(kind, 0)
    ]
    return f"{listing(parts)}." if parts else "No penalties or investigations."


def race_control_section(race: races.Race) -> None:
    st.subheader("What did race control decide?")
    table = conditions.race_control(race.messages)
    if table.empty:
        st.info("Race control messages aren't available for this race.")
        return
    counts = table["Type"].value_counts()
    st.markdown(decisions_text(counts))
    present = [kind for kind in conditions.MESSAGE_TYPES if kind in counts.index]
    default = [kind for kind in ("Penalty", "Investigation", "Safety car") if kind in present]
    chosen = st.pills(
        "Show",
        present,
        selection_mode="multi",
        default=default or present,
        key=race_key(race, "message_types"),
    )
    shown = table[table["Type"].isin(chosen or present)]
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=min(420, 35 * (len(shown) + 1) + 3),
        column_config={
            "Lap": st.column_config.NumberColumn(width="small"),
            "Message": st.column_config.TextColumn(width=760),
        },
    )


def team_radio_section(race: races.Race) -> None:
    st.subheader("What was said on team radio?")
    clips = conditions.radio_by_lap(race.radio, race.laps)
    if clips.empty:
        st.info("Team radio isn't available for this race.")
        return
    speakers = [driver for driver in race.drivers if driver in set(clips["Driver"])]
    options = [ALL_DRIVERS, *speakers]
    liked = favourite()
    chosen = st.selectbox(
        "Driver",
        options,
        index=options.index(liked) if liked in speakers else 0,
        key=race_key(race, f"radio_driver-{liked or ''}"),
    )
    shown = clips if chosen == ALL_DRIVERS else clips[clips["Driver"] == chosen]
    shown = shown.reset_index(drop=True)
    table = pd.DataFrame(
        {
            "Lap": shown["Lap"].map(lambda lap: "Before the start" if lap == 0 else str(lap)),
            "Driver": shown["Driver"],
        }
    )
    picked = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(320, 35 * (len(table) + 1) + 3),
        on_select="rerun",
        selection_mode="single-row",
        key=f"radio:{race.year}:{race.round_number}:{chosen}",
    )
    rows = picked.selection.rows
    if rows:
        clip = shown.iloc[rows[0]]
        st.audio(clip["Url"], format="audio/mpeg")
    st.caption(
        f"{plural(len(shown), 'clip')}. Pick one to play it; the audio streams from the "
        "official timing service."
    )


def overview_tab(race: races.Race, summary: analysis.RaceSummary) -> None:
    if "Time" not in race.laps.columns:
        st.info("Lap timing isn't available for this race.")
        return
    highlight = running_order(race, summary)
    st.divider()
    gaps_section(race, highlight)
    st.divider()
    conditions_section(race)
    st.divider()
    race_control_section(race)
    st.divider()
    team_radio_section(race)
