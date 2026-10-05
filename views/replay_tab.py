import streamlit as st

from src import races
from views import data


def replay_tab(race: races.Race, bundle_version: str) -> None:
    st.subheader("How did the race unfold on track?")
    if race.track.empty or not {"LapStartTime", "Time"} <= set(race.laps.columns):
        st.info("A track map isn't available for this session.")
        return
    figure = data.replay_chart(race.year, race.event, race.session, bundle_version, race)
    st.plotly_chart(figure, config={"displayModeBar": False})
    st.caption(
        "Press Play or drag the slider. Positions within a lap are estimated from lap times "
        "and the winner's fastest-lap speed profile, and the order is the order on track, "
        "before any penalties."
    )
