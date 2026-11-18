import json
from pathlib import Path

import streamlit as st

from src import races
from views import data
from views.loading import loading
from views.state import favourite

TEMPLATE = Path(__file__).with_name("replay.html")
HEIGHT = 600


def replay_tab(race: races.Race, bundle_version: str) -> None:
    st.subheader("How did the race unfold on track?")
    if race.track.empty or not {"LapStartTime", "Time"} <= set(race.laps.columns):
        st.info("A track map isn't available for this session.")
        return
    with loading(
        "Building the replay", lambda: data.venue_facts(race.year, race.event, bundle_version)
    ):
        replay = data.replay_data(race.year, race.event, race.session, bundle_version, race)
    if not replay:
        st.info("There are no lap times to replay for this session.")
        return
    replay |= {"compounds": race.compound_colors, "favourite": favourite() or ""}
    # The data sits inside a script tag, so a "</" in it must not close the tag.
    payload = json.dumps(replay).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("/*DATA*/", payload).replace("/*HEIGHT*/", str(HEIGHT))
    st.iframe(page, height=HEIGHT + 8)
    st.caption(
        "Press play, pick a speed or drag the bar. 1× plays a race minute every second. "
        "Positions within a lap are estimated from lap times and the winner's fastest-lap speed "
        "profile, gaps are measured at the same point on track, and the order is the order on "
        "track, before any penalties."
    )
