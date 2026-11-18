"""The app's pages, in one place so that one page can link to another."""

import streamlit as st

from views import championship, predictions_page, race

RACE = st.Page(race.render, title="Race", icon=":material/flag:", url_path="race", default=True)
CHAMPIONSHIP = st.Page(
    championship.render,
    title="Championship",
    icon=":material/emoji_events:",
    url_path="championship",
)
PREDICTIONS = st.Page(
    predictions_page.render,
    title="Predictions",
    icon=":material/insights:",
    url_path="predictions",
)
ALL = [RACE, CHAMPIONSHIP, PREDICTIONS]
