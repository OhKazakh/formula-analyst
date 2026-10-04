import streamlit as st

from src import races

WIDGET_PREFIX = "race:"


def race_key(race: races.Race, name: str) -> str:
    return f"{WIDGET_PREFIX}{name}:{race.year}:{race.round_number}"


# Tabs only run while open, and Streamlit drops the state of widgets it didn't draw.
def keep_widget_state() -> None:
    for key in [key for key in st.session_state if str(key).startswith(WIDGET_PREFIX)]:
        st.session_state[key] = st.session_state[key]
