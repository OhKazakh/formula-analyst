import streamlit as st

from src import races
from views.links import requested

WIDGET_PREFIX = "race:"
FAVOURITE = "favourite"
FAVOURITE_PICKER = "favourite_picker"


def race_key(race: races.Race, name: str) -> str:
    return f"{WIDGET_PREFIX}{name}:{race.year}:{race.round_number}"


# Tabs only run while open, and Streamlit drops the state of widgets it didn't draw.
def keep_widget_state() -> None:
    for key in [key for key in st.session_state if str(key).startswith(WIDGET_PREFIX)]:
        st.session_state[key] = st.session_state[key]


def favourite() -> str | None:
    if FAVOURITE not in st.session_state:
        st.session_state[FAVOURITE] = requested("driver").upper()
    return st.session_state[FAVOURITE] or None


def _store_favourite() -> None:
    st.session_state[FAVOURITE] = st.session_state[FAVOURITE_PICKER]


# The picker only shows drivers in this race or season, but the choice is kept for the next one.
def pick_favourite(drivers: list[str]) -> str | None:
    chosen = favourite()
    st.session_state[FAVOURITE_PICKER] = chosen if chosen in drivers else ""
    st.sidebar.selectbox(
        "Favourite driver",
        ["", *drivers],
        format_func=lambda driver: driver or "None",
        key=FAVOURITE_PICKER,
        on_change=_store_favourite,
        help="Highlighted in tables and picked by default in charts.",
    )
    return favourite()


def with_favourite(drivers: list[str], available: list[str]) -> list[str]:
    chosen = favourite()
    if chosen in available and chosen not in drivers:
        return [*drivers, chosen]
    return drivers
