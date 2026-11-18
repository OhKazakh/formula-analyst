import re
from collections.abc import Callable
from typing import TypeVar

import streamlit as st

from src import races
from views.links import requested

WIDGET_PREFIX = "race:"
FAVOURITE = "favourite"
FAVOURITE_PICKER = "favourite_picker"
FAVOURITE_SAVED = "favourite_saved"
# A year, so the favourite is still there on the next visit.
COOKIE = "favourite"
COOKIE_AGE = 365 * 24 * 3600
# The season, round and tabs follow you between the race and championship pages. Streamlit drops
# a page's widget state when you leave it, so the pages keep their own copy here.
SEASON = "shared:season"
ROUND = "shared:round"


def race_key(race: races.Race, name: str) -> str:
    return f"{WIDGET_PREFIX}{name}:{race.year}:{race.round_number}"


# Tabs only run while open, and Streamlit drops the state of widgets it didn't draw.
def keep_widget_state() -> None:
    for key in [key for key in st.session_state if str(key).startswith(WIDGET_PREFIX)]:
        st.session_state[key] = st.session_state[key]


T = TypeVar("T")


# Streamlit resets a widget whose default changes between reruns, which undoes the very click
# that changed it. So a default is worked out when its widget appears, then kept as it is.
def initial(widget: str, choose: Callable[[], T], context: object = None) -> T:
    store = f"initial:{widget}"
    saved = st.session_state.get(store)
    if widget not in st.session_state or saved is None or saved[0] != context:
        saved = (context, choose())
        st.session_state[store] = saved
    return saved[1]


def remember(season: int, round_number: int) -> None:
    st.session_state[SEASON] = season
    st.session_state[ROUND] = round_number


def remembered_season() -> int | None:
    return st.session_state.get(SEASON)


def remembered_round(season: int) -> int | None:
    return st.session_state.get(ROUND) if st.session_state.get(SEASON) == season else None


def remembered_tab(key: str, labels: list[str]) -> str | None:
    """The tab last open under this key, if this page has it."""
    label = st.session_state.get(f"shared:{key}")
    return label if label in labels else None


def remember_tab(key: str, label: str) -> None:
    st.session_state[f"shared:{key}"] = label


def driver_code(value: object) -> str:
    """A driver code from a link or cookie, or nothing: it ends up inside a script."""
    if not isinstance(value, str):
        return ""
    value = value.strip().upper()
    return value if re.fullmatch(r"[A-Z0-9]{2,4}", value) else ""


def favourite() -> str | None:
    if FAVOURITE not in st.session_state:
        saved = driver_code(st.context.cookies.get(COOKIE, ""))
        st.session_state[FAVOURITE] = driver_code(requested("driver")) or saved
        st.session_state[FAVOURITE_SAVED] = saved
    return st.session_state[FAVOURITE] or None


def _store_favourite() -> None:
    st.session_state[FAVOURITE] = st.session_state[FAVOURITE_PICKER]


def _save_favourite() -> None:
    """Writes the favourite to a cookie when it changed, so it's back on the next visit."""
    chosen = st.session_state[FAVOURITE]
    if chosen == st.session_state.get(FAVOURITE_SAVED):
        return
    st.session_state[FAVOURITE_SAVED] = chosen
    age = COOKIE_AGE if chosen else 0
    st.sidebar.html(
        f'<script>document.cookie = "{COOKIE}={chosen}; max-age={age}; path=/; SameSite=Lax";'
        "</script>",
        unsafe_allow_javascript=True,
    )


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
        help="Highlighted in tables and picked by default in charts. Kept for your next visit.",
    )
    _save_favourite()
    return favourite()


def with_favourite(drivers: list[str], available: list[str]) -> list[str]:
    chosen = favourite()
    if chosen in available and chosen not in drivers:
        return [*drivers, chosen]
    return drivers
