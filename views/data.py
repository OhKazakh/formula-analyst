import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.figure import Figure

from src import analysis, races, seasons


@st.cache_data(ttl="1h", show_spinner=False)
def live_timing() -> bool:
    return races.live_timing_available()


@st.cache_data(show_spinner=False)
def bundled_races(bundle_version: str) -> pd.DataFrame:
    return races.saved_races()


@st.cache_data(ttl="1d", show_spinner=False)
def calendar(year: int) -> pd.DataFrame:
    return races.race_calendar(year)


@st.cache_data(show_spinner=False, max_entries=16)
def race(year: int, event: str, bundle_version: str) -> races.Race:
    return races.load_race(year, event)


@st.cache_data(show_spinner=False, max_entries=8)
def replay_chart(year: int, event: str, bundle_version: str, _race: races.Race) -> go.Figure:
    traces = (_race.trace(driver) for driver in _race.drivers)
    reference = next((trace for trace in traces if not trace.empty), _race.telemetry)
    times = analysis.replay_times(_race.laps, _race.total_laps)
    positions = analysis.race_positions(_race.laps, times, analysis.lap_profile(reference))
    return analysis.replay_figure(
        _race.track,
        _race.map_corners,
        positions,
        _race.drivers,
        _race.driver_styles,
        _race.total_laps,
    )


@st.cache_data(show_spinner=False)
def bundled_seasons(bundle_version: str) -> list[int]:
    return seasons.saved_seasons()


@st.cache_data(show_spinner=False, max_entries=8)
def season(year: int, bundle_version: str) -> seasons.Season:
    return seasons.load_season(year)


def show(fig: Figure) -> None:
    st.pyplot(fig)
