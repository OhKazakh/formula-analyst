import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analysis, degradation, predictions, races, seasons, teammates


@st.cache_data(ttl="1h", show_spinner=False)
def live_timing() -> bool:
    return races.live_timing_available()


@st.cache_data(show_spinner=False)
def bundled_sessions(bundle_version: str) -> pd.DataFrame:
    return races.saved_sessions()


@st.cache_data(ttl="1d", show_spinner=False)
def calendar(year: int) -> pd.DataFrame:
    return races.race_calendar(year)


@st.cache_data(show_spinner=False, max_entries=16)
def race(year: int, event: str, session: str, bundle_version: str) -> races.Race:
    return races.load_race(year, event, session)


@st.cache_data(show_spinner=False, max_entries=8)
def replay_data(
    year: int, event: str, session: str, bundle_version: str, _race: races.Race
) -> dict:
    traces = (_race.trace(driver) for driver in _race.drivers)
    reference = next((trace for trace in traces if not trace.empty), _race.telemetry)
    return analysis.replay_data(
        _race.laps,
        _race.track,
        _race.map_corners,
        _race.drivers,
        _race.driver_styles,
        analysis.lap_profile(reference),
        _race.total_laps,
        starting_grid(_race, bundle_version),
    )


def starting_grid(race: races.Race, bundle_version: str) -> list[str]:
    if race.year not in seasons.saved_seasons() or race.session not in races.RACES:
        return []
    results = season(race.year, bundle_version).results
    start = results[(results["Round"] == race.round_number) & (results["Session"] == race.session)]
    # Grid 0 is a pit-lane start, behind everyone.
    return start.sort_values("Grid", key=lambda grid: grid.replace(0, 99))["Driver"].tolist()


@st.cache_data(show_spinner=False)
def bundled_seasons(bundle_version: str) -> list[int]:
    return seasons.saved_seasons()


@st.cache_data(show_spinner=False, max_entries=8)
def season(year: int, bundle_version: str) -> seasons.Season:
    return seasons.load_season(year)


@st.cache_data(show_spinner=False, max_entries=8)
def season_qualifying(year: int, bundle_version: str) -> pd.DataFrame:
    return teammates.season_qualifying(year)


@st.cache_data(show_spinner=False)
def tyre_curves(bundle_version: str) -> pd.DataFrame:
    return degradation.load_curves()


@st.cache_data(show_spinner=False)
def tyre_metrics(bundle_version: str) -> dict | None:
    return degradation.load_metrics()


@st.cache_data(show_spinner=False)
def race_predictions(bundle_version: str) -> pd.DataFrame:
    return predictions.load_predictions()


@st.cache_data(show_spinner=False)
def prediction_metrics(bundle_version: str) -> dict:
    return predictions.load_metrics()


def show(figure: go.Figure) -> None:
    st.plotly_chart(figure, config={"displayModeBar": False})
