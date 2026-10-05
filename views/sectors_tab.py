import streamlit as st

from src import races
from views.pace_tab import sectors_section, speed_profile_section


def sectors_tab(race: races.Race) -> None:
    sectors_section(race)
    if not race.telemetry.empty:
        st.divider()
        speed_profile_section(race)
