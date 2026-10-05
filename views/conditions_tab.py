import streamlit as st

from src import races
from views.overview_tab import conditions_section, race_control_section, team_radio_section


def conditions_tab(race: races.Race) -> None:
    conditions_section(race)
    st.divider()
    race_control_section(race)
    st.divider()
    team_radio_section(race)
