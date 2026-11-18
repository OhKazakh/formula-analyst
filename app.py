import streamlit as st

from src import races
from views import pages, theme

st.set_page_config(page_title="Formula Analyst", page_icon="🏁", layout="wide")
st.html(theme.GLOBAL_STYLE)
races.enable_cache()

page = st.navigation(pages.ALL)
page.run()
st.sidebar.caption(
    "Timing data via [FastF1](https://github.com/theOehrly/Fast-F1) and "
    "[OpenF1](https://openf1.org). Unofficial and non-commercial."
)
