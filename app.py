import streamlit as st

from src import races
from views import championship, race, theme

st.set_page_config(page_title="Formula Analyst", page_icon="🏁", layout="wide")
st.html(theme.GLOBAL_STYLE)
races.enable_cache()

page = st.navigation(
    [
        st.Page(race.render, title="Race", icon=":material/flag:", url_path="race", default=True),
        st.Page(
            championship.render,
            title="Championship",
            icon=":material/emoji_events:",
            url_path="championship",
        ),
    ]
)
page.run()
st.sidebar.caption(
    "Timing data via [FastF1](https://github.com/theOehrly/Fast-F1). Unofficial and non-commercial."
)
