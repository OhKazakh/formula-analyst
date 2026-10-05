import pandas as pd
import streamlit as st

from src import analysis, races, sessions
from views import logos, theme, tyres
from views.data import show
from views.state import favourite

PER_LAP = st.column_config.NumberColumn("Degradation", format="%+.3f s/lap")


def long_runs_tab(race: races.Race) -> None:
    st.subheader("Who looked quickest on a long run?")
    runs = sessions.long_runs(race.laps)
    if runs.empty:
        st.info(
            f"Nobody ran {sessions.LONG_RUN_LAPS} or more unbroken laps at race pace in this "
            "session."
        )
        return
    best = runs.drop_duplicates("Driver")
    top = best.iloc[0]
    compound = str(top["Compound"]).lower() if pd.notna(top["Compound"]) else "an unknown tyre"
    st.markdown(
        f"**{top['Driver']}** had the quickest long run, averaging "
        f"{analysis.format_seconds(top['Average'])} over {top['Laps']} laps on the {compound}."
    )
    laps = sessions.long_run_laps(race.laps)
    show(
        analysis.lap_distribution_figure(
            laps, best["Driver"].tolist(), race.driver_styles, race.compound_colors
        )
    )

    table = pd.DataFrame(
        {
            "Badge": [
                logos.badge(str(team), race.driver_styles.get(driver, {}).get("color"), race.year)
                for driver, team in zip(runs["Driver"], runs["Team"], strict=True)
            ],
            "Driver": runs["Driver"],
            "Tyre": [
                tyres.badge(compound, race.compound_colors) if pd.notna(compound) else None
                for compound in runs["Compound"]
            ],
            "From lap": runs["Start"],
            "Laps": runs["Laps"],
            "Average": runs["Average"].map(analysis.format_seconds),
            "Best": runs["Best"].map(analysis.format_seconds),
            "Degradation": runs["Degradation"],
        }
    )
    st.dataframe(
        theme.favourite_rows(table, table["Driver"].eq(favourite())),
        hide_index=True,
        width="stretch",
        column_config={
            "Badge": st.column_config.ImageColumn(" ", width=44),
            "Tyre": st.column_config.ImageColumn("Tyre", width=60),
            "Degradation": PER_LAP,
        },
    )
    st.caption(
        f"A long run is at least {sessions.LONG_RUN_LAPS} unbroken laps on one set of tyres, "
        f"each within {sessions.RUN_SPREAD:.1%} of that stint's typical lap, which is how "
        "teams practise for the race. Fuel loads aren't public, so a quicker run may just be "
        "a lighter car."
    )
