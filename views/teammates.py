import pandas as pd
import streamlit as st

from src import championship, seasons, teammates
from views import data, logos, theme
from views.state import favourite
from views.text import surname

# Enough sessions together for a record to say something.
MIN_SESSIONS = 5


def quicker(row) -> str:
    if pd.isna(row.Gap):
        return "–"
    if not row.Gap:
        return "Level"
    faster = row.Driver if row.Gap < 0 else row.Teammate
    return f"{faster} by {abs(row.Gap):.3f} s"


def headline(table: pd.DataFrame, names: pd.Series, races: int) -> str:
    sessions = table["QualiAhead"] + table["QualiBehind"]
    # Regular pairings only: a few races as stand-in can look lopsided by chance.
    regular = table[(sessions >= MIN_SESSIONS) & (table["Rounds"] * 2 >= races)]
    if regular.empty:
        return ""
    share = regular["QualiAhead"] / (regular["QualiAhead"] + regular["QualiBehind"])
    best = regular.assign(Share=share).sort_values(["Share", "QualiAhead"]).iloc[-1]
    text = (
        f"Most one-sided in qualifying: **{surname(names[best['Driver']])}** against "
        f"{surname(names[best['Teammate']])}, {best['QualiAhead']}–{best['QualiBehind']}."
    )
    gaps = regular["Gap"].abs().dropna()
    if not gaps.empty:
        close = regular.loc[gaps.idxmin()]
        text += (
            f" Closest over a lap: **{surname(names[close['Driver']])}** and "
            f"**{surname(names[close['Teammate']])}**, {abs(close['Gap']):.3f} s apart."
        )
    return text


def team_order(season: seasons.Season, after: int, table: pd.DataFrame) -> list[str]:
    standings = championship.team_standings(season.team_standings, after)
    if not standings.empty:
        return standings["Team"].tolist()
    totals = (table["Points"] + table["TeammatePoints"]).groupby(table["Team"]).sum()
    return totals.sort_values(ascending=False).index.tolist()


def teammates_view(
    season: seasons.Season, after: int, colors: dict[str, str], bundle_version: str
) -> None:
    qualifying = data.season_qualifying(season.year, bundle_version)
    table = teammates.head_to_head(season.results, qualifying, after)
    if table.empty:
        st.info("No races have been run this season yet.")
        return
    order = {team: index for index, team in enumerate(team_order(season, after, table))}
    table = (
        table.assign(Order=table["Team"].map(order))
        .sort_values(["Order", "Rounds"], ascending=[True, False])
        .reset_index(drop=True)
    )
    names = season.results.drop_duplicates("Driver", keep="last").set_index("Driver")["Name"]

    st.subheader("Who beat their teammate?")
    races = season.results.loc[season.results["Round"] <= after, "Round"].nunique()
    st.markdown(headline(table, names, races))
    total = table["Points"] + table["TeammatePoints"]
    shown = pd.DataFrame(
        {
            "Badge": [logos.badge(team, colors.get(team), season.year) for team in table["Team"]],
            "Team": table["Team"],
            "Driver": table["Driver"].map(names),
            "Qualifying": [
                f"{ahead}–{behind}"
                for ahead, behind in zip(table["QualiAhead"], table["QualiBehind"], strict=True)
            ],
            "Race": [
                f"{ahead}–{behind}"
                for ahead, behind in zip(table["RaceAhead"], table["RaceBehind"], strict=True)
            ],
            "Points": [
                f"{mine:g}–{theirs:g}"
                for mine, theirs in zip(table["Points"], table["TeammatePoints"], strict=True)
            ],
            "Share": (table["Points"] / total * 100).where(total > 0),
            "Teammate": table["Teammate"].map(names),
            "Quali gap": [quicker(row) for row in table.itertuples()],
            "Rounds": table["Rounds"],
        }
    )
    marked = table["Driver"].eq(favourite()) | table["Teammate"].eq(favourite())
    st.dataframe(
        theme.favourite_rows(shown, marked),
        hide_index=True,
        width="stretch",
        height=35 * (len(shown) + 1) + 3,
        column_config={
            "Badge": st.column_config.ImageColumn(" ", width=40),
            "Team": st.column_config.TextColumn(width=116),
            "Driver": st.column_config.TextColumn(width=144),
            "Qualifying": st.column_config.TextColumn(
                width=84, help="Who qualified ahead, when both set a time"
            ),
            "Race": st.column_config.TextColumn(
                width=64, help="Who finished ahead, when both were classified"
            ),
            "Points": st.column_config.TextColumn(width=92, help="Race and sprint points"),
            "Share": st.column_config.ProgressColumn(
                width=100, min_value=0, max_value=100, format="%.0f%%", help="Share of the points"
            ),
            "Teammate": st.column_config.TextColumn(width=144),
            "Quali gap": st.column_config.TextColumn(
                width=128, help="Median gap in the last part of qualifying both reached"
            ),
            "Rounds": st.column_config.NumberColumn(width=64, help="Races as teammates"),
        },
    )
    st.caption(
        "Each pair of drivers who raced for the same team, with the higher scorer first. "
        "Qualifying counts the Grand Prix sessions in which both set a time, and the gap is the "
        "median difference in the last part both reached. The race counts the races both "
        "finished. Points include sprints."
    )
