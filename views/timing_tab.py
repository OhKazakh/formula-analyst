import pandas as pd
import streamlit as st

from src import analysis, races, seasons, timing
from views import logos, theme, tyres
from views.data import show
from views.state import favourite, race_key, with_favourite

BADGE = st.column_config.ImageColumn(" ", width="small")
UNCLASSIFIED = {"Disqualified": "DSQ", "Did not start": "DNS", "Withdrew": "WD"}
LAP_TABLE_DRIVERS = 4
HISTORY_WIDTH = 220


def flagged(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def lap_time(seconds: float) -> str:
    return "" if pd.isna(seconds) else analysis.format_seconds(seconds)


def laps_behind(count: int) -> str:
    return f"+{count} lap" + ("" if count == 1 else "s")


def gap_text(row: pd.Series) -> str:
    if row["Out"]:
        return f"Out, {row['Laps']} {'lap' if row['Laps'] == 1 else 'laps'}"
    if row["Position"] == 1:
        return "Leader"
    if row["LapsDown"]:
        return laps_behind(row["LapsDown"])
    return f"+{row['Gap']:.3f}"


def interval_text(row: pd.Series) -> str:
    if row["Out"] or row["Position"] == 1:
        return ""
    if row["IntervalLaps"]:
        return laps_behind(row["IntervalLaps"])
    return f"+{row['Interval']:.3f}"


def tyre_capsule(row: pd.Series, colors: dict[str, str]) -> str | None:
    if pd.isna(row["Compound"]):
        return None
    if pd.isna(row["TyreAge"]):
        return tyres.badge(row["Compound"], colors)
    return tyres.stints([(row["Compound"], int(row["TyreAge"]))], colors)


def race_results(race: races.Race, season: seasons.Season | None) -> pd.DataFrame | None:
    if season is None or "Status" not in season.results:
        return None
    results = season.results
    rows = results[(results["Round"] == race.round_number) & (results["Session"] == "Race")]
    return rows.set_index("Driver") if not rows.empty else None


def result_text(row: pd.Series) -> str:
    if row["Classified"]:
        return str(int(row["Position"]))
    return UNCLASSIFIED.get(row["Status"], "DNF")


def tower_table(
    race: races.Race, standing: pd.DataFrame, results: pd.DataFrame | None
) -> pd.DataFrame:
    colors = {
        driver: race.driver_styles.get(driver, {}).get("color") for driver in standing["Driver"]
    }
    rows = [row for _, row in standing.iterrows()]
    table = pd.DataFrame(
        {
            "Pos": standing["Position"],
            "Team": [logos.badge(row["Team"], colors[row["Driver"]]) for row in rows],
            "Driver": standing["Driver"],
            "Interval": [interval_text(row) for row in rows],
            "Gap": [gap_text(row) for row in rows],
            "Last lap": standing["LastLap"].map(lap_time),
            "Best lap": standing["BestLap"].map(lap_time),
            "Tyre": [tyre_capsule(row, race.compound_colors) for row in rows],
            "Tyre history": [
                tyres.stints(row["Stints"], race.compound_colors, HISTORY_WIDTH) for row in rows
            ],
        }
    )
    if results is not None:
        known = standing["Driver"].isin(results.index)
        grid = standing["Driver"].map(results["Grid"])
        table["Grid"] = [
            ("Pit lane" if start == 0 else str(int(start))) if ok else ""
            for start, ok in zip(grid, known, strict=True)
        ]
        table["Result"] = [
            result_text(results.loc[driver]) if ok else ""
            for driver, ok in zip(standing["Driver"], known, strict=True)
        ]
        table["Points"] = standing["Driver"].map(results["Points"]).fillna(0)
    return table


def tower_style(table: pd.DataFrame, standing: pd.DataFrame, chosen: str | None):
    def colours(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        if chosen:
            styles.loc[(standing["Driver"] == chosen).to_numpy(), :] = theme.FAVOURITE_ROW
        styles.loc[standing["LastPersonalBest"].to_numpy(), "Last lap"] = theme.PERSONAL_BEST
        styles.loc[standing["LastFastest"].to_numpy(), "Last lap"] = theme.FASTEST
        styles.loc[standing["BestFastest"].to_numpy(), "Best lap"] = theme.FASTEST
        return styles

    return table.style.apply(colours, axis=None)


def lap_summary(standing: pd.DataFrame, lap: int, total: int) -> str:
    leader = standing.iloc[0]
    second = standing.iloc[1] if len(standing) > 1 else None
    if lap == total:
        if second is None or second["LapsDown"] or second["Out"]:
            return f"**{leader['Driver']}** took the flag."
        return (
            f"**{leader['Driver']}** took the flag {second['Gap']:.1f} s ahead of "
            f"{second['Driver']}."
        )
    if second is None or second["LapsDown"] or second["Out"]:
        return f"After lap {lap} of {total}, **{leader['Driver']}** leads."
    return (
        f"After lap {lap} of {total}, **{leader['Driver']}** leads {second['Driver']} by "
        f"{second['Gap']:.1f} s."
    )


def tower_section(race: races.Race, season: seasons.Season | None) -> None:
    st.subheader("Where was everyone after each lap?")
    total = int(race.laps["LapNumber"].max())
    lap = (
        st.slider("After lap", 1, total, total, key=race_key(race, "tower_lap"))
        if total > 1
        else total
    )
    standing = timing.tower(race.laps, lap)
    if standing.empty:
        st.info("No timed laps yet at this point of the race.")
        return
    chosen = favourite()
    st.markdown(lap_summary(standing, lap, total))
    show(timing.spread_figure(standing, race.driver_styles, chosen))
    results = race_results(race, season) if lap == total else None
    table = tower_table(race, standing, results)
    st.dataframe(
        tower_style(table, standing, chosen),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={
            "Pos": st.column_config.NumberColumn(width=44),
            "Team": st.column_config.ImageColumn(" ", width=44),
            "Driver": st.column_config.TextColumn(width=56),
            "Interval": st.column_config.TextColumn(width=70),
            "Gap": st.column_config.TextColumn(width=82),
            "Last lap": st.column_config.TextColumn(width=78),
            "Best lap": st.column_config.TextColumn(width=78),
            "Tyre": st.column_config.ImageColumn("Tyre", width=64),
            "Tyre history": st.column_config.ImageColumn("Tyre history", width=HISTORY_WIDTH + 12),
            "Grid": st.column_config.TextColumn(width=48),
            "Result": st.column_config.TextColumn(width=52),
            "Points": st.column_config.NumberColumn(format="%g", width=52),
        },
    )
    caption = (
        "Gaps are taken as each car crosses the timing line, so the order is the order on "
        "track. Purple is the fastest lap of the race so far, green a driver's personal best."
    )
    if results is not None:
        caption += " Result is the official classification, after any time penalties."
    st.caption(caption)


def lap_table(race: races.Race, drivers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = timing.lap_times(race.laps, drivers)
    laps = sorted(times["LapNumber"].unique())
    table = pd.DataFrame({"Lap": [int(lap) for lap in laps]})
    styles = pd.DataFrame("", index=table.index, columns=["Lap"])
    for driver in drivers:
        rows = times[times["Driver"] == driver].set_index("LapNumber").reindex(laps)
        table[driver] = [
            lap_time(seconds) + (" · pit" if flagged(pit) else "")
            for seconds, pit in zip(rows["LapTimeSeconds"], rows["Pit"], strict=True)
        ]
        table[f"{driver} tyre"] = [
            tyres.stints([(compound, int(age))], race.compound_colors)
            if pd.notna(compound) and pd.notna(age)
            else None
            for compound, age in zip(rows["Compound"], rows["TyreLife"], strict=True)
        ]
        styles[driver] = [
            theme.FASTEST if flagged(fastest) else theme.PERSONAL_BEST if flagged(best) else ""
            for fastest, best in zip(rows["Fastest"], rows["PersonalBest"], strict=True)
        ]
        styles[f"{driver} tyre"] = ""
    return table, styles


def lap_by_lap(race: races.Race) -> None:
    st.subheader("How did their lap times compare, lap by lap?")
    chosen = favourite()
    drivers = st.multiselect(
        "Drivers",
        race.drivers,
        default=with_favourite(race.drivers[:3], race.drivers),
        max_selections=LAP_TABLE_DRIVERS,
        key=race_key(race, f"lap_table-{chosen or ''}"),
    )
    if not drivers:
        st.info("Pick at least one driver.")
        return
    table, styles = lap_table(race, drivers)
    st.dataframe(
        table.style.apply(lambda _: styles, axis=None),
        hide_index=True,
        width="content",
        height=560,
        column_config={
            "Lap": st.column_config.NumberColumn(width=50),
            **{driver: st.column_config.TextColumn(width=120) for driver in drivers},
            **{f"{driver} tyre": st.column_config.ImageColumn(" ", width=70) for driver in drivers},
        },
    )
    st.caption(
        "Purple is the fastest lap of the race and green each driver's best. "
        "The tyre shows its age in laps."
    )


def timing_tab(race: races.Race, season: seasons.Season | None) -> None:
    if "Time" not in race.laps.columns:
        st.info("Lap timing isn't available for this race.")
        return
    tower_section(race, season)
    st.divider()
    lap_by_lap(race)
