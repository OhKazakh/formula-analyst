import pandas as pd
import streamlit as st

from src import analysis, ideal_lap, races, sessions
from views import data, logos, theme, tyres
from views.data import show
from views.state import favourite

BADGE = st.column_config.ImageColumn(" ", width=44)
COMPUTER_GAP = "vs computer"


def lap_time(seconds: float) -> str:
    return "" if pd.isna(seconds) else analysis.format_seconds(seconds)


def gap_text(gap: float) -> str:
    if pd.isna(gap):
        return ""
    return f"+{gap:.3f}" if gap else "–"


def team_badges(race: races.Race, table: pd.DataFrame) -> list[str]:
    return [
        logos.badge(str(team), race.driver_styles.get(driver, {}).get("color"), race.year)
        for driver, team in zip(table["Driver"], table["Team"], strict=True)
    ]


def fastest_cells(table: pd.DataFrame, columns: list[str], times: pd.DataFrame):
    chosen = favourite()

    def colours(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        if chosen:
            styles.loc[(table["Driver"] == chosen).to_numpy(), :] = theme.FAVOURITE_ROW
        for column in columns:
            fastest = times[column].min()
            styles.loc[(times[column] == fastest).to_numpy(), column] = theme.FASTEST
        return styles

    return table.style.apply(colours, axis=None)


def beaten_cells(table: pd.DataFrame, beaten: pd.Series):
    chosen = favourite()

    def colours(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        if chosen:
            styles.loc[(table["Driver"] == chosen).to_numpy(), :] = theme.FAVOURITE_ROW
        styles.loc[beaten.to_numpy(), COMPUTER_GAP] = theme.PERSONAL_BEST
        return styles

    return table.style.apply(colours, axis=None)


def signed(seconds: float) -> str:
    return "" if pd.isna(seconds) else f"{seconds:+.3f}".replace("-", "−")


def perfect_lap_text(laps: pd.DataFrame, fastest: float) -> str:
    perfect = ideal_lap.perfect_lap(laps)
    if perfect.empty or perfect["Time"].sum() > fastest:
        return ""
    total = perfect["Time"].sum()
    owners = ", ".join(f"S{row.Sector} {row.Driver}" for row in perfect.itertuples())
    return (
        f"The **perfect lap**, the session's fastest three sectors joined up ({owners}), is "
        f"{lap_time(total)}: {fastest - total:.3f} s quicker than the fastest lap anyone drove."
    )


def computer_text(drivers: pd.Series, versus: pd.Series) -> str:
    if versus.isna().all():
        return ""
    text = "The **computer lap** simulates each car round this track, and "
    beaten = versus[versus < 0].sort_values()
    if len(beaten) == 1:
        driver = drivers[beaten.index[0]]
        return text + f"**{driver}** beat theirs by {-beaten.iloc[0]:.3f} s."
    if len(beaten) > 1:
        driver = drivers[beaten.index[0]]
        return (
            text + f"{len(beaten)} drivers beat theirs, **{driver}** by the most: "
            f"{-beaten.iloc[0]:.3f} s."
        )
    closest = versus.idxmin()
    return (
        text + f"nobody beat theirs. **{drivers[closest]}** came closest, "
        f"{versus[closest]:.3f} s off."
    )


def qualifying_tab(race: races.Race, bundle_version: str) -> None:
    sprint = race.session != "Qualifying"
    st.subheader("Who took pole for the sprint?" if sprint else "Who took pole?")
    results = sessions.qualifying_results(race.laps, race.order, race.session)
    labels = sessions.part_labels(race.session)
    if results.empty or results["Best"].isna().all():
        st.info("No lap times were set in this session.")
        return
    pole = results.iloc[0]
    second = results.iloc[1] if len(results) > 1 else None
    text = f"**{pole['Driver']}** took pole with {lap_time(pole['Best'])}"
    if second is not None and pd.notna(second["Gap"]):
        text += f", {second['Gap']:.3f} s ahead of {second['Driver']}"
    st.markdown(text + ".")

    reached = [int((results["Out"] == label).sum()) for label in labels]
    cutoffs = (reached[2], reached[2] + reached[1])
    show(
        sessions.gap_figure(
            results, race.driver_styles, cutoffs, favourite(), axis="Gap to pole (s)"
        )
    )

    table = pd.DataFrame(
        {
            "Pos": results["Position"],
            "Badge": team_badges(race, results),
            "Driver": results["Driver"],
            **{label: results[label].map(lap_time) for label in labels},
            "Gap": results["Gap"].map(gap_text),
        }
    )
    st.dataframe(
        fastest_cells(table, labels, results[labels]),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={"Pos": st.column_config.NumberColumn(width=44), "Badge": BADGE},
    )
    st.caption(
        f"Each driver's best lap in {', '.join(labels[:-1])} and {labels[-1]}, without laps "
        "deleted by the stewards, with the fastest in each part in purple. The gap is to pole, "
        "from the last part each driver reached. Grid penalties aren't applied."
    )
    computer_section(race, results, labels, bundle_version)


def computer_section(
    race: races.Race, results: pd.DataFrame, labels: list[str], bundle_version: str
) -> None:
    best = results[labels].min(axis=1)
    cars, missing = data.computer_laps(race.year, race.event, race.session, bundle_version, race)
    computer = results["Team"].map(cars)
    versus = best - computer
    st.subheader("What was the perfect lap?" if missing else "Did anyone beat the computer?")
    for line in (perfect_lap_text(race.laps, best.min()), computer_text(results["Driver"], versus)):
        if line:
            st.markdown(line)
    explained = (
        "The perfect lap joins the session's fastest time through each of the three timing sectors."
    )
    if missing:
        st.caption(f"{explained} There's no computer lap for this session: {missing}.")
        return

    order = versus.sort_values(na_position="last").index
    table = pd.DataFrame(
        {
            "Badge": team_badges(race, results),
            "Driver": results["Driver"],
            "Best lap": best.map(lap_time),
            "Computer lap": computer.map(lap_time),
            COMPUTER_GAP: versus.map(signed),
        }
    ).loc[order]
    st.dataframe(
        beaten_cells(table, (versus < 0).loc[order]),
        hide_index=True,
        width="content",
        height=35 * (len(table) + 1) + 3,
        column_config={"Badge": BADGE},
    )
    st.caption(
        f"{explained} The computer lap is a simulation of each car, built from its corner "
        "speeds, acceleration and braking on its fastest laps here and tuned on dry qualifying "
        "at this circuit in other years. Each driver's best lap is set against their car's. "
        "It's rough: in the 2018 to 2026 sessions that have one, it was within a second of a "
        "car's best lap 77% of the time, so a small gap either way means little. Green means "
        "the driver beat it."
    )


def practice_tab(race: races.Race) -> None:
    st.subheader("Who was quickest?")
    results = sessions.practice_results(race.laps)
    if results.empty:
        st.info("No lap times were set in this session.")
        return
    first = results.iloc[0]
    text = f"**{first['Driver']}** was quickest with {lap_time(first['Best'])}"
    if len(results) > 1:
        second = results.iloc[1]
        text += f", {second['Gap']:.3f} s ahead of {second['Driver']}"
    st.markdown(text + ".")
    show(sessions.gap_figure(results, race.driver_styles, highlight=favourite()))

    table = pd.DataFrame(
        {
            "Pos": results["Position"],
            "Badge": team_badges(race, results),
            "Driver": results["Driver"],
            "Best lap": results["Best"].map(lap_time),
            "Gap": results["Gap"].map(gap_text),
            "Tyre": [
                tyres.badge(compound, race.compound_colors) if pd.notna(compound) else None
                for compound in results["Compound"]
            ],
            "Laps": results["Laps"],
        }
    )
    st.dataframe(
        theme.favourite_rows(table, table["Driver"].eq(favourite())),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={
            "Pos": st.column_config.NumberColumn(width=44),
            "Badge": BADGE,
            "Tyre": st.column_config.ImageColumn("Tyre", width=60),
        },
    )
    st.caption(
        "Each driver's fastest lap that counted, on the tyre it was set on. Teams run different "
        "fuel loads and programmes in practice, so the order flatters whoever went for a quick "
        "lap on soft tyres."
    )
