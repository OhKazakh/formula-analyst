import pandas as pd
import streamlit as st

from src import analysis, championship, conditions, races, seasons
from views import logos, theme
from views.data import show
from views.state import favourite, race_key, with_favourite
from views.text import lap_ranges, listing, plural

ALL_DRIVERS = "All drivers"
BADGE = st.column_config.ImageColumn(" ", width="small")
STANDINGS_HEIGHT = 423
DECISIONS = {
    "Penalty": ("penalty", "penalties"),
    "Investigation": ("investigation", "investigations"),
    "Track limits": ("lap time deleted for track limits", "lap times deleted for track limits"),
}


def laps_led_text(summary: analysis.RaceSummary) -> str:
    leaders = list(summary.laps_led.items())
    if not leaders:
        return ""
    first, laps = leaders[0]
    if len(leaders) == 1:
        return f"**{first}** led every lap."
    others = listing([f"{driver} {count}" for driver, count in leaders[1:5]])
    changes = "once" if summary.lead_changes == 1 else f"{summary.lead_changes} times"
    return f"**{first}** led {laps} laps, {others}. The lead changed hands {changes}."


def running_order(race: races.Race, summary: analysis.RaceSummary) -> list[str]:
    st.subheader("How did the running order change?")
    if text := laps_led_text(summary):
        st.markdown(text)
    highlight = st.multiselect(
        "Highlight drivers",
        race.drivers,
        default=with_favourite(race.drivers[:3], race.drivers),
        key=race_key(race, f"position_highlight-{favourite() or ''}"),
    )
    positions = analysis.lap_positions(race.laps)
    show(analysis.position_figure(positions, race.drivers, race.driver_styles, highlight))
    st.caption("Position at the end of each lap, from the order drivers crossed the timing line.")
    return highlight


def winning_margin(gaps: pd.DataFrame, order: list[str]) -> str:
    final = gaps[gaps["LapNumber"] == gaps["LapNumber"].max()].set_index("Driver")["Gap"]
    if len(order) < 2 or order[1] not in final.index:
        return ""
    return f"**{order[0]}** won by {final[order[1]]:.1f} s from {order[1]}."


def gaps_section(race: races.Race, highlight: list[str]) -> None:
    st.subheader("How far behind the leader was everyone?")
    gaps = analysis.gap_to_leader(race.laps)
    if text := winning_margin(gaps, race.order):
        st.markdown(text)
    neutralised = analysis.neutralised_laps(race.laps)
    show(analysis.gap_figure(gaps, race.drivers, race.driver_styles, highlight, neutralised))
    caption = "The gap to the leader each time a driver crossed the timing line."
    if any(neutralised.values()):
        caption += " Shaded laps ran behind the safety car (SC) or the virtual safety car (VSC)."
    st.caption(caption)


def move_text(moved: float) -> str:
    if pd.isna(moved) or moved == 0:
        return "–"
    return f"▲{int(moved)}" if moved > 0 else f"▼{int(-moved)}"


def previous_round(season: seasons.Season, round_number: int) -> int | None:
    earlier = [completed for completed in season.completed_rounds if completed < round_number]
    return earlier[-1] if earlier else None


def driver_impact(season: seasons.Season, round_number: int) -> pd.DataFrame:
    after = championship.standings(season.results, round_number)
    before_round = previous_round(season, round_number)
    before = (
        championship.standings(season.results, before_round) if before_round else after.iloc[0:0]
    )
    return championship.standings_change(after, before, "Driver")


def team_impact(season: seasons.Season, round_number: int) -> pd.DataFrame:
    after = championship.team_standings(season.team_standings, round_number)
    before_round = previous_round(season, round_number)
    before = championship.team_standings(season.team_standings, before_round or 0)
    return championship.standings_change(after, before, "Team")


def impact_text(drivers: pd.DataFrame, teams: pd.DataFrame) -> str:
    parts = []
    for table, key, verb, label in (
        (drivers, "Name", "leads", "drivers"),
        (teams, "Team", "lead", "teams"),
    ):
        if len(table) > 1:
            first, second = table.iloc[0], table.iloc[1]
            margin = first["Points"] - second["Points"]
            parts.append(
                f"**{first[key]}** {verb} the {label} by {margin:g} "
                f"{'point' if margin == 1 else 'points'} over {second[key]}"
            )
    return f"After this race, {listing(parts)}." if parts else ""


def standings_frame(
    table: pd.DataFrame, key: str, label: str, colors: dict[str, str], previous: bool
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pos": table["Position"],
            "Badge": [logos.badge(team, colors.get(team)) for team in table["Team"]],
            label: table[key],
            "Points": table["Points"],
            "This race": table["Gained"],
            "Move": table["Moved"].map(move_text) if previous else "",
        }
    )


def championship_section(race: races.Race, season: seasons.Season | None) -> None:
    st.subheader("What did it mean for the championship?")
    if season is None or race.round_number not in season.completed_rounds:
        st.info("Championship standings aren't available for this race yet.")
        return
    drivers = driver_impact(season, race.round_number)
    teams = team_impact(season, race.round_number)
    if text := impact_text(drivers, teams):
        st.markdown(text)
    styles = season.driver_styles | race.driver_styles
    colors = championship.team_colors(season.results, styles)
    previous = previous_round(season, race.round_number) is not None
    config = {
        "Pos": st.column_config.NumberColumn(width="small"),
        "Badge": BADGE,
        "Driver": st.column_config.TextColumn(width=170),
        "Team": st.column_config.TextColumn(width=170),
        "Points": st.column_config.NumberColumn(format="%g"),
        "This race": st.column_config.NumberColumn(format="+%g"),
    }
    left, right = st.columns(2)
    left.dataframe(
        theme.favourite_rows(
            standings_frame(drivers, "Name", "Driver", colors, previous),
            drivers["Driver"].eq(favourite()),
        ),
        hide_index=True,
        width="stretch",
        height=STANDINGS_HEIGHT,
        column_config=config,
    )
    if not teams.empty:
        right.dataframe(
            standings_frame(teams, "Team", "Team", colors, previous),
            hide_index=True,
            width="stretch",
            height=STANDINGS_HEIGHT,
            column_config=config,
        )
    st.caption(
        "Standings after this round, with the points scored here, sprint included, and the "
        "places gained or lost. Team standings are the official ones, so they include any "
        "points deductions."
    )


def weather_text(summary: dict) -> str:
    text = (
        f"Track {summary['track_start']:.0f} °C at the start and {summary['track_end']:.0f} °C "
        f"at the finish, air {summary['air_start']:.0f} °C, humidity {summary['humidity']:.0f}% "
        f"and wind up to {summary['wind']:.1f} m/s."
    )
    rain = summary["rain_laps"]
    return f"{text} Rain on {lap_ranges(rain)}." if rain else f"{text} No rain."


def conditions_section(race: races.Race) -> None:
    st.subheader("What were the conditions?")
    by_lap = conditions.weather_by_lap(race.weather, race.laps)
    if by_lap.empty:
        st.info("Weather data isn't available for this race.")
        return
    st.markdown(weather_text(conditions.weather_summary(by_lap)))
    show(conditions.weather_figure(by_lap))


def decisions_text(counts: pd.Series) -> str:
    parts = [
        plural(int(counts[kind]), one, many)
        for kind, (one, many) in DECISIONS.items()
        if counts.get(kind, 0)
    ]
    return f"{listing(parts)}." if parts else "No penalties or investigations."


def track_limits_text(limits: pd.DataFrame) -> str:
    deleted = limits[limits["Deleted"] > 0]
    warned = limits.loc[limits["Warned"], "Driver"].tolist()
    parts = []
    if not deleted.empty:
        counts = listing([f"{row.Driver} {row.Deleted}" for row in deleted.itertuples()])
        parts.append(f"**Laps deleted for track limits:** {counts}.")
    if warned:
        parts.append(f"**Black-and-white flag:** {listing(warned)}.")
    return " ".join(parts)


def penalties_table(race: races.Race, penalties: pd.DataFrame) -> pd.DataFrame:
    teams = race.laps.drop_duplicates("Driver").set_index("Driver")["Team"]
    return pd.DataFrame(
        {
            "Lap": penalties["Lap"],
            "Badge": [
                logos.badge(
                    str(teams.get(driver, "")),
                    race.driver_styles.get(driver, {}).get("color"),
                )
                for driver in penalties["Driver"]
            ],
            "Driver": penalties["Driver"],
            "Penalty": penalties["Penalty"],
            "Reason": penalties["Reason"],
        }
    )


def race_control_section(race: races.Race) -> None:
    st.subheader("What did race control decide?")
    table = conditions.race_control(race.messages)
    if table.empty:
        st.info("Race control messages aren't available for this race.")
        return
    counts = table["Type"].value_counts()
    st.markdown(decisions_text(counts))
    penalties = conditions.penalties(race.messages)
    if not penalties.empty:
        st.dataframe(
            penalties_table(race, penalties),
            hide_index=True,
            width="stretch",
            column_config={
                "Lap": st.column_config.NumberColumn(width="small", format="%d"),
                "Badge": BADGE,
                "Penalty": st.column_config.TextColumn(width=170),
                "Reason": st.column_config.TextColumn(width=480),
            },
        )
    if text := track_limits_text(conditions.track_limits(race.messages)):
        st.markdown(text)
    present = [kind for kind in conditions.MESSAGE_TYPES if kind in counts.index]
    default = [kind for kind in ("Penalty", "Investigation", "Safety car") if kind in present]
    chosen = st.pills(
        "Show",
        present,
        selection_mode="multi",
        default=default or present,
        key=race_key(race, "message_types"),
    )
    shown = table[table["Type"].isin(chosen or present)]
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=min(420, 35 * (len(shown) + 1) + 3),
        column_config={
            "Lap": st.column_config.NumberColumn(width="small"),
            "Message": st.column_config.TextColumn(width=760),
        },
    )


def team_radio_section(race: races.Race) -> None:
    st.subheader("What was said on team radio?")
    clips = conditions.radio_by_lap(race.radio, race.laps)
    if clips.empty:
        st.info("Team radio isn't available for this race.")
        return
    speakers = [driver for driver in race.drivers if driver in set(clips["Driver"])]
    options = [ALL_DRIVERS, *speakers]
    liked = favourite()
    chosen = st.selectbox(
        "Driver",
        options,
        index=options.index(liked) if liked in speakers else 0,
        key=race_key(race, f"radio_driver-{liked or ''}"),
    )
    shown = clips if chosen == ALL_DRIVERS else clips[clips["Driver"] == chosen]
    shown = shown.reset_index(drop=True)
    table = pd.DataFrame(
        {
            "Lap": shown["Lap"].map(lambda lap: "Before the start" if lap == 0 else str(lap)),
            "Driver": shown["Driver"],
        }
    )
    picked = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(320, 35 * (len(table) + 1) + 3),
        on_select="rerun",
        selection_mode="single-row",
        key=f"radio:{race.year}:{race.round_number}:{chosen}",
    )
    rows = picked.selection.rows
    if rows:
        clip = shown.iloc[rows[0]]
        st.audio(clip["Url"], format="audio/mpeg")
    st.caption(
        f"{plural(len(shown), 'clip')}. Pick one to play it; the audio streams from the "
        "official timing service."
    )


def overview_tab(
    race: races.Race, summary: analysis.RaceSummary, season: seasons.Season | None
) -> None:
    if "Time" not in race.laps.columns:
        st.info("Lap timing isn't available for this race.")
        return
    highlight = running_order(race, summary)
    st.divider()
    gaps_section(race, highlight)
    st.divider()
    championship_section(race, season)
    st.divider()
    conditions_section(race)
    st.divider()
    race_control_section(race)
    st.divider()
    team_radio_section(race)
