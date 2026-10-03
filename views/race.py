import html
from datetime import date

import pandas as pd
import streamlit as st

from src import analysis, races, seasons
from views import data
from views.data import show

FIRST_SEASON = 2018
DEFAULT_YEAR = 2024
DEFAULT_EVENT = "Italian Grand Prix"
FASTER_WHERE = "Who's faster where"
SURFACE = "#F3F4F6"
HEADER_STYLE = f"""<style>
.race-podium {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.race-place {{
  flex: 1 1 180px; border-left: 4px solid; border-radius: 0 10px 10px 0;
  background: {SURFACE}; padding: 8px 14px;
}}
.race-place .position {{ font-size: 13px; font-weight: 600; color: {analysis.SECONDARY_INK}; }}
.race-place .name {{ font-size: 18px; font-weight: 600; color: {analysis.INK}; }}
.race-place .team {{ font-size: 14px; color: {analysis.SECONDARY_INK}; }}
.race-facts {{
  display: flex; flex-wrap: wrap; gap: 4px 24px; margin-top: 12px;
  font-size: 15px; color: {analysis.INK}; font-variant-numeric: tabular-nums;
}}
.race-facts .label {{ color: {analysis.SECONDARY_INK}; }}
@media (max-width: 640px) {{
  .race-podium {{ gap: 6px; }}
  .race-place {{
    flex-basis: 100%; display: flex; flex-wrap: wrap; align-items: baseline;
    column-gap: 10px; padding: 6px 12px;
  }}
  .race-place .name {{ font-size: 16px; }}
}}
</style>"""
COUNTRY_CODES = {
    "Abu Dhabi": "AE",
    "Australia": "AU",
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Bahrain": "BH",
    "Belgium": "BE",
    "Brazil": "BR",
    "Canada": "CA",
    "China": "CN",
    "France": "FR",
    "Germany": "DE",
    "Great Britain": "GB",
    "Hungary": "HU",
    "Italy": "IT",
    "Japan": "JP",
    "Mexico": "MX",
    "Monaco": "MC",
    "Netherlands": "NL",
    "Portugal": "PT",
    "Qatar": "QA",
    "Russia": "RU",
    "Saudi Arabia": "SA",
    "Singapore": "SG",
    "Spain": "ES",
    "Turkey": "TR",
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "United States": "US",
}


def flag(country: str | None) -> str:
    code = COUNTRY_CODES.get(country or "", "")
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in code)


def race_key(race: races.Race, name: str) -> str:
    return f"{name}-{race.year}-{race.round_number}"


def season_events(year: int, bundled: pd.DataFrame, online: bool) -> list[str]:
    offline_events = bundled.loc[bundled["Year"] == year, "EventName"].tolist()
    if not online:
        return offline_events
    try:
        return data.calendar(year)["EventName"].tolist()
    except Exception:
        return offline_events


def bundled_season(year: int, bundle_version: str) -> seasons.Season | None:
    if year not in data.bundled_seasons(bundle_version):
        return None
    return data.season(year, bundle_version)


def event_countries(year: int, bundle_version: str, online: bool) -> dict[str, str]:
    season = bundled_season(year, bundle_version)
    if season is not None and "Country" in season.schedule.columns:
        return dict(zip(season.schedule["EventName"], season.schedule["Country"], strict=True))
    if online:
        try:
            calendar = data.calendar(year)
        except Exception:
            return {}
        return dict(zip(calendar["EventName"], calendar["Country"], strict=True))
    return {}


def sidebar(bundle_version: str) -> tuple[int, str]:
    bundled = data.bundled_races(bundle_version)
    online = data.live_timing()

    years_available = set(bundled["Year"])
    if online:
        years_available |= set(range(FIRST_SEASON, date.today().year + 1))
    if not years_available:
        st.sidebar.error("No race data is available.")
        st.stop()
    years = sorted(years_available, reverse=True)
    default_year = years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years else 0
    year = st.sidebar.selectbox("Season", years, index=default_year, key="race_season")

    events = season_events(year, bundled, online)
    if not events:
        st.sidebar.warning("No completed races for this season yet.")
        st.stop()
    countries = event_countries(year, bundle_version, online)
    default = events.index(DEFAULT_EVENT) if DEFAULT_EVENT in events else len(events) - 1
    event = st.sidebar.selectbox(
        "Grand Prix",
        events,
        index=default,
        key="race_event",
        format_func=lambda name: f"{flag(countries.get(name))} {name}".strip(),
    )
    return year, event


def podium(race: races.Race, season: seasons.Season | None) -> list[tuple[str, str, str]]:
    people: dict[str, tuple[str, str]] = {}
    if season is not None:
        results = season.results
        classified = results[
            (results["Round"] == race.round_number) & (results["Session"] == "Race")
        ]
        people = {row.Driver: (row.Name, row.Team) for row in classified.itertuples()}
    teams = race.laps.drop_duplicates("Driver", keep="last").set_index("Driver")["Team"]
    return [
        (driver, *people.get(driver, (driver, str(teams.get(driver, "")))))
        for driver in race.order[:3]
    ]


def race_facts(summary: analysis.RaceSummary) -> list[str]:
    def plural(count: int, word: str) -> str:
        return f"{count} {word}" + ("" if count == 1 else "s")

    facts = []
    if summary.fastest is not None:
        fastest = summary.fastest
        facts.append(
            f'<span class="label">Fastest lap</span> <b>{html.escape(fastest.driver)}</b> '
            f"{analysis.format_lap_time(fastest.time)} on lap {fastest.lap}"
        )
    facts.append(plural(summary.pit_stops, "pit stop"))
    if summary.red_flag:
        facts.append("Red flag")
    if summary.safety_car_laps:
        facts.append(f"Safety car for {plural(summary.safety_car_laps, 'lap')}")
    if summary.virtual_safety_car_laps:
        facts.append(f"Virtual safety car for {plural(summary.virtual_safety_car_laps, 'lap')}")
    if not (summary.red_flag or summary.safety_car_laps or summary.virtual_safety_car_laps):
        facts.append("No safety car")
    return facts


def header(race: races.Race, summary: analysis.RaceSummary, season: seasons.Season | None) -> None:
    st.title(f"{race.year} {race.event}")
    details = [f"Round {race.round_number}"]
    if season is not None:
        details[0] += f" of {int(season.schedule['Round'].max())}"
        when = season.schedule.loc[season.schedule["Round"] == race.round_number, "Date"]
        if not when.empty and pd.notna(when.iloc[0]):
            details.append(f"{when.iloc[0].day} {when.iloc[0]:%B %Y}")
    details.append(f"{race.total_laps} laps")
    st.caption(" · ".join(details))

    places = "".join(
        f'<div class="race-place" style="border-left-color:'
        f'{race.driver_styles.get(driver, {}).get("color", analysis.FALLBACK_COLOR)}">'
        f'<div class="position">P{position}</div>'
        f'<div class="name">{html.escape(name)}</div>'
        f'<div class="team">{html.escape(team)}</div></div>'
        for position, (driver, name, team) in enumerate(podium(race, season), start=1)
    )
    facts = "".join(f"<span>{fact}</span>" for fact in race_facts(summary))
    st.html(
        f'{HEADER_STYLE}<div class="race-podium">{places}</div>'
        f'<div class="race-facts">{facts}</div>'
    )


def laps_led_text(summary: analysis.RaceSummary) -> str:
    leaders = list(summary.laps_led.items())
    if not leaders:
        return ""
    first, laps = leaders[0]
    if len(leaders) == 1:
        return f"**{first}** led every lap."
    others = [f"{driver} {count}" for driver, count in leaders[1:5]]
    listed = others[0] if len(others) == 1 else ", ".join(others[:-1]) + f" and {others[-1]}"
    changes = "once" if summary.lead_changes == 1 else f"{summary.lead_changes} times"
    return f"**{first}** led {laps} laps, {listed}. The lead changed hands {changes}."


def overview_tab(race: races.Race, summary: analysis.RaceSummary) -> None:
    st.subheader("How did the running order change?")
    if "Time" not in race.laps.columns:
        st.info("Lap timing isn't available for this race.")
        return
    if text := laps_led_text(summary):
        st.markdown(text)
    highlight = st.multiselect(
        "Highlight drivers",
        race.drivers,
        default=race.drivers[:3],
        key=race_key(race, "position_highlight"),
    )
    positions = analysis.lap_positions(race.laps)
    show(analysis.position_figure(positions, race.drivers, race.driver_styles, highlight))
    st.caption("Position at the end of each lap, from the order drivers crossed the timing line.")


def replay_tab(race: races.Race, bundle_version: str) -> None:
    st.subheader("How did the race unfold on track?")
    if race.track.empty or not {"LapStartTime", "Time"} <= set(race.laps.columns):
        st.info("A track map isn't available for this race.")
        return
    figure = data.replay_chart(race.year, race.event, bundle_version, race)
    st.plotly_chart(figure, config={"displayModeBar": False})
    st.caption(
        "Press Play or drag the slider. Positions within a lap are estimated from lap times "
        "and the winner's fastest-lap speed profile, and the order is the order on track, "
        "before any penalties."
    )


def driver_pace(race: races.Race, clean: pd.DataFrame) -> None:
    st.subheader("How did each driver's pace change over the race?")
    drivers = st.multiselect(
        "Drivers", race.drivers, default=race.drivers[:3], key=race_key(race, "pace_drivers")
    )
    with st.container(horizontal=True):
        representative = st.toggle(
            "Representative laps only",
            value=True,
            help="Excludes pit in/out laps, laps under safety car or flags, "
            "and laps FastF1 marks as inaccurate.",
        )
        fuel = st.toggle(
            "Fuel-corrected",
            value=False,
            help="Normalises every lap to an empty tank so tyre wear isn't hidden "
            "by the car getting lighter.",
        )

    if not drivers:
        st.info("Pick at least one driver.")
        return

    source = clean if representative else race.laps
    laps = source[source["Driver"].isin(drivers)].copy()
    if laps.empty:
        st.info(
            "None of these drivers have representative laps in this race. "
            "Turn off “Representative laps only” to see every lap."
            if representative
            else "No laps to show for these drivers."
        )
        return

    column = "LapTimeSeconds"
    if fuel:
        laps["FuelCorrected"] = analysis.fuel_corrected(laps, race.total_laps)
        column = "FuelCorrected"
    show(analysis.pace_figure(laps, drivers, race.driver_styles, column))


def team_pace(race: races.Race) -> None:
    st.subheader("Which teams were quickest?")
    pace = analysis.team_pace(race.laps)
    if pace.empty:
        st.info("This race has no representative laps to compare.")
        return
    if len(pace) > 1:
        st.markdown(
            f"**{pace['Team'].iloc[0]}** were quickest, {pace['Gap'].iloc[1]:.3f} s a lap "
            f"faster than {pace['Team'].iloc[1]}."
        )
    chart, table = st.columns([2, 1])
    with chart:
        show(analysis.team_pace_figure(race.laps, race.driver_styles))
    table.dataframe(
        pd.DataFrame(
            {
                "Team": pace["Team"],
                "Median lap": pace["Median"].map(analysis.format_seconds),
                "Gap": pace["Gap"].map(lambda gap: f"+{gap:.3f}s" if gap else "–"),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Racing laps within 107% of the fastest, without pit laps, safety car periods or "
        "inaccurate laps. Teams are ordered by their median lap time."
    )


def lap_distribution(race: races.Race) -> None:
    st.subheader("How consistent was each driver?")
    drivers = st.multiselect(
        "Drivers",
        race.drivers,
        default=race.drivers[:10],
        key=race_key(race, "distribution_drivers"),
    )
    if not drivers:
        st.info("Pick at least one driver.")
        return
    show(
        analysis.lap_distribution_figure(
            race.laps, drivers, race.driver_styles, race.compound_colors
        )
    )
    st.caption(
        "Each dot is one lap, coloured by tyre compound; the shape shows how lap times spread."
    )


def pace_tab(race: races.Race, clean: pd.DataFrame) -> None:
    driver_pace(race, clean)
    st.divider()
    team_pace(race)
    st.divider()
    lap_distribution(race)


def tyre_strategy(race: races.Race) -> None:
    st.subheader("What tyre strategy did everyone run?")
    stints = analysis.stint_summary(race.laps)
    show(analysis.strategy_figure(stints, race.drivers, race.compound_colors))
    with st.expander("Stint table"):
        st.dataframe(
            stints.rename(columns={"StartLap": "Start lap", "EndLap": "End lap"}),
            hide_index=True,
            width="stretch",
        )


def degradation(race: races.Race, clean: pd.DataFrame) -> None:
    st.subheader("How fast did the tyres degrade?")
    left, right = st.columns(2)
    min_laps = left.slider("Minimum laps per stint", 3, 20, 8)
    fuel_effect = right.slider(
        "Fuel effect (s per lap of fuel)", 0.0, 0.1, analysis.DEFAULT_FUEL_EFFECT, 0.005
    )

    laps = clean.copy()
    laps["FuelCorrected"] = analysis.fuel_corrected(laps, race.total_laps, fuel_effect)
    deg = analysis.degradation(laps, "FuelCorrected", min_laps)
    if deg.empty:
        st.info("No stints long enough to fit.")
        return

    per_lap = st.column_config.NumberColumn(format="%.3f s/lap")
    summary, table = st.columns([2, 3])
    summary.markdown("**By compound**")
    summary.dataframe(
        analysis.degradation_by_compound(deg)
        .reset_index()
        .rename(columns={"count": "Stints", "mean": "Mean", "median": "Median"}),
        hide_index=True,
        width="stretch",
        column_config={"Mean": per_lap, "Median": per_lap},
    )
    table.markdown("**By stint**")
    table.dataframe(
        deg.rename(columns={"DegPerLap": "Degradation"}),
        hide_index=True,
        width="stretch",
        column_config={"Degradation": per_lap},
    )
    st.caption(
        "Seconds lost per lap of tyre age, after fuel correction. A negative rate means the "
        "laps got quicker through the stint, for example as the track rubbered in or "
        "traffic cleared."
    )

    row = st.selectbox(
        "Inspect stint",
        deg.index,
        format_func=lambda i: (
            f"{deg.at[i, 'Driver']} · stint {deg.at[i, 'Stint']} · {deg.at[i, 'Compound']}"
        ),
    )
    driver, stint = deg.at[row, "Driver"], deg.at[row, "Stint"]
    stint_laps = laps[(laps["Driver"] == driver) & (laps["Stint"] == stint)]
    color = race.driver_styles.get(driver, {}).get("color", analysis.FALLBACK_COLOR)
    show(analysis.stint_fit_figure(stint_laps, "FuelCorrected", color))


def strategy_tab(race: races.Race, clean: pd.DataFrame) -> None:
    tyre_strategy(race)
    st.divider()
    degradation(race, clean)


def telemetry_tab(race: races.Race) -> None:
    st.subheader("How do two drivers compare on their fastest lap?")
    left, right = st.columns(2)
    first = left.selectbox("Driver A", race.drivers, index=0)
    second = right.selectbox("Driver B", race.drivers, index=min(1, len(race.drivers) - 1))

    drivers = list(dict.fromkeys([first, second]))
    laps = {driver: analysis.fastest_lap(race.laps, driver) for driver in drivers}
    timed = {driver: lap for driver, lap in laps.items() if lap is not None}
    if missing := [driver for driver in drivers if driver not in timed]:
        st.warning(f"No timed lap for {', '.join(missing)}.")
    if not timed:
        return

    summary = pd.DataFrame(
        [
            {
                "Driver": driver,
                "Lap": int(lap["LapNumber"]),
                "Time": analysis.format_lap_time(lap["LapTime"]),
                "Compound": (
                    lap["Compound"] if pd.notna(lap["Compound"]) else analysis.UNKNOWN_COMPOUND
                ),
                "Tyre age": int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
            }
            for driver, lap in timed.items()
        ]
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    if summary["Lap"].nunique() > 1:
        st.caption(
            "These laps were set at different points in the race, so fuel load and tyre age differ."
        )

    traces = {driver: race.trace(driver) for driver in timed}
    traces = {driver: trace for driver, trace in traces.items() if not trace.empty}
    if not traces:
        st.warning("Telemetry isn't available for these laps.")
        return
    show(analysis.speed_trace_figure(traces, race.driver_styles, race.corners))

    if race.track.empty:
        return
    st.subheader("Where on the lap?")
    options = [f"{driver} speed" for driver in traces]
    if len(traces) == 2:
        options.append(FASTER_WHERE)
    view = st.radio(
        "Colour the track by",
        options,
        horizontal=True,
        key=race_key(race, f"track_map_view-{first}-{second}"),
    )
    chart, _ = st.columns([3, 1])
    with chart:
        if view == FASTER_WHERE:
            show(
                analysis.dominance_map_figure(
                    race.track, race.map_corners, traces, race.driver_styles
                )
            )
            st.caption(
                f"The lap is split into {analysis.MINI_SECTORS} equal mini-sectors, each coloured "
                "by the driver who was quicker through it."
            )
        else:
            driver = view.removesuffix(" speed")
            show(analysis.speed_map_figure(race.track, race.map_corners, traces[driver]))


def render() -> None:
    bundle_version = races.bundle_version()
    year, event = sidebar(bundle_version)

    with st.spinner("Loading timing data. The first load of a race takes about a minute."):
        try:
            race = data.race(year, event, bundle_version)
        except races.RaceDataUnavailable as exc:
            st.title(f"{year} {event}")
            st.error(str(exc))
            st.button("Try again")
            st.stop()
        except Exception as exc:
            st.title(f"{year} {event}")
            st.error(f"Couldn't load this race: {exc}")
            st.button("Try again")
            st.stop()

    summary = analysis.race_summary(race.laps, race.order[0] if race.order else "")
    header(race, summary, bundled_season(year, bundle_version))

    clean = analysis.representative_laps(race.laps)
    overview, replay, pace, strategy, telemetry = st.tabs(
        ["Overview", "Replay", "Pace", "Strategy", "Telemetry"]
    )
    with overview:
        overview_tab(race, summary)
    with replay:
        replay_tab(race, bundle_version)
    with pace:
        pace_tab(race, clean)
    with strategy:
        strategy_tab(race, clean)
    with telemetry:
        telemetry_tab(race)
