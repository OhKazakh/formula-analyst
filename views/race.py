import html
from datetime import date

import pandas as pd
import streamlit as st

from src import analysis, races, seasons
from src.charts import FALLBACK_COLOR
from views import data, logos, theme
from views.links import requested, requested_int
from views.overview_tab import overview_tab
from views.pace_tab import pace_tab
from views.replay_tab import replay_tab
from views.state import keep_widget_state, pick_favourite
from views.strategy_tab import strategy_tab
from views.telemetry_tab import telemetry_tab
from views.timing_tab import timing_tab

FIRST_SEASON = 2018
DEFAULT_YEAR = 2024
DEFAULT_EVENT = "Italian Grand Prix"
TABS = ["Overview", "Timing", "Replay", "Pace", "Strategy", "Telemetry"]
HEADER_STYLE = f"""<style>
.race-podium {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.race-place {{
  flex: 1 1 180px; border-left: 4px solid; border-radius: 0 10px 10px 0;
  background: {theme.SURFACE}; padding: 8px 14px;
}}
.race-place .position {{ font-size: 13px; font-weight: 600; color: {theme.SECONDARY_TEXT}; }}
.race-place .name {{ font-size: 18px; font-weight: 600; }}
.race-place .team {{ font-size: 14px; color: {theme.SECONDARY_TEXT}; }}
.race-place .team img {{ width: 20px; height: 20px; vertical-align: -5px; margin-right: 6px; }}
.race-facts {{
  display: flex; flex-wrap: wrap; gap: 4px 24px; margin-top: 12px;
  font-size: 15px; font-variant-numeric: tabular-nums;
}}
.race-facts .label {{ color: {theme.SECONDARY_TEXT}; }}
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
    requested_year = requested_int("season")
    if requested_year in years:
        default_year = years.index(requested_year)
    else:
        default_year = years.index(DEFAULT_YEAR) if DEFAULT_YEAR in years else 0
    year = st.sidebar.selectbox("Season", years, index=default_year, key="race_season")

    events = season_events(year, bundled, online)
    if not events:
        st.sidebar.warning("No completed races for this season yet.")
        st.stop()
    countries = event_countries(year, bundle_version, online)
    requested_event = {races.slugify(name): name for name in events}.get(requested("race"))
    fallback = DEFAULT_EVENT if DEFAULT_EVENT in events else events[-1]
    default = events.index(requested_event or fallback)
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

    places = []
    for position, (driver, name, team) in enumerate(podium(race, season), start=1):
        color = race.driver_styles.get(driver, {}).get("color", FALLBACK_COLOR)
        places.append(
            f'<div class="race-place" style="border-left-color:{color}">'
            f'<div class="position">P{position}</div>'
            f'<div class="name">{html.escape(name)}</div>'
            f'<div class="team"><img src="{logos.badge(team, color)}" alt="">'
            f"{html.escape(team)}</div>"
            "</div>"
        )
    facts = "".join(f"<span>{fact}</span>" for fact in race_facts(summary))
    st.html(
        f'{HEADER_STYLE}<div class="race-podium">{"".join(places)}</div>'
        f'<div class="race-facts">{facts}</div>'
    )


def requested_tab() -> str | None:
    return {label.lower(): label for label in TABS}.get(requested("tab"))


def render() -> None:
    keep_widget_state()
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

    favourite = pick_favourite(race.drivers)
    season = bundled_season(year, bundle_version)
    summary = analysis.race_summary(race.laps, race.order[0] if race.order else "")
    header(race, summary, season)

    clean = analysis.representative_laps(race.laps)
    overview, timing, replay, pace, strategy, telemetry = st.tabs(
        TABS, default=requested_tab(), key="race_tab", on_change="rerun"
    )
    params = {
        "season": str(year),
        "race": races.slugify(event),
        "tab": st.session_state.get("race_tab", TABS[0]).lower(),
    }
    st.query_params.from_dict(params | ({"driver": favourite} if favourite else {}))
    if overview.open:
        with overview:
            overview_tab(race, summary)
    if timing.open:
        with timing:
            timing_tab(race, season)
    if replay.open:
        with replay:
            replay_tab(race, bundle_version)
    if pace.open:
        with pace:
            pace_tab(race, clean)
    if strategy.open:
        with strategy:
            strategy_tab(race, clean, bundle_version)
    if telemetry.open:
        with telemetry:
            telemetry_tab(race)
