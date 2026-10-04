from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from fastf1.ergast.interface import ErgastMultiResponse

from src.charts import Styles
from src.races import DATA_DIR, read_optional

RESULT_COLUMNS = [
    "Round",
    "Session",
    "Driver",
    "Name",
    "Team",
    "Position",
    "Classified",
    "Points",
    "Grid",
    "Status",
]
TEAM_COLUMNS = ["Round", "Team", "Position", "Points", "Wins"]
SCHEDULE_COLUMNS = ["Round", "EventName", "Country", "Location", "Sprint", "Date"]
SPRINT_FORMATS = {"sprint", "sprint_shootout", "sprint_qualifying"}


@dataclass(frozen=True, eq=False)
class Season:
    year: int
    schedule: pd.DataFrame
    results: pd.DataFrame
    driver_styles: Styles = field(default_factory=dict)
    team_standings: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TEAM_COLUMNS))

    @property
    def completed_rounds(self) -> list[int]:
        return sorted(int(r) for r in self.results["Round"].unique())

    def event_name(self, round_number: int) -> str:
        names = self.schedule.set_index("Round")["EventName"]
        return str(names.get(round_number, f"Round {round_number}"))


def fetch_season(year: int) -> Season:
    ergast = Ergast(result_type="pandas", auto_cast=True)
    sessions = [
        _results(ergast.get_race_results, year, "Race"),
        _results(ergast.get_sprint_results, year, "Sprint"),
    ]
    held = [results for results in sessions if not results.empty]
    results = pd.concat(held, ignore_index=True) if held else sessions[0]
    rounds = sorted(int(round_number) for round_number in results["Round"].unique())
    return Season(
        year=year,
        schedule=_schedule(year),
        results=results,
        team_standings=_team_standings(ergast, year, rounds),
    )


def _results(fetch: Callable[..., ErgastMultiResponse], year: int, session: str) -> pd.DataFrame:
    frames = []
    response = fetch(season=year, limit=100)
    while True:
        for description, content in zip(
            response.description.itertuples(), response.content, strict=True
        ):
            frames.append(content.assign(Round=description.round))
        try:
            response = response.get_next_result_page()
        except ValueError:
            break
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = pd.concat(frames, ignore_index=True)
    return pd.DataFrame(
        {
            "Round": results["Round"].astype(int),
            "Session": session,
            "Driver": results["driverCode"],
            "Name": results["givenName"] + " " + results["familyName"],
            "Team": results["constructorName"],
            "Position": results["position"].astype(int),
            "Classified": results["positionText"].astype(str).str.isdigit(),
            "Points": results["points"].astype(float),
            "Grid": results["grid"].astype(int),
            "Status": results["status"].astype(str),
        }
    )


# Official standings rather than summed driver points, which miss penalties such as
# Racing Point's 15-point deduction in 2020.
def _team_standings(ergast: Ergast, year: int, rounds: list[int]) -> pd.DataFrame:
    frames = []
    for round_number in rounds:
        response = ergast.get_constructor_standings(season=year, round=round_number)
        if not response.content:
            continue
        standings = response.content[0]
        position = pd.to_numeric(standings["position"], errors="coerce")
        frames.append(
            pd.DataFrame(
                {
                    "Round": round_number,
                    "Team": standings["constructorName"],
                    "Position": position.astype("Int64"),
                    "Points": standings["points"].astype(float),
                    "Wins": standings["wins"].astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TEAM_COLUMNS)


def _schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return pd.DataFrame(
        {
            "Round": schedule["RoundNumber"].astype(int),
            "EventName": schedule["EventName"],
            "Country": schedule["Country"],
            "Location": schedule["Location"],
            "Sprint": schedule["EventFormat"].isin(SPRINT_FORMATS),
            "Date": schedule["Session5DateUtc"],
        }
    ).reset_index(drop=True)


def save_season(season: Season, root: Path = DATA_DIR) -> Path:
    path = root / str(season.year)
    path.mkdir(parents=True, exist_ok=True)
    season.results.to_parquet(path / "results.parquet", index=False)
    season.schedule.to_parquet(path / "schedule.parquet", index=False)
    season.team_standings.to_parquet(path / "team_standings.parquet", index=False)
    return path


def saved_seasons(root: Path = DATA_DIR) -> list[int]:
    return sorted(int(path.parent.name) for path in root.glob("*/results.parquet"))


def load_season(year: int, root: Path = DATA_DIR) -> Season:
    path = root / str(year)
    return Season(
        year=year,
        schedule=pd.read_parquet(path / "schedule.parquet"),
        results=pd.read_parquet(path / "results.parquet"),
        driver_styles=_latest_styles(path),
        team_standings=read_optional(path / "team_standings.parquet", TEAM_COLUMNS),
    )


def _latest_styles(season_dir: Path) -> Styles:
    styles: Styles = {}
    for meta_path in sorted(season_dir.glob("*/race.json"), reverse=True):
        for driver, style in json.loads(meta_path.read_text())["driver_styles"].items():
            styles.setdefault(driver, style)
    return styles
