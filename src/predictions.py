from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import races, seasons, sessions

MODEL_DIR = "race-model"
FORM_RACES = 5
FEATURES = ["QualiPosition", "QualiGap", "LongRun", "Form", "TeamForm"]
TABLE_COLUMNS = ["Year", "Round", "Driver", "Team", *FEATURES, "Position"]
PREDICTION_COLUMNS = ["Year", "Round", "Driver", "Team", "QualiPosition", "Predicted", "Position"]
RECORD_COLUMNS = [
    "Round",
    "Picked",
    "PickedFrom",
    "Winner",
    "WinnerFrom",
    "Called",
    "Podium",
    "ModelError",
    "GridError",
]


def model_dir(root: Path = races.DATA_DIR) -> Path:
    return root / MODEL_DIR


def qualifying_features(path: Path) -> pd.DataFrame:
    table = sessions.saved_qualifying(path)
    pole = table["Best"].iloc[0] if len(table) else np.nan
    return pd.DataFrame(
        {
            "Driver": table["Driver"],
            "Team": table["Team"],
            "QualiPosition": table["Position"].astype(float),
            "QualiGap": table["Gap"] / pole * 100,
        }
    )


def long_run_pace(paths: list[Path]) -> pd.Series:
    """Each driver's quickest long run of the weekend, in percent off the quickest one."""
    gaps = []
    for path in paths:
        laps = pd.read_parquet(path / "laps.parquet")
        runs = sessions.long_runs(races.tyre_stints(laps))
        if runs.empty:
            continue
        fastest = runs.groupby("Driver")["Average"].min()
        gaps.append((fastest / fastest.min() - 1) * 100)
    return pd.concat(gaps, axis=1).min(axis=1) if gaps else pd.Series(dtype=float)


def recent_form(race_results: pd.DataFrame, by: str, before: int) -> pd.Series:
    """Average finishing position over the last few races before a round (`Order` below it)."""
    earlier = race_results[race_results["Order"] < before]
    if earlier.empty:
        return pd.Series(dtype=float)
    # Each driver's (or team's) own last races, which skips the rounds it missed.
    recent = earlier.groupby(by)["Order"].rank(method="dense", ascending=False) <= FORM_RACES
    return earlier[recent].groupby(by)["Position"].mean()


def all_race_results(root: Path = races.DATA_DIR) -> pd.DataFrame:
    frames = []
    for year in seasons.saved_seasons(root):
        results = seasons.load_season(year, root).results
        frames.append(results[results["Session"] == "Race"].assign(Year=year))
    results = pd.concat(frames, ignore_index=True)
    return results.assign(Order=results["Year"] * 100 + results["Round"])


def feature_table(root: Path = races.DATA_DIR) -> pd.DataFrame:
    """One row per driver and race weekend with qualifying, using only what's known beforehand."""
    saved = races.saved_sessions(root)
    results = all_race_results(root)
    finishes = results.set_index(["Order", "Driver"])["Position"]
    rows = []
    for (year, round_number), weekend in saved.groupby(["Year", "Round"]):
        qualifying = weekend[weekend["Session"] == "Qualifying"]
        if qualifying.empty:
            continue
        order = int(year) * 100 + int(round_number)
        table = qualifying_features(qualifying["Path"].iloc[0])
        practice = weekend.loc[weekend["Session"].isin(races.PRACTICE), "Path"].tolist()
        table["LongRun"] = table["Driver"].map(long_run_pace(practice))
        table["Form"] = table["Driver"].map(recent_form(results, "Driver", order))
        # Timing data and the results name teams differently, so go through each driver.
        teams = results[results["Order"] <= order].sort_values("Order").groupby("Driver")["Team"]
        team_form = recent_form(results, "Team", order)
        table["TeamForm"] = table["Driver"].map(teams.last()).map(team_form)
        table["Position"] = [finishes.get((order, driver), np.nan) for driver in table["Driver"]]
        rows.append(table.assign(Year=int(year), Round=int(round_number)))
    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    return pd.concat(rows, ignore_index=True)[TABLE_COLUMNS]


def scores(predictions: pd.DataFrame, column: str) -> dict[str, float]:
    """How well an order (`Predicted` or `QualiPosition`) matched the race results."""
    known = predictions.dropna(subset=["Position"])
    hits, podiums, errors = [], [], []
    for _, race in known.groupby(["Year", "Round"]):
        ranked = race[column].rank(method="first")
        hits.append(float(race.loc[ranked.idxmin(), "Position"] == 1))
        predicted_podium = set(race.loc[ranked <= 3, "Driver"])
        podiums.append(len(predicted_podium & set(race.loc[race["Position"] <= 3, "Driver"])) / 3)
        errors.extend((ranked - race["Position"]).abs())
    return {
        "winner": round(float(np.mean(hits)), 3),
        "podium": round(float(np.mean(podiums)), 3),
        "position_error": round(float(np.mean(errors)), 2),
    }


def upcoming_weekend(predictions: pd.DataFrame) -> tuple[int, int] | None:
    """The latest weekend predicted but not raced yet, when nobody has a finishing position.

    A driver who didn't start has no position either, so one gap doesn't make a race upcoming.
    """
    if predictions.empty:
        return None
    waiting = predictions["Position"].isna().groupby([predictions["Year"], predictions["Round"]])
    waiting = waiting.all()
    waiting = waiting[waiting]
    if waiting.empty:
        return None
    year, round_number = waiting.index.max()
    return int(year), int(round_number)


def season_record(predictions: pd.DataFrame, year: int) -> pd.DataFrame:
    """Each race of a season: the model's pick against the winner, the podium and the errors."""
    rows = []
    for round_number, weekend in predictions[predictions["Year"] == year].groupby("Round"):
        raced = weekend.dropna(subset=["Position"])
        if raced.empty:
            continue
        # The pick stands even if that driver didn't start.
        picked = weekend.loc[weekend["Predicted"].idxmin()]
        winner = raced.loc[raced["Position"].idxmin()]
        podium = set(weekend.loc[weekend["Predicted"] <= 3, "Driver"]) & set(
            raced.loc[raced["Position"] <= 3, "Driver"]
        )
        rows.append(
            {
                "Round": int(round_number),
                "Picked": picked["Driver"],
                "PickedFrom": picked["QualiPosition"],
                "Winner": winner["Driver"],
                "WinnerFrom": winner["QualiPosition"],
                "Called": picked["Driver"] == winner["Driver"],
                "Podium": len(podium),
                "ModelError": (raced["Predicted"] - raced["Position"]).abs().mean(),
                "GridError": (raced["QualiPosition"] - raced["Position"]).abs().mean(),
            }
        )
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


def load_predictions(root: Path = races.DATA_DIR) -> pd.DataFrame:
    return races.read_optional(model_dir(root) / "predictions.parquet", PREDICTION_COLUMNS)


def load_metrics(root: Path = races.DATA_DIR) -> dict:
    path = model_dir(root) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}
