"""Teammate head-to-heads: the only rival with the same car."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import races, sessions

PARTS = ["Q3", "Q2", "Q1"]
QUALIFYING_COLUMNS = ["Round", "Driver", "Position", *PARTS]
HEAD_TO_HEAD_COLUMNS = [
    "Team",
    "Driver",
    "Teammate",
    "Rounds",
    "QualiAhead",
    "QualiBehind",
    "RaceAhead",
    "RaceBehind",
    "Points",
    "TeammatePoints",
    "Gap",
]
# Swapping the two drivers swaps these columns.
MIRRORED = {
    "Driver": "Teammate",
    "QualiAhead": "QualiBehind",
    "RaceAhead": "RaceBehind",
    "Points": "TeammatePoints",
}


def season_qualifying(year: int, root: Path = races.DATA_DIR) -> pd.DataFrame:
    """Every bundled Grand Prix qualifying of a season, one row per driver and round."""
    saved = races.saved_sessions(root)
    chosen = saved[(saved["Year"] == year) & (saved["Session"] == "Qualifying")]
    frames = [
        sessions.saved_qualifying(path).assign(Round=int(round_number))
        for round_number, path in zip(chosen["Round"], chosen["Path"], strict=True)
    ]
    if not frames:
        return pd.DataFrame(columns=QUALIFYING_COLUMNS)
    return pd.concat(frames, ignore_index=True)[QUALIFYING_COLUMNS]


def pairs(results: pd.DataFrame, after_round: int) -> pd.DataFrame:
    """Each round's pairs of teammates in the race, once per pair."""
    race = results[(results["Session"] == races.RACE) & (results["Round"] <= after_round)]
    both = race.merge(race, on=["Round", "Team"], suffixes=("", "Mate"))
    return both[both["Driver"] < both["DriverMate"]].reset_index(drop=True)


def head_to_head(results: pd.DataFrame, qualifying: pd.DataFrame, after_round: int) -> pd.DataFrame:
    """Who beat whom in qualifying and the race, for every pair of teammates.

    Qualifying counts the sessions both drivers set a time in, and the gap is the median
    difference in the last part both reached, negative when `Driver` was quicker. The race
    counts the races both finished. Points include sprints. `Driver` is the one with more.
    """
    paired = pairs(results, after_round)
    if paired.empty:
        return pd.DataFrame(columns=HEAD_TO_HEAD_COLUMNS)
    timed = qualifying[qualifying["Round"] <= after_round].set_index(["Round", "Driver"])
    scored = results[results["Round"] <= after_round].groupby(["Round", "Driver"])["Points"].sum()
    for side in ("", "Mate"):
        rows = pd.MultiIndex.from_arrays([paired["Round"], paired["Driver" + side]])
        paired["Grid" + side] = timed["Position"].reindex(rows).to_numpy()
        for part in PARTS:
            paired[part + side] = timed[part].reindex(rows).to_numpy()
        paired["Scored" + side] = scored.reindex(rows).fillna(0).to_numpy()
    # The first part from Q3 down where both have a time.
    gaps = pd.DataFrame({part: paired[part] - paired[part + "Mate"] for part in PARTS})
    finished = paired["Classified"] & paired["ClassifiedMate"]
    table = (
        paired.assign(
            QualiAhead=paired["Grid"] < paired["GridMate"],
            QualiBehind=paired["Grid"] > paired["GridMate"],
            RaceAhead=finished & (paired["Position"] < paired["PositionMate"]),
            RaceBehind=finished & (paired["Position"] > paired["PositionMate"]),
            Gap=gaps.bfill(axis=1).iloc[:, 0],
        )
        .groupby(["Team", "Driver", "DriverMate"], as_index=False)
        .agg(
            Rounds=("Round", "nunique"),
            QualiAhead=("QualiAhead", "sum"),
            QualiBehind=("QualiBehind", "sum"),
            RaceAhead=("RaceAhead", "sum"),
            RaceBehind=("RaceBehind", "sum"),
            Points=("Scored", "sum"),
            TeammatePoints=("ScoredMate", "sum"),
            Gap=("Gap", "median"),
        )
        .rename(columns={"DriverMate": "Teammate"})
    )
    behind = (table["TeammatePoints"] > table["Points"]) | (
        (table["TeammatePoints"] == table["Points"]) & (table["QualiBehind"] > table["QualiAhead"])
    )
    swapped = table.rename(columns=MIRRORED | {value: key for key, value in MIRRORED.items()})
    table.loc[behind] = swapped.loc[behind, table.columns]
    table.loc[behind, "Gap"] = -table.loc[behind, "Gap"]
    return table[HEAD_TO_HEAD_COLUMNS]
