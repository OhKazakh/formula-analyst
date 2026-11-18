"""Facts to read while a session loads, from the bundled race results.

Only Grands Prix before the one being loaded count, so a fact never gives its result away.
"""

from __future__ import annotations

import pandas as pd

FIRST_SEASON = 2018
PIT_LANE = 0
COMEBACK = 5  # grid slot from which a win counts as a comeback
MIN_RACES = 3  # fewer races here than this make a share meaningless


def times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def joined(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def from_grid(grid: int) -> str:
    if grid == 1:
        return "from pole"
    if grid == PIT_LANE:
        return "from the pit lane"
    return f"from P{grid}"


def earlier_winners(results: pd.DataFrame, schedule: pd.DataFrame, year: int) -> pd.DataFrame:
    """Every Grand Prix winner before `year`, with where the race was held."""
    races = results[(results["Session"] == "Race") & (results["Position"] == 1)]
    races = races[races["Year"] < year]
    return races.merge(schedule[["Year", "Round", "EventName", "Location"]], on=["Year", "Round"])


def venue_facts(results: pd.DataFrame, schedule: pd.DataFrame, year: int, event: str) -> list[str]:
    """What happened at this circuit in earlier seasons.

    Circuits are matched by location, so a renamed race still finds its history and a new
    venue under an old name doesn't.
    """
    location = schedule.loc[
        (schedule["Year"] == year) & (schedule["EventName"] == event), "Location"
    ]
    if location.empty:
        return []
    winners = earlier_winners(results, schedule, year)
    here = winners[winners["Location"] == location.iloc[0]].sort_values("Year")
    if here.empty:
        return []
    facts = []
    last = here.iloc[-1]
    facts.append(
        f"Last time here, in {last['Year']}, {last['Name']} won {from_grid(int(last['Grid']))}."
    )
    wins = here["Name"].value_counts()
    if wins.iloc[0] >= 2:
        leaders = sorted(wins[wins == wins.iloc[0]].index)
        if len(leaders) == 1:
            facts.append(
                f"{leaders[0]} has won here {times(wins.iloc[0])} since {FIRST_SEASON}, "
                "more than anyone else."
            )
        else:
            names = joined(leaders)
            facts.append(f"{names} have each won here {times(wins.iloc[0])} since {FIRST_SEASON}.")
    if len(here) >= MIN_RACES:
        poles = int((here["Grid"] == 1).sum())
        facts.append(f"Pole has won {poles} of the {len(here)} races here since {FIRST_SEASON}.")
        teams = here["Team"].value_counts()
        if teams.iloc[0] >= 2:
            facts.append(
                f"{teams.index[0]} has won {teams.iloc[0]} of the {len(here)} races here "
                f"since {FIRST_SEASON}."
            )
    # A pit-lane start is the furthest back of all.
    furthest = here.assign(Back=here["Grid"].replace(PIT_LANE, 99)).sort_values("Back").iloc[-1]
    if furthest["Back"] >= COMEBACK:
        facts.append(
            f"The biggest comeback here since {FIRST_SEASON}: {furthest['Name']} won "
            f"{from_grid(int(furthest['Grid']))} in {furthest['Year']}."
        )
    return facts


def season_facts(results: pd.DataFrame, schedule: pd.DataFrame, year: int) -> list[str]:
    """Records across every Grand Prix before `year`."""
    winners = earlier_winners(results, schedule, year)
    if len(winners) < MIN_RACES:
        return []
    span = f"From {FIRST_SEASON} to {year - 1}"
    wins = winners["Name"].value_counts()
    leaders = sorted(wins[wins == wins.iloc[0]].index)
    most = (
        f"{span}, {leaders[0]} won {wins.iloc[0]} Grands Prix, more than anyone else."
        if len(leaders) == 1
        else f"{span}, {joined(leaders)} won {wins.iloc[0]} Grands Prix each, the most of anyone."
    )
    poles = (winners["Grid"] == 1).mean()
    furthest = winners.assign(Back=winners["Grid"].replace(PIT_LANE, 99))
    furthest = furthest.sort_values(["Back", "Year"]).iloc[-1]
    return [
        most,
        f"{span}, pole turned into a win in {poles:.0%} of Grands Prix.",
        f"{span}, the furthest back anyone won from was "
        f"{from_grid(int(furthest['Grid'])).removeprefix('from ')}: {furthest['Name']} at the "
        f"{furthest['Year']} {furthest['EventName']}.",
    ]
