import numpy as np
import pandas as pd
import pytest

from src import teammates


def result(round_number, driver, position, points, session="Race", classified=True):
    return {
        "Round": round_number,
        "Session": session,
        "Driver": driver,
        "Team": "Team",
        "Position": position,
        "Classified": classified,
        "Points": points,
    }


def timed(round_number, driver, position, q1, q2=np.nan, q3=np.nan):
    return {
        "Round": round_number,
        "Driver": driver,
        "Position": position,
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
    }


@pytest.fixture
def results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            result(1, "AAA", 1, 25.0),
            result(1, "BBB", 3, 15.0),
            result(2, "BBB", 2, 18.0),
            result(2, "AAA", 4, 12.0),
            result(2, "BBB", 1, 8.0, session="Sprint"),
            result(3, "AAA", 2, 18.0),
            result(3, "BBB", 20, 0.0, classified=False),
            # CCC stands in for BBB at the last round.
            result(4, "CCC", 5, 10.0),
            result(4, "AAA", 6, 8.0),
        ]
    )


@pytest.fixture
def qualifying() -> pd.DataFrame:
    return pd.DataFrame(
        [
            timed(1, "AAA", 1, 81.0, 80.5, 80.0),
            timed(1, "BBB", 2, 81.0, 80.5, 80.2),
            # AAA went out in Q2, so the two are compared on their Q2 times.
            timed(2, "BBB", 3, 81.0, 80.0, 80.1),
            timed(2, "AAA", 12, 81.0, 79.9),
            # AAA set no time in round 3.
            timed(3, "BBB", 9, 81.0, 80.0, 80.3),
        ]
    )


def test_head_to_head_counts_qualifying_races_and_points(results, qualifying):
    table = teammates.head_to_head(results, qualifying, after_round=3)

    row = table.iloc[0]
    assert len(table) == 1
    assert (row["Driver"], row["Teammate"], row["Rounds"]) == ("AAA", "BBB", 3)
    assert (row["QualiAhead"], row["QualiBehind"]) == (1, 1)
    # BBB didn't finish round 3, so only the first two races count.
    assert (row["RaceAhead"], row["RaceBehind"]) == (1, 1)
    # BBB's sprint points count too.
    assert (row["Points"], row["TeammatePoints"]) == (55.0, 41.0)
    assert row["Gap"] == pytest.approx(-0.15)


def test_the_higher_scorer_comes_first(results, qualifying):
    after_two = teammates.head_to_head(results, qualifying, after_round=2).iloc[0]

    assert (after_two["Driver"], after_two["Teammate"]) == ("BBB", "AAA")
    assert (after_two["Points"], after_two["TeammatePoints"]) == (41.0, 37.0)
    assert (after_two["RaceAhead"], after_two["RaceBehind"]) == (1, 1)
    # AAA was quicker on both days, so the gap is positive from BBB's side.
    assert after_two["Gap"] == pytest.approx(0.15)


def test_a_stand_in_is_a_pairing_of_its_own(results, qualifying):
    table = teammates.head_to_head(results, qualifying, after_round=4)

    rounds = table.set_index(["Driver", "Teammate"])["Rounds"].to_dict()
    # CCC outscored AAA in the one race they shared.
    assert rounds == {("AAA", "BBB"): 3, ("CCC", "AAA"): 1}
    # Neither has a qualifying time for that round.
    assert table["Gap"].isna().tolist() == [False, True]
