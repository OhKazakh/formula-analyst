import pandas as pd
import pytest

from src import championship


def result(round_number, driver, position, points, session="Race", classified=True):
    return {
        "Round": round_number,
        "Session": session,
        "Driver": driver,
        "Name": f"Driver {driver}",
        "Team": "Team",
        "Position": position,
        "Classified": classified,
        "Points": points,
    }


@pytest.fixture
def results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            result(1, "AAA", 1, 25.0),
            result(1, "BBB", 2, 18.0),
            result(1, "CCC", 3, 15.0),
            result(2, "BBB", 1, 25.0),
            result(2, "AAA", 2, 18.0),
            result(2, "CCC", 3, 15.0),
            result(2, "AAA", 1, 8.0, session="Sprint"),
        ]
    )


def test_standings_sum_race_and_sprint_points(results):
    table = championship.standings(results, after_round=2)

    assert table["Driver"].tolist() == ["AAA", "BBB", "CCC"]
    assert table["Points"].tolist() == [51.0, 43.0, 30.0]
    assert table["Wins"].tolist() == [1, 1, 0]


def test_standings_after_an_earlier_round(results):
    table = championship.standings(results, after_round=1)

    assert table["Points"].tolist() == [25.0, 18.0, 15.0]


def test_countback_ignores_unclassified_finishes():
    results = pd.DataFrame(
        [
            result(1, "AAA", 14, 0.0),
            result(2, "AAA", 15, 0.0),
            result(1, "BBB", 14, 0.0),
            result(2, "BBB", 14, 0.0, classified=False),
        ]
    )

    assert championship.standings(results, after_round=2)["Driver"].tolist() == ["AAA", "BBB"]


@pytest.mark.parametrize(
    ("year", "sprint", "expected"),
    [(2018, False, 25), (2021, True, 29), (2024, True, 34), (2025, True, 33), (2025, False, 25)],
)
def test_max_round_points(year, sprint, expected):
    assert championship.max_round_points(year, sprint) == expected


def test_remaining_points_counts_rounds_after_the_cutoff():
    schedule = pd.DataFrame({"Round": [1, 2, 3], "Sprint": [False, True, False]})

    assert championship.remaining_points(schedule, 2025, after_round=1) == 33 + 25


def test_title_contenders_compare_maximum_with_leader(results):
    table = championship.standings(results, after_round=2)

    contenders = championship.title_contenders(table, remaining=10)

    assert contenders["CanWin"].tolist() == [True, True, False]
    assert contenders["Maximum"].tolist() == [61.0, 53.0, 40.0]


def test_points_heatmap_includes_totals(results):
    schedule = pd.DataFrame({"Round": [1, 2], "EventName": ["Alpha Grand Prix", "Beta Grand Prix"]})

    figure = championship.points_heatmap(results, schedule, ["AAA", "BBB", "CCC"], upto_round=2)

    rounds, totals = figure.data
    assert list(rounds.x) == ["Alpha", "Beta"]
    assert totals.z.ravel().tolist() == [51.0, 43.0, 30.0]
    assert "Sprint P1" in rounds.customdata[0][1]


def test_progression_figure_is_cumulative(results):
    schedule = pd.DataFrame({"Round": [1, 2], "EventName": ["Alpha Grand Prix", "Beta Grand Prix"]})
    styles = {"BBB": {"color": "#123456", "linestyle": "dashed"}}

    figure = championship.progression_figure(results, schedule, ["AAA", "BBB"], styles, 2)

    first, second = figure.data
    assert list(first.y) == [25.0, 51.0]
    assert (second.line.color, second.line.dash) == ("#123456", "dash")
