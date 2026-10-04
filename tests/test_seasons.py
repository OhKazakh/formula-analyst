import json
from types import SimpleNamespace

import pandas as pd

from src import seasons


class FakeResponse:
    def __init__(self, pages: list[list[tuple[int, list[str]]]], index: int = 0):
        self.pages = pages
        self.index = index
        page = pages[index]
        self.description = pd.DataFrame({"round": [round_number for round_number, _ in page]})
        self.content = [
            pd.DataFrame(
                {
                    "driverCode": codes,
                    "givenName": "Given",
                    "familyName": codes,
                    "constructorName": "Team",
                    "position": range(1, len(codes) + 1),
                    "positionText": ["1", "R"][: len(codes)],
                    "points": [25.0, 0.0][: len(codes)],
                    "grid": [2, 1][: len(codes)],
                    "status": ["Finished", "Engine"][: len(codes)],
                }
            )
            for _, codes in page
        ]

    def get_next_result_page(self):
        if self.index + 1 >= len(self.pages):
            raise ValueError("No more data after this response.")
        return FakeResponse(self.pages, self.index + 1)


def test_results_follow_every_page():
    pages = [[(1, ["AAA", "BBB"])], [(1, ["CCC"]), (2, ["AAA"])]]

    results = seasons._results(lambda **_: FakeResponse(pages), 2024, "Race")

    assert results["Round"].tolist() == [1, 1, 1, 2]
    assert results["Driver"].tolist() == ["AAA", "BBB", "CCC", "AAA"]
    assert results["Classified"].tolist() == [True, False, True, True]
    assert results["Status"].tolist() == ["Finished", "Engine", "Finished", "Finished"]
    assert (results["Session"] == "Race").all()


class FakeErgast:
    def get_constructor_standings(self, season: int, round: int):
        standings = pd.DataFrame(
            {
                "position": [1, 2],
                "points": [25.0 * round, 18.0 * round],
                "wins": [round, 0],
                "constructorName": ["Team A", "Team B"],
            }
        )
        return SimpleNamespace(content=[standings] if round < 3 else [])


def test_team_standings_keep_every_round_that_has_standings():
    standings = seasons._team_standings(FakeErgast(), 2024, [1, 2, 3])

    assert standings["Round"].tolist() == [1, 1, 2, 2]
    assert standings.loc[standings["Round"] == 2, "Points"].tolist() == [50.0, 36.0]
    assert standings["Position"].tolist() == [1, 2, 1, 2]


def test_save_and_load_season_with_styles_from_races(tmp_path):
    season = seasons.Season(
        year=2024,
        schedule=pd.DataFrame(
            {"Round": [1, 2], "EventName": ["A", "B"], "Sprint": [False, True], "Date": pd.NaT}
        ),
        results=pd.DataFrame(
            [
                {
                    "Round": 1,
                    "Session": "Race",
                    "Driver": "AAA",
                    "Name": "A A",
                    "Team": "T",
                    "Position": 1,
                    "Classified": True,
                    "Points": 25.0,
                    "Grid": 3,
                    "Status": "Finished",
                }
            ]
        ),
        team_standings=pd.DataFrame(
            {"Round": [1], "Team": ["T"], "Position": [1], "Points": [25.0], "Wins": [1]}
        ),
    )
    seasons.save_season(season, tmp_path)
    for name, color in [("01-a", "#111111"), ("02-b", "#222222")]:
        (tmp_path / "2024" / name).mkdir()
        meta = {"driver_styles": {"AAA": {"color": color, "linestyle": "solid"}}}
        (tmp_path / "2024" / name / "race.json").write_text(json.dumps(meta))

    loaded = seasons.load_season(2024, tmp_path)

    assert seasons.saved_seasons(tmp_path) == [2024]
    pd.testing.assert_frame_equal(loaded.results, season.results)
    pd.testing.assert_frame_equal(loaded.team_standings, season.team_standings)
    assert loaded.completed_rounds == [1]
    assert loaded.event_name(2) == "B"
    assert loaded.driver_styles["AAA"]["color"] == "#222222"
