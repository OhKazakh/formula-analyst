import json

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
    assert (results["Session"] == "Race").all()


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
                }
            ]
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
    assert loaded.completed_rounds == [1]
    assert loaded.event_name(2) == "B"
    assert loaded.driver_styles["AAA"]["color"] == "#222222"
