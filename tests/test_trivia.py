import pandas as pd

from src import trivia
from views import loading


def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2022, 2023, 2024, 2025, 2025, 2026, 2026],
            "Round": [1, 1, 1, 1, 2, 1, 2],
            "EventName": [
                "Spanish GP",
                "Spanish GP",
                "Spanish GP",
                "Spanish GP",
                "Other GP",
                "Barcelona GP",
                "Spanish GP",
            ],
            "Location": [
                "Barcelona",
                "Barcelona",
                "Barcelona",
                "Barcelona",
                "Elsewhere",
                "Barcelona",
                "Madrid",
            ],
        }
    )


def results() -> pd.DataFrame:
    winners = [
        (2022, 1, "Ann Able", "Red", 1),
        (2023, 1, "Ann Able", "Red", 6),
        (2024, 1, "Bob Best", "Blue", 1),
        (2025, 1, "Cat Cole", "Red", 2),
        (2025, 2, "Eve East", "Blue", 0),
        # The race being loaded: no fact may give it away.
        (2026, 1, "Dan Dale", "Green", 20),
    ]
    return pd.DataFrame(
        [
            {
                "Year": year,
                "Round": round_number,
                "Session": "Race",
                "Name": name,
                "Team": team,
                "Position": 1,
                "Grid": grid,
            }
            for year, round_number, name, team, grid in winners
        ]
    )


def test_a_renamed_race_keeps_its_history():
    facts = trivia.venue_facts(results(), schedule(), 2026, "Barcelona GP")

    assert facts == [
        "Last time here, in 2025, Cat Cole won from P2.",
        "Ann Able has won here twice since 2018, more than anyone else.",
        "Pole has won 2 of the 4 races here since 2018.",
        "Red has won 3 of the 4 races here since 2018.",
        "The biggest comeback here since 2018: Ann Able won from P6 in 2023.",
    ]
    assert not any("Dan Dale" in fact for fact in facts)


def test_a_new_venue_under_an_old_name_has_no_history():
    assert trivia.venue_facts(results(), schedule(), 2026, "Spanish GP") == []


def test_season_facts_only_look_back():
    facts = trivia.season_facts(results(), schedule(), 2026)

    assert facts[0] == "From 2018 to 2025, Ann Able won 2 Grands Prix, more than anyone else."
    assert facts[1] == "From 2018 to 2025, pole turned into a win in 40% of Grands Prix."
    # A pit-lane start is the furthest back, further than P6.
    assert facts[2].endswith("the pit lane: Eve East at the 2025 Other GP.")


def test_a_tie_for_most_wins_names_everyone_in_it():
    tied = results().assign(Name=lambda rows: rows["Name"].replace("Cat Cole", "Bob Best"))

    facts = trivia.season_facts(tied, schedule(), 2026)

    assert facts[0] == (
        "From 2018 to 2025, Ann Able and Bob Best won 2 Grands Prix each, the most of anyone."
    )


def test_the_loader_escapes_its_text_and_lights_five_lamps():
    page = loading.loader_html("Loading <the> race", "Ann & Bob")

    assert "Loading &lt;the&gt; race" in page and "Ann &amp; Bob" in page
    assert page.count("<span></span>") == loading.LIGHTS
    assert "prefers-reduced-motion" in page
