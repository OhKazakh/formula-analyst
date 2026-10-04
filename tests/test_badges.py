import base64

import pytest

from views import logos, tyres


def svg(uri: str) -> str:
    return base64.b64decode(uri.split(",", 1)[1]).decode()


@pytest.mark.parametrize(
    ("team", "letters"),
    [
        ("Mercedes", "MER"),
        ("RB", "RB"),
        ("RB F1 Team", "RB"),
        ("Kick Sauber", "SAU"),
        ("Lotus", "LOT"),
    ],
)
def test_teams_without_a_mark_get_initials(team, letters):
    assert logos.initials(team) == letters
    assert f">{letters}</text>" in logos.badge_svg(team, "#123456")


def test_badges_use_the_seasons_official_logo():
    mark = logos.badge_svg("Mercedes", "#27F4D2", 2026)
    tile = logos.badge_svg("Racing Point", "#F596C8", 2020)

    assert "<image" in mark and 'filter="url(#ink)"' in mark
    assert "<image" in tile and "<rect" not in tile
    assert ">FI</text>" in logos.badge_svg("Force India", "#F596C8", 2018)


def test_badges_pick_readable_ink_and_outline_pale_colours():
    assert logos.ink("#15151E") == logos.LIGHT_INK
    assert logos.ink("#FFF500") == logos.DARK_INK
    assert "stroke=" in logos.badge_svg("Williams", "#FFFFFF")
    assert "stroke=" not in logos.badge_svg("Williams", "#1868DB")


def test_tyre_history_is_drawn_at_a_fixed_width():
    history = [("MEDIUM", 15), ("HARD", 38)]
    colors = {"MEDIUM": "#FFD12E", "HARD": "#F0F0EC"}

    natural = svg(tyres.stints(history, colors))
    fixed = svg(tyres.stints(history * 4, colors, width=120))

    assert 'width="112"' in natural
    assert 'width="120"' in fixed
    assert "scale(" in fixed
