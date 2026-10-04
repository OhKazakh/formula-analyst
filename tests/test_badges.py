import pytest

from views import logos


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


def test_teams_with_a_mark_use_its_path():
    badge = logos.badge_svg("Red Bull Racing", "#3671C6")

    assert "<path" in badge
    assert "<text" not in badge


def test_badges_pick_readable_ink_and_outline_pale_colours():
    assert logos.ink("#15151E") == logos.LIGHT_INK
    assert logos.ink("#FFF500") == logos.DARK_INK
    assert "stroke=" in logos.badge_svg("Williams", "#FFFFFF")
    assert "stroke=" not in logos.badge_svg("Williams", "#1868DB")
