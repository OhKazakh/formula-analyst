import pytest
from PIL import Image

from src import team_logos


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Red Bull Racing", "redbull"),
        ("Red Bull", "redbull"),
        ("RB", "racingbulls"),
        ("RB F1 Team", "racingbulls"),
        ("Racing Bulls", "racingbulls"),
        ("Kick Sauber", "sauber"),
        ("Alfa Romeo Racing", "alfaromeo"),
        ("Haas F1 Team", "haas"),
    ],
)
def test_team_names_from_both_sources_match_one_team(name, key):
    assert team_logos.team_for(name).key == key


def logo(card: str, ink: str) -> Image.Image:
    """A colour logo as formula1.com had them: a mark on a card, with transparent corners."""
    image = Image.new("RGBA", (40, 40))
    image.paste(Image.new("RGBA", (36, 36), card), (2, 2))
    image.paste(Image.new("RGBA", (12, 12), ink), (14, 14))
    return image


def test_fetch_falls_back_to_the_next_season(tmp_path, monkeypatch):
    def download(url):
        found = "/2019/" in url and "toro-rosso" in url
        return logo("white", "navy").resize((206, 206)) if found else None

    monkeypatch.setattr(team_logos, "_download", download)

    saved = team_logos.fetch_logos(2018, ["Toro Rosso", "Force India"], tmp_path)

    path = team_logos.saved_logo(2018, "Toro Rosso", tmp_path)
    assert saved == ["tororosso"]
    assert path.name == "tororosso.mark.png"
    # The colour logo became a white mark that fills the frame, its card gone.
    saved_mark = Image.open(path)
    assert saved_mark.size == (team_logos.SIZE, team_logos.SIZE)
    assert saved_mark.getchannel("A").getbbox() == (0, 0, team_logos.SIZE, team_logos.SIZE)
    assert team_logos.saved_logo(2018, "Force India", tmp_path) is None


@pytest.mark.parametrize(
    ("card", "ink"),
    [("white", "#E10600"), ("white", "#15151E"), ("#00594F", "white")],
)
def test_colour_logos_become_white_marks(card, ink):
    mark = team_logos.white_mark(logo(card, ink))

    # The mark is solid white, and the card and the corners around it disappear.
    assert mark.getpixel((20, 20)) == (255, 255, 255, 255)
    assert mark.getpixel((5, 5))[3] == 0
    assert mark.getpixel((0, 0))[3] == 0
