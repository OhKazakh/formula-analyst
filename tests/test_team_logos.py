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


def test_fetch_falls_back_to_the_next_season(tmp_path, monkeypatch):
    def download(url):
        return Image.new("RGBA", (206, 206)) if "/2019/" in url and "toro-rosso" in url else None

    monkeypatch.setattr(team_logos, "_download", download)

    saved = team_logos.fetch_logos(2018, ["Toro Rosso", "Force India"], tmp_path)

    path, style = team_logos.saved_logo(2018, "Toro Rosso", tmp_path)
    assert saved == ["tororosso"]
    assert style == "tile"
    assert Image.open(path).size == (team_logos.SIZE, team_logos.SIZE)
    assert team_logos.saved_logo(2018, "Force India", tmp_path) is None
