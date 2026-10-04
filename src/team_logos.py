from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from src.races import DATA_DIR

LOGO_DIR = "team-logos"
SIZE = 96
STYLES = ("mark", "tile")
# formula1.com has white marks for recent seasons and colour logos on a white tile before that.
MARK_URL = (
    "https://media.formula1.com/image/upload/c_lfill,w_{size}/f_png/q_auto/v1740000001/"
    "common/f1/{year}/{slug}/{year}{slug}logowhite.webp"
)
TILE_URL = (
    "https://media.formula1.com/content/dam/fom-website/teams/{year}/{slug}-logo.png"
    ".transform/2col/image.png"
)


@dataclass(frozen=True)
class Team:
    key: str
    names: tuple[str, ...]
    marks: tuple[str, ...] = ()
    tiles: tuple[str, ...] = ()


TEAMS = (
    Team("mercedes", ("mercedes",), ("mercedes",), ("mercedes",)),
    Team("ferrari", ("ferrari",), ("ferrari",), ("ferrari",)),
    Team("redbull", ("red bull",), ("redbullracing",), ("red-bull-racing",)),
    Team("mclaren", ("mclaren",), ("mclaren",), ("mclaren",)),
    Team("astonmartin", ("aston martin",), ("astonmartin",), ("aston-martin",)),
    Team("alpine", ("alpine",), ("alpine",), ("alpine",)),
    Team("williams", ("williams",), ("williams",), ("williams",)),
    Team("racingbulls", ("racing bulls", "rb f1"), ("racingbulls", "rb")),
    Team("alphatauri", ("alphatauri",), tiles=("alphatauri",)),
    Team("tororosso", ("toro rosso",), tiles=("toro-rosso",)),
    Team("haas", ("haas",), ("haasf1team", "haas"), ("haas-f1-team",)),
    Team("alfaromeo", ("alfa romeo",), tiles=("alfa-romeo-racing", "alfa-romeo")),
    Team("sauber", ("sauber",), ("kicksauber",)),
    Team("racingpoint", ("racing point",), tiles=("racing-point",)),
    Team("forceindia", ("force india",), tiles=("force-india",)),
    Team("renault", ("renault",), tiles=("renault",)),
    Team("audi", ("audi",), ("audi",)),
    Team("cadillac", ("cadillac",), ("cadillac",)),
)


def team_for(name: str) -> Team | None:
    lowered = name.strip().lower()
    if lowered == "rb":
        return next(team for team in TEAMS if team.key == "racingbulls")
    return next((team for team in TEAMS if any(part in lowered for part in team.names)), None)


def logo_dir(root: Path = DATA_DIR) -> Path:
    return root / LOGO_DIR


def saved_logo(year: int, name: str, root: Path = DATA_DIR) -> tuple[Path, str] | None:
    team = team_for(name)
    if team is None:
        return None
    for style in STYLES:
        path = logo_dir(root) / str(year) / f"{team.key}.{style}.png"
        if path.exists():
            return path, style
    return None


def _download(url: str) -> Image.Image | None:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    except requests.RequestException:
        return None
    if response.status_code != 200 or not response.content:
        return None
    return Image.open(io.BytesIO(response.content)).convert("RGBA")


def _find(team: Team, year: int) -> tuple[Image.Image, str] | None:
    for slug in team.marks:
        if image := _download(MARK_URL.format(size=SIZE, year=year, slug=slug)):
            return image, "mark"
    for slug in team.tiles:
        if image := _download(TILE_URL.format(year=year, slug=slug)):
            return image, "tile"
    return None


# The 2018 logos aren't online any more; the next season's are used where the team kept its name.
def fetch_logos(year: int, names: list[str], root: Path = DATA_DIR) -> list[str]:
    folder = logo_dir(root) / str(year)
    saved = []
    for team in {team for name in names if (team := team_for(name))}:
        if any((folder / f"{team.key}.{style}.png").exists() for style in STYLES):
            continue
        found = _find(team, year) or _find(team, year + 1)
        if found is None:
            continue
        image, style = found
        image.thumbnail((SIZE, SIZE))
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / f"{team.key}.{style}.png", optimize=True)
        saved.append(team.key)
    return sorted(saved)
