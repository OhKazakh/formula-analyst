import base64
from functools import cache
from pathlib import Path

from src.charts import FALLBACK_COLOR, OUTLINE, contrast, hard_to_see

MARKS_DIR = Path(__file__).parent / "marks"
# Marks from Simple Icons (CC0). Teams without a freely licensed mark get initials instead.
MARKS = {
    "aston martin": "astonmartin",
    "audi": "audi",
    "cadillac": "cadillac",
    "ferrari": "ferrari",
    "mclaren": "mclaren",
    "red bull": "redbull",
    "renault": "renault",
}
INITIALS = {
    "alfa romeo": "ALF",
    "alphatauri": "AT",
    "alpine": "ALP",
    "force india": "FI",
    "haas": "HAA",
    "mercedes": "MER",
    "racing bulls": "RB",
    "racing point": "RP",
    "rb f1": "RB",
    "sauber": "SAU",
    "toro rosso": "TR",
    "williams": "WIL",
}
LIGHT_INK = "#FFFFFF"
DARK_INK = "#15151E"
SIZE = 28


def _lookup(team: str, table: dict[str, str]) -> str | None:
    name = team.lower()
    return next((value for key, value in table.items() if key in name), None)


@cache
def _mark_path(slug: str) -> str:
    svg = (MARKS_DIR / f"{slug}.svg").read_text()
    start = svg.index(' d="') + 4
    return svg[start : svg.index('"', start)]


def initials(team: str) -> str:
    if team.strip().upper() == "RB":
        return "RB"
    return _lookup(team, INITIALS) or team.strip()[:3].upper()


def ink(color: str) -> str:
    return LIGHT_INK if contrast(color, LIGHT_INK) >= 3 else DARK_INK


def badge_svg(team: str, color: str | None) -> str:
    fill = color if color and color.startswith("#") else FALLBACK_COLOR
    foreground = ink(fill)
    if slug := _lookup(team, MARKS):
        glyph = (
            f'<path transform="translate(3.2 3.2) scale(0.9)" fill="{foreground}" '
            f'd="{_mark_path(slug)}"/>'
        )
    else:
        letters = initials(team)
        font = 11 if len(letters) <= 2 else 9
        glyph = (
            f'<text x="14" y="{14 + font * 0.36:.1f}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{font}" font-weight="700" '
            f'fill="{foreground}">{letters}</text>'
        )
    outline = f' stroke="{OUTLINE}" stroke-width="1.5"' if hard_to_see(fill) else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
        f'viewBox="0 0 {SIZE} {SIZE}"><rect x="0.75" y="0.75" width="{SIZE - 1.5}" '
        f'height="{SIZE - 1.5}" rx="6" fill="{fill}"{outline}/>{glyph}</svg>'
    )


def badge(team: str, color: str | None) -> str:
    svg = badge_svg(team, color)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
