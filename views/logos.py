import base64
from functools import cache

from src import team_logos
from src.charts import FALLBACK_COLOR, OUTLINE, contrast, hard_to_see

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
# Turns a white mark carbon, for team colours too light for white.
DARK_MARK = (
    '<filter id="ink"><feColorMatrix type="matrix" '
    'values="0 0 0 0 0.082 0 0 0 0 0.082 0 0 0 0 0.118 0 0 0 1 0"/></filter>'
)


def initials(team: str) -> str:
    if team.strip().upper() == "RB":
        return "RB"
    name = team.lower()
    letters = next((value for key, value in INITIALS.items() if key in name), None)
    return letters or team.strip()[:3].upper()


def ink(color: str) -> str:
    return LIGHT_INK if contrast(color, LIGHT_INK) >= 3 else DARK_INK


@cache
def _logo(year: int, team: str) -> tuple[str, str] | None:
    found = team_logos.saved_logo(year, team)
    if found is None:
        return None
    path, style = found
    return base64.b64encode(path.read_bytes()).decode(), style


def _image(data: str, size: float, offset: float = 0, extra: str = "") -> str:
    return (
        f'<image href="data:image/png;base64,{data}" x="{offset:g}" y="{offset:g}" '
        f'width="{size:g}" height="{size:g}"{extra}/>'
    )


def badge_svg(team: str, color: str | None, year: int | None = None) -> str:
    fill = color if color and color.startswith("#") else FALLBACK_COLOR
    foreground = ink(fill)
    outline = f' stroke="{OUTLINE}" stroke-width="1.5"' if hard_to_see(fill) else ""
    square = (
        f'<rect x="0.75" y="0.75" width="{SIZE - 1.5}" height="{SIZE - 1.5}" rx="6" '
        f'fill="{fill}"{outline}/>'
    )
    logo = _logo(year, team) if year else None
    if logo and logo[1] == "tile":
        body = _image(logo[0], SIZE)
    elif logo:
        dark = foreground == DARK_INK
        body = (
            (DARK_MARK if dark else "")
            + square
            + _image(logo[0], SIZE - 6, 3, ' filter="url(#ink)"' if dark else "")
        )
    else:
        letters = initials(team)
        font = 11 if len(letters) <= 2 else 9
        body = square + (
            f'<text x="14" y="{14 + font * 0.36:.1f}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{font}" font-weight="700" '
            f'fill="{foreground}">{letters}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
        f'viewBox="0 0 {SIZE} {SIZE}">{body}</svg>'
    )


def badge(team: str, color: str | None, year: int | None = None) -> str:
    svg = badge_svg(team, color, year)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
