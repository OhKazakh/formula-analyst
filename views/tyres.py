import base64

from src.charts import FALLBACK_COLOR

LETTERS = {
    "SOFT": "S",
    "MEDIUM": "M",
    "HARD": "H",
    "INTERMEDIATE": "I",
    "WET": "W",
    "HYPERSOFT": "HS",
    "ULTRASOFT": "US",
    "SUPERSOFT": "SS",
    "SUPERHARD": "SH",
}
RUBBER = "#15151E"
SIZE = 28
GAP = 6


def _badge(compound: str, color: str, centre: float) -> str:
    letter = LETTERS.get(str(compound), "?")
    font = 12 if len(letter) == 1 else 9
    return (
        f'<circle cx="{centre}" cy="14" r="12" fill="{RUBBER}" stroke="{color}" stroke-width="3"/>'
        f'<text x="{centre}" y="{18 if font == 12 else 17}" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{font}" font-weight="700" '
        f'fill="{color}">{letter}</text>'
    )


def badges(compounds: list[str] | tuple[str, ...], colors: dict[str, str]) -> str:
    width = SIZE * len(compounds) + GAP * (len(compounds) - 1)
    body = "".join(
        _badge(compound, colors.get(compound, FALLBACK_COLOR), SIZE / 2 + index * (SIZE + GAP))
        for index, compound in enumerate(compounds)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{SIZE}" '
        f'viewBox="0 0 {width} {SIZE}">{body}</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def badge(compound: str, colors: dict[str, str]) -> str:
    return badges([compound], colors)
