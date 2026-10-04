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
LABEL = "#FFFFFF"
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


def _svg(body: str, width: float) -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{SIZE}" '
        f'viewBox="0 0 {width:g} {SIZE}">{body}</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def badges(compounds: list[str] | tuple[str, ...], colors: dict[str, str]) -> str:
    width = SIZE * len(compounds) + GAP * (len(compounds) - 1)
    body = "".join(
        _badge(compound, colors.get(compound, FALLBACK_COLOR), SIZE / 2 + index * (SIZE + GAP))
        for index, compound in enumerate(compounds)
    )
    return _svg(body, width)


def badge(compound: str, colors: dict[str, str]) -> str:
    return badges([compound], colors)


# A dark capsule keeps the lap count readable on both light and dark tables.
def _capsule(compound: str, laps: int, colors: dict[str, str], x: float) -> tuple[str, float]:
    label = str(laps)
    width = SIZE + 10 + 7.5 * len(label)
    body = (
        f'<rect x="{x:g}" width="{width:g}" height="{SIZE}" rx="{SIZE / 2:g}" fill="{RUBBER}"/>'
        + _badge(compound, colors.get(compound, FALLBACK_COLOR), x + SIZE / 2)
        + f'<text x="{x + SIZE + 3:g}" y="19" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="13" font-weight="700" fill="{LABEL}">{label}</text>'
    )
    return body, width


# A fixed width keeps every row's history left-aligned in a table; long ones shrink to fit.
def stints(
    history: list[tuple[str, int]], colors: dict[str, str], width: float | None = None
) -> str:
    parts, x = [], 0.0
    for compound, laps in history:
        part, capsule = _capsule(compound, laps, colors, x)
        parts.append(part)
        x += capsule + GAP
    content = max(x - GAP, 1)
    if width is None:
        return _svg("".join(parts), content)
    scale = min(1.0, width / content)
    offset = SIZE * (1 - scale) / 2
    body = f'<g transform="translate(0 {offset:g}) scale({scale:g})">{"".join(parts)}</g>'
    return _svg(body, width)
