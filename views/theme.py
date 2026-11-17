import base64
import html

import pandas as pd
from pandas.io.formats.style import Styler

SURFACE = "light-dark(#F7F4F1, #1C1C25)"
SECONDARY_TEXT = "light-dark(#606066, #AAAAAA)"

# Tabs read as pill buttons, the selected one filled with the formula1.com red: white on that
# red is 4.8:1, where red text on carbon was too faint. Slider values keep the text colour.
GLOBAL_STYLE = """<style>
[data-testid="stSliderThumbValue"] { color: inherit; }
[data-testid="stTabs"] [role="tablist"] { gap: 8px; flex-wrap: wrap; padding-bottom: 6px; }
[data-testid="stTabs"] [role="tablist"]::after,
[data-testid="stTab"] .react-aria-SelectionIndicator { display: none; }
[data-testid="stTab"] {
  height: 34px; padding: 0 16px; border-radius: 999px; outline: none;
  border: 1px solid light-dark(#E0DEDC, #47464C);
  background: light-dark(#F7F4F1, #1C1C25); color: light-dark(#606066, #AAAAAA);
  transition: background-color 0.15s, border-color 0.15s, color 0.15s;
}
[data-testid="stTab"] p { font-weight: 600; }
[data-testid="stTab"]:hover { color: inherit; border-color: light-dark(#909094, #8A8A90); }
[data-testid="stTab"][aria-selected="true"] {
  background: #E10600; border-color: #E10600; color: #FFFFFF;
}
[data-testid="stTab"][data-focus-visible] { outline: 2px solid #E10600; outline-offset: 2px; }
</style>"""

# formula1.com's sector purple and positive green, used for the fastest and personal-best laps.
FASTEST = "background-color: #5300A6; color: #FFFFFF"
PERSONAL_BEST = "background-color: #28973E; color: #15151E"
FAVOURITE_ROW = "background-color: rgba(225, 6, 0, 0.12)"


def pill(text: str, fill: str, ink: str, width: int) -> str:
    """A rounded label as an image, for table columns that can only hold images or text."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="22" '
        f'viewBox="0 0 {width} 22"><rect width="{width}" height="22" rx="11" '
        f'fill="{fill}"/><text x="{width / 2:g}" y="15" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="700" '
        f'fill="{ink}">{html.escape(text)}</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def favourite_rows(table: pd.DataFrame, marked: pd.Series) -> Styler:
    rows = marked.to_numpy()
    return table.style.apply(
        lambda frame: pd.DataFrame(
            [[FAVOURITE_ROW if mark else ""] * frame.shape[1] for mark in rows],
            index=frame.index,
            columns=frame.columns,
        ),
        axis=None,
    )
