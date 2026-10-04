import pandas as pd
from pandas.io.formats.style import Styler

SURFACE = "light-dark(#F7F4F1, #1C1C25)"
SECONDARY_TEXT = "light-dark(#606066, #AAAAAA)"

# Red text on carbon is too low contrast, so the selected tab and slider values keep the
# text colour and only the underline and track stay red.
GLOBAL_STYLE = """<style>
[data-testid="stTab"][aria-selected="true"],
[data-testid="stSliderThumbValue"] { color: inherit; }
</style>"""

# formula1.com's sector purple and positive green, used for the fastest and personal-best laps.
FASTEST = "background-color: #5300A6; color: #FFFFFF"
PERSONAL_BEST = "background-color: #28973E; color: #15151E"
FAVOURITE_ROW = "background-color: rgba(225, 6, 0, 0.12)"


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
