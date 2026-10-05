from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis import GREEN_FLAG, RED_FLAG, linear_fit
from src.charts import OUTLINE, Styles, driver_style, finish

PARTS = (1, 2, 3)
LONG_RUN_LAPS = 5
LONG_RUN_LIMIT = 1.12
RUN_SPREAD = 0.025
RESULT_COLUMNS = ["Position", "Driver", "Team", "Best", "Gap", "Laps", "Compound"]
LONG_RUN_COLUMNS = [
    "Driver",
    "Team",
    "Stint",
    "Compound",
    "Start",
    "Laps",
    "Average",
    "Best",
    "Degradation",
]


def part_labels(session: str) -> list[str]:
    prefix = "SQ" if session.startswith("Sprint") else "Q"
    return [f"{prefix}{part}" for part in PARTS]


def _teams(laps: pd.DataFrame) -> pd.Series:
    return laps.drop_duplicates("Driver", keep="last").set_index("Driver")["Team"]


def qualifying_results(laps: pd.DataFrame, order: list[str], session: str) -> pd.DataFrame:
    labels = part_labels(session)
    valid = laps[laps["LapTimeSeconds"].notna() & ~laps["Deleted"]]
    best = valid.pivot_table(
        index="Driver", columns="Part", values="LapTimeSeconds", aggfunc="min"
    ).reindex(columns=PARTS)
    drivers = [driver for driver in order if driver in set(laps["Driver"])]
    table = pd.DataFrame(
        {
            "Position": range(1, len(drivers) + 1),
            "Driver": drivers,
            "Team": _teams(laps).reindex(drivers).to_numpy(),
        }
    )
    for part, label in zip(PARTS, labels, strict=True):
        table[label] = table["Driver"].map(best[part])
    # A driver is classified on their time from the last part they reached.
    reached = table[labels[::-1]].bfill(axis=1).iloc[:, 0]
    table["Best"] = reached
    table["Gap"] = reached - reached.iloc[0] if len(table) else reached
    table["Out"] = [
        next((label for label in labels[::-1] if pd.notna(row[label])), "")
        for _, row in table.iterrows()
    ]
    return table


def practice_results(laps: pd.DataFrame) -> pd.DataFrame:
    bests = laps[laps["IsPersonalBest"]].dropna(subset=["LapTimeSeconds"])
    if bests.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    best = bests.loc[bests.groupby("Driver")["LapTimeSeconds"].idxmin()]
    best = best.sort_values("LapTimeSeconds").reset_index(drop=True)
    counts = laps.groupby("Driver")["LapNumber"].count()
    return pd.DataFrame(
        {
            "Position": range(1, len(best) + 1),
            "Driver": best["Driver"],
            "Team": best["Team"],
            "Best": best["LapTimeSeconds"],
            "Gap": best["LapTimeSeconds"] - best["LapTimeSeconds"].iloc[0],
            "Laps": best["Driver"].map(counts).astype(int),
            "Compound": best["Compound"],
        }
    )


# Race simulations: unbroken laps of one stint that stay close to that stint's typical pace.
# Heavy-fuel runs are well off the session's fastest lap, so the session limit only drops
# cool-down laps. FastF1 marks many practice laps inaccurate, so that flag isn't used here.
def long_run_laps(laps: pd.DataFrame, minimum: int = LONG_RUN_LAPS) -> pd.DataFrame:
    clean = laps[
        laps["PitInTime"].isna() & laps["PitOutTime"].isna() & (laps["TrackStatus"] == GREEN_FLAG)
    ].dropna(subset=["LapTimeSeconds", "Stint"])
    if clean.empty:
        return clean.assign(Run=pd.Series(dtype=int))
    clean = clean[clean["LapTimeSeconds"] <= clean["LapTimeSeconds"].min() * LONG_RUN_LIMIT]
    runs, run = [], 0
    for _, stint in clean.sort_values(["Driver", "Stint", "LapNumber"]).groupby(
        ["Driver", "Stint"]
    ):
        typical = stint["LapTimeSeconds"].median()
        steady = stint[(stint["LapTimeSeconds"] - typical).abs() <= typical * RUN_SPREAD]
        breaks = np.flatnonzero(np.diff(steady["LapNumber"].to_numpy()) != 1) + 1
        for piece in np.split(np.arange(len(steady)), breaks):
            if len(piece) >= minimum:
                run += 1
                runs.append(steady.iloc[piece].assign(Run=run))
    if not runs:
        return clean.iloc[0:0].assign(Run=pd.Series(dtype=int))
    return pd.concat(runs, ignore_index=True)


def long_runs(laps: pd.DataFrame, minimum: int = LONG_RUN_LAPS) -> pd.DataFrame:
    rows = []
    for _, run in long_run_laps(laps, minimum).groupby("Run"):
        fit = linear_fit(run["LapNumber"], run["LapTimeSeconds"])
        rows.append(
            {
                "Driver": run["Driver"].iloc[0],
                "Team": run["Team"].iloc[0],
                "Stint": int(run["Stint"].iloc[0]),
                "Compound": run["Compound"].iloc[0],
                "Start": int(run["LapNumber"].iloc[0]),
                "Laps": len(run),
                "Average": float(run["LapTimeSeconds"].mean()),
                "Best": float(run["LapTimeSeconds"].min()),
                "Degradation": fit.slope,
            }
        )
    table = pd.DataFrame(rows, columns=LONG_RUN_COLUMNS)
    return table.sort_values("Average", ignore_index=True)


def red_flags(laps: pd.DataFrame) -> bool:
    return bool(laps["TrackStatus"].fillna("").astype(str).str.contains(RED_FLAG).any())


def gap_figure(
    table: pd.DataFrame,
    styles: Styles,
    cutoffs: tuple[int, ...] = (),
    highlight: str | None = None,
    axis: str = "Gap to the fastest (s)",
) -> go.Figure:
    timed = table.dropna(subset=["Gap"])
    colors = [driver_style(styles, driver)[0] for driver in timed["Driver"]]
    widths = [2.5 if driver == highlight else 1 for driver in timed["Driver"]]
    figure = go.Figure(
        go.Bar(
            x=timed["Gap"],
            y=timed["Driver"],
            orientation="h",
            marker={"color": colors, "line": {"color": OUTLINE, "width": widths}},
            text=[f"+{gap:.3f}" if gap else "" for gap in timed["Gap"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>+%{x:.3f} s<extra></extra>",
        )
    )
    for cutoff in cutoffs:
        if cutoff < len(timed):
            figure.add_hline(y=cutoff - 0.5, line={"color": OUTLINE, "width": 1, "dash": "dot"})
    figure.update_yaxes(autorange="reversed", title=None)
    figure.update_xaxes(title=axis, rangemode="tozero")
    return finish(figure, max(320, 24 * len(timed) + 80), showlegend=False)
