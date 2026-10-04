from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import analysis
from src.charts import OUTLINE, finish

MAX_PIT_LANE = 60.0
UNDERCUT_GAP = 3.0
UNDERCUT_WINDOW = 5
MIN_STINT = 5
SOFTEST_FIRST = ("SOFT", "MEDIUM", "HARD")
STOP_COLUMNS = ["Driver", "Team", "Lap", "PitLane", "TimeLost", "From", "To"]
UNDERCUT_COLUMNS = ["Lap", "Attacker", "Defender", "Gap", "Worked"]


def pit_stops(laps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    green = analysis.representative_laps(laps)
    typical = green.groupby("Driver")["LapTimeSeconds"].median()
    for driver, driver_laps in laps.sort_values("LapNumber").groupby("Driver"):
        driver_laps = driver_laps.reset_index(drop=True)
        for index in driver_laps.index[driver_laps["PitInTime"].notna()]:
            if index + 1 >= len(driver_laps):
                continue
            in_lap, out_lap = driver_laps.loc[index], driver_laps.loc[index + 1]
            if pd.isna(out_lap["PitOutTime"]):
                continue
            pit_lane = (out_lap["PitOutTime"] - in_lap["PitInTime"]).total_seconds()
            if not 0 < pit_lane <= MAX_PIT_LANE:
                continue
            lost = (
                in_lap["LapTimeSeconds"]
                + out_lap["LapTimeSeconds"]
                - 2 * typical.get(driver, np.nan)
            )
            rows.append(
                {
                    "Driver": driver,
                    "Team": in_lap["Team"],
                    "Lap": int(in_lap["LapNumber"]),
                    "PitLane": pit_lane,
                    "TimeLost": lost,
                    "From": in_lap["Compound"],
                    "To": out_lap["Compound"],
                }
            )
    return pd.DataFrame(rows, columns=STOP_COLUMNS)


def pit_loss(stops: pd.DataFrame) -> float | None:
    lost = stops["TimeLost"].dropna()
    lost = lost[(lost > 5) & (lost < 60)]
    return float(lost.median()) if not lost.empty else None


def pit_stop_figure(stops: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    order = stops.groupby("Team")["PitLane"].median().sort_values().index.tolist()
    figure = go.Figure()
    for team in order:
        rows = stops[stops["Team"] == team]
        figure.add_trace(
            go.Box(
                x=[team] * len(rows),
                y=rows["PitLane"],
                name=team,
                boxpoints="all",
                pointpos=0,
                jitter=0.3,
                fillcolor="rgba(0, 0, 0, 0)",
                line={"color": OUTLINE, "width": 1},
                marker={
                    "color": colors.get(team, OUTLINE),
                    "size": 8,
                    "line": {"color": OUTLINE, "width": 1},
                },
                customdata=np.column_stack([rows["Driver"], rows["Lap"]]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · lap %{customdata[1]}<br>%{y:.1f} s<extra></extra>"
                ),
            )
        )
    figure.update_yaxes(title="Time in the pit lane (s)")
    figure.update_xaxes(tickangle=-30)
    return finish(figure, 400, showlegend=False)


def undercuts(
    laps: pd.DataFrame,
    stops: pd.DataFrame,
    gap: float = UNDERCUT_GAP,
    window: int = UNDERCUT_WINDOW,
) -> pd.DataFrame:
    timed = laps.dropna(subset=["Time"]).assign(
        Seconds=lambda rows: rows["Time"].dt.total_seconds()
    )
    positions = analysis.lap_positions(laps).set_index(["LapNumber", "Driver"])["Position"]
    seconds = timed.set_index(["LapNumber", "Driver"])["Seconds"]
    pitted = {(row.Driver, row.Lap) for row in stops.itertuples()}
    rows = []
    for stop in stops.sort_values("Lap").itertuples():
        before = stop.Lap - 1
        if (before, stop.Driver) not in positions.index:
            continue
        place = positions[(before, stop.Driver)]
        ahead = positions.loc[before]
        ahead = ahead[ahead == place - 1].index.tolist()
        if not ahead:
            continue
        rival = ahead[0]
        interval = seconds.get((before, stop.Driver)) - seconds.get((before, rival))
        if not interval <= gap:
            continue
        reply = next(
            (lap for lap in range(stop.Lap + 1, stop.Lap + window + 1) if (rival, lap) in pitted),
            None,
        )
        if reply is None:
            continue
        after = reply + 1
        if (after, stop.Driver) not in positions.index or (after, rival) not in positions.index:
            continue
        rows.append(
            {
                "Lap": stop.Lap,
                "Attacker": stop.Driver,
                "Defender": rival,
                "Gap": interval,
                "Worked": bool(positions[(after, stop.Driver)] < positions[(after, rival)]),
            }
        )
    return pd.DataFrame(rows, columns=UNDERCUT_COLUMNS)


def compound_offsets(
    laps: pd.DataFrame,
    total_laps: int,
    compounds: list[str],
    min_laps: int = 15,
    min_drivers: int = 2,
) -> dict[str, float]:
    quick = analysis.quick_laps(laps).dropna(subset=["Stint", "TyreLife"])
    quick = quick[quick["Compound"].isin(compounds)]
    usage = quick.groupby("Compound").agg(laps=("LapNumber", "size"), drivers=("Driver", "nunique"))
    used = usage[(usage["laps"] >= min_laps) & (usage["drivers"] >= min_drivers)].index
    quick = quick[quick["Compound"].isin(used)]
    if len(used) < 2:
        return {}
    # Fresh-tyre pace per compound, with each driver's pace, tyre wear and the track
    # rubbering in taken out, so a compound only used early or late in the race isn't
    # mistaken for a slow or fast one.
    compound = pd.get_dummies(quick["Compound"], dtype=float)
    design = pd.concat(
        [
            compound,
            compound.mul(quick["TyreLife"], axis=0).add_suffix(" age"),
            pd.get_dummies(quick["Driver"], drop_first=True, dtype=float),
            quick["LapNumber"].astype(float),
        ],
        axis=1,
    )
    target = analysis.fuel_corrected(quick, total_laps).to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design.to_numpy(), target, rcond=None)
    pace = pd.Series(coefficients[: len(compound.columns)], index=compound.columns)
    return {name: float(value - pace.min()) for name, value in pace.items()}


def _stint_costs(curve: pd.DataFrame, offset: float, length: int) -> np.ndarray:
    ages = np.arange(1, length + 1, dtype=float)
    loss = np.interp(ages, curve["TyreLife"], curve["Predicted"], left=0.0)
    return np.concatenate([[0.0], np.cumsum(offset + loss)])


@dataclass(frozen=True)
class Plan:
    compounds: tuple[str, ...]
    pit_laps: tuple[int, ...]
    time: float


def plans(
    curves: pd.DataFrame,
    offsets: dict[str, float],
    total_laps: int,
    pit_loss: float,
    max_stops: int = 2,
) -> list[Plan]:
    usable = [compound for compound in offsets if compound in set(curves["Compound"])]
    longest = curves.groupby("Compound")["TyreLife"].max().to_dict()
    costs = {
        compound: _stint_costs(
            curves[curves["Compound"] == compound], offsets[compound], total_laps
        )
        for compound in usable
    }
    best: list[Plan] = []
    for stops in range(1, max_stops + 1):
        for sequence in product(usable, repeat=stops + 1):
            if len(set(sequence)) < 2:
                continue
            plan = _best_split(sequence, costs, longest, total_laps)
            if plan is not None:
                pit_laps, time = plan
                best.append(Plan(sequence, pit_laps, time + stops * pit_loss))
    # Stint cost only depends on compound and length, so M-H and H-M tie; keep the one
    # that starts on the softer tyre.
    softness = {compound: index for index, compound in enumerate(SOFTEST_FIRST)}
    best.sort(key=lambda plan: (round(plan.time, 6), [softness.get(c, 9) for c in plan.compounds]))
    unique, seen = [], set()
    for plan in best:
        bounds = (0, *plan.pit_laps, total_laps)
        key = tuple(sorted(zip(plan.compounds, np.diff(bounds), strict=True)))
        if key not in seen:
            seen.add(key)
            unique.append(plan)
    return unique


def _best_split(
    sequence: tuple[str, ...],
    costs: dict[str, np.ndarray],
    longest: dict[str, float],
    total_laps: int,
) -> tuple[tuple[int, ...], float] | None:
    best = None
    for lengths in _splits(len(sequence), total_laps):
        if any(
            length > longest[compound] for compound, length in zip(sequence, lengths, strict=True)
        ):
            continue
        time = sum(
            costs[compound][length] for compound, length in zip(sequence, lengths, strict=True)
        )
        if best is None or time < best[1]:
            pit_laps = tuple(int(lap) for lap in np.cumsum(lengths)[:-1])
            best = (pit_laps, float(time))
    return best


def _splits(stints: int, total_laps: int) -> list[tuple[int, ...]]:
    if stints == 1:
        return [(total_laps,)]
    splits = []
    for first in range(MIN_STINT, total_laps - MIN_STINT * (stints - 1) + 1):
        splits += [(first, *rest) for rest in _splits(stints - 1, total_laps - first)]
    return splits


def driver_plan(laps: pd.DataFrame, driver: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    stints = analysis.stint_summary(laps[laps["Driver"] == driver]).sort_values("Stint")
    return tuple(stints["Compound"]), tuple(int(lap) for lap in stints["EndLap"].iloc[:-1])


def plan_time(
    plan: tuple[tuple[str, ...], tuple[int, ...]],
    curves: pd.DataFrame,
    offsets: dict[str, float],
    total_laps: int,
    pit_loss: float,
) -> float | None:
    compounds, pit_laps = plan
    if any(compound not in offsets for compound in compounds):
        return None
    bounds = (0, *pit_laps, total_laps)
    time = len(pit_laps) * pit_loss
    for compound, start, end in zip(compounds, bounds[:-1], bounds[1:], strict=True):
        curve = curves[curves["Compound"] == compound]
        if curve.empty:
            return None
        time += _stint_costs(curve, offsets[compound], end - start)[-1]
    return float(time)
