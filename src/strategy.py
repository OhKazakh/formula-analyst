from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import analysis
from src.charts import ACCENT, NEUTRAL, OUTLINE, driver_style, finish, hard_to_see

MAX_PIT_LANE = 60.0
UNDERCUT_GAP = 3.0
UNDERCUT_WINDOW = 5
MIN_STINT = 5
SOFTEST_FIRST = ("SOFT", "MEDIUM", "HARD")
STOP_COLUMNS = ["Driver", "Team", "Lap", "PitLane", "TimeLost", "From", "To"]
UNDERCUT_COLUMNS = ["Lap", "Attacker", "Defender", "Gap", "Worked"]
HOVER_GAP = " · +%{customdata:.1f} s<extra></extra>"


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
            if pd.isna(out_lap["PitOutTime"]) or out_lap["Stint"] == in_lap["Stint"]:
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


@dataclass(frozen=True)
class Rejoin:
    driver: str
    lap: int
    loss: float
    stops: int  # how many stops the loss is the median of
    neutralised: bool  # safety car or VSC on the lap, so the loss is the neutralised one
    gap_before: float
    position_before: int
    gap_after: float
    position_after: int
    ahead: tuple[str, float] | None  # the car ahead and how far ahead of the driver
    behind: tuple[str, float] | None  # the car behind and how far behind the driver
    gaps: pd.DataFrame  # everyone's gap to the leader at the end of the lap
    actual_position: int | None  # where the driver came out if they really stopped on this lap


def _neutral(laps: pd.DataFrame) -> set[int]:
    laps_under = analysis.neutralised_laps(laps)
    return set(laps_under["SC"]) | set(laps_under["VSC"])


def rejoin(laps: pd.DataFrame, stops: pd.DataFrame, driver: str, lap: int) -> Rejoin | None:
    """Where a driver comes out if they pit at the end of `lap`.

    Their gap to the leader at the line plus the race's pit loss, set among everyone else's
    gaps on that lap. It assumes the others keep their gaps for the lap the stop takes,
    the usual pit-window estimate. A stop under a safety car or VSC costs less, so the loss
    comes from stops made in the same conditions when there are any.
    """
    neutral = _neutral(laps)
    under = lap in neutral or lap + 1 in neutral
    same = stops[stops["Lap"].isin(neutral) == under] if not stops.empty else stops
    loss = pit_loss(same) if not same.empty else None
    if loss is None:
        loss, same, under = pit_loss(stops), stops, False
    if loss is None:
        return None
    gaps = analysis.gap_to_leader(laps)
    on_lap = gaps[gaps["LapNumber"] == lap].sort_values("Gap").reset_index(drop=True)
    own = on_lap[on_lap["Driver"] == driver]
    if own.empty:
        return None
    gap = float(own["Gap"].iloc[0])
    after = gap + loss
    others = on_lap[on_lap["Driver"] != driver]
    in_front, behind = others[others["Gap"] <= after], others[others["Gap"] > after]
    actual = None
    if not stops[(stops["Driver"] == driver) & (stops["Lap"] == lap)].empty:
        nxt = gaps[gaps["LapNumber"] == lap + 1].sort_values("Gap").reset_index(drop=True)
        hit = nxt.index[nxt["Driver"] == driver]
        actual = int(hit[0]) + 1 if len(hit) else None
    lost = same["TimeLost"].dropna()
    return Rejoin(
        driver=driver,
        lap=int(lap),
        loss=loss,
        stops=int(((lost > 5) & (lost < 60)).sum()),
        neutralised=under,
        gap_before=gap,
        position_before=int(own.index[0]) + 1,
        gap_after=after,
        position_after=len(in_front) + 1,
        ahead=(in_front.iloc[-1]["Driver"], after - float(in_front.iloc[-1]["Gap"]))
        if not in_front.empty
        else None,
        behind=(behind.iloc[0]["Driver"], float(behind.iloc[0]["Gap"]) - after)
        if not behind.empty
        else None,
        gaps=on_lap,
        actual_position=actual,
    )


def rejoin_figure(result: Rejoin, styles: analysis.Styles, lap_time: float) -> go.Figure:
    """The lap as a clock: one lap of race pace around the ring, the leader at the top and
    everyone else clockwise by their gap. The driver's pit loss is the arc they travel."""
    angle = lambda gap: (gap / lap_time) * 360.0  # noqa: E731
    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=[1.0] * 361,
            theta=list(range(361)),
            mode="lines",
            line={"color": NEUTRAL, "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    arc = np.linspace(angle(result.gap_before), angle(result.gap_after), 60)
    figure.add_trace(
        go.Scatterpolar(
            r=[1.0] * len(arc),
            theta=arc,
            mode="lines",
            opacity=0.35,
            line={"color": ACCENT, "width": 14},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    others = result.gaps[result.gaps["Driver"] != result.driver]
    marker_colors = [driver_style(styles, driver)[0] for driver in others["Driver"]]
    figure.add_trace(
        go.Scatterpolar(
            r=[1.0] * len(others),
            theta=[angle(gap) for gap in others["Gap"]],
            mode="markers",
            showlegend=False,
            marker={
                "size": 12,
                "color": marker_colors,
                "line": {
                    "color": [
                        OUTLINE if hard_to_see(c) else "rgba(0,0,0,0)" for c in marker_colors
                    ],
                    "width": 1,
                },
            },
            customdata=np.column_stack([others["Driver"], others["Gap"]]),
            hovertemplate="<b>%{customdata[0]}</b> · +%{customdata[1]:.1f} s<extra></extra>",
        )
    )
    angles, radii, last = [angle(gap) for gap in others["Gap"]], [], None
    for a in angles:
        close = last is not None and abs(a - last[0]) < 7
        radii.append(1.3 if close and last[1] == 1.17 else 1.17)
        last = (a, radii[-1])
    figure.add_trace(
        go.Scatterpolar(
            r=radii,
            theta=angles,
            mode="text",
            text=others["Driver"],
            textfont={"size": 11},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=[1.0],
            theta=[angle(result.gap_before)],
            mode="markers",
            showlegend=False,
            marker={"size": 14, "color": "rgba(0,0,0,0)", "line": {"color": ACCENT, "width": 2}},
            hovertemplate=f"<b>{result.driver}</b> on lap {result.lap}{HOVER_GAP}",
            customdata=[result.gap_before],
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=[1.0, 0.78],
            theta=[angle(result.gap_after)] * 2,
            mode="markers+text",
            text=["", f"{result.driver} P{result.position_after}"],
            textfont={"size": 13, "color": ACCENT},
            marker={"size": [18, 0], "color": ACCENT},
            showlegend=False,
            hovertemplate=f"<b>{result.driver}</b> after the stop{HOVER_GAP}",
            customdata=[result.gap_after, result.gap_after],
        )
    )
    figure.update_layout(
        polar={
            "bgcolor": "rgba(0, 0, 0, 0)",
            "radialaxis": {"visible": False, "range": [0, 1.4]},
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "showticklabels": False,
                "showgrid": False,
                "showline": False,
                "ticks": "",
            },
        }
    )
    return finish(figure, 520, showlegend=False)


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
