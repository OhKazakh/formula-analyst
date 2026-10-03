from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import analysis
from src.races import DATA_DIR

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
# 2018 still used the old compound names (hypersoft, ultrasoft, ...).
FIRST_SEASON = 2019
MIN_STINT_LAPS = 6
OPENING_LAPS = 3
FRESH_TYRE_AGE = 2
HORIZON = 15
MODEL_DIR = "tyre-model"
LOSS_COLUMNS = ["Driver", "Stint", "Compound", "TyreLife", "StartAge", "Loss"]
CURVE_COLUMNS = ["Race", "Compound", "TyreLife", "Predicted"]


def race_id(year: int, round_number: int) -> str:
    return f"{year}-{round_number:02d}"


def stint_losses(
    laps: pd.DataFrame, total_laps: int, min_laps: int = MIN_STINT_LAPS
) -> pd.DataFrame:
    quick = analysis.quick_laps(laps).dropna(subset=["Stint", "TyreLife", "LapTimeSeconds"])
    quick = quick[quick["Compound"].isin(DRY_COMPOUNDS)]
    quick = quick.assign(Corrected=analysis.fuel_corrected(quick, total_laps))
    frames = []
    for (driver, stint), stint_laps in quick.groupby(["Driver", "Stint"]):
        if len(stint_laps) < min_laps:
            continue
        start_age = stint_laps["TyreLife"].min()
        opening = stint_laps["TyreLife"] < start_age + OPENING_LAPS
        baseline = stint_laps.loc[opening, "Corrected"].median()
        frames.append(
            pd.DataFrame(
                {
                    "Driver": driver,
                    "Stint": int(stint),
                    "Compound": analysis.stint_compound(stint_laps["Compound"]),
                    "TyreLife": stint_laps["TyreLife"].to_numpy(dtype=float),
                    "StartAge": float(start_age),
                    "Loss": (stint_laps["Corrected"] - baseline).to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=LOSS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def model_dir(root: Path = DATA_DIR) -> Path:
    return root / MODEL_DIR


def load_curves(root: Path = DATA_DIR) -> pd.DataFrame:
    path = model_dir(root) / "curves.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=CURVE_COLUMNS)


def load_metrics(root: Path = DATA_DIR) -> dict | None:
    path = model_dir(root) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def race_curves(curves: pd.DataFrame, year: int, round_number: int) -> pd.DataFrame:
    rows = curves[curves["Race"] == race_id(year, round_number)]
    return rows[["Compound", "TyreLife", "Predicted"]].reset_index(drop=True)


def loss_after(curves: pd.DataFrame, laps: int = HORIZON) -> dict[str, float]:
    age = FRESH_TYRE_AGE + laps
    at_horizon = curves[curves["TyreLife"] == age]
    return {
        compound: float(at_horizon.loc[at_horizon["Compound"] == compound, "Predicted"].iat[0])
        for compound in DRY_COMPOUNDS
        if (at_horizon["Compound"] == compound).any()
    }
