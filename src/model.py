from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src import degradation, races, seasons

CATEGORICAL = ["Compound", "Location"]
NUMERIC = ["TyreLife", "StartAge", "Year"]
FEATURES = CATEGORICAL + NUMERIC
FOLDS = 5
LONGEST_STINT_QUANTILE = 0.95


def pipeline() -> Pipeline:
    encode = ColumnTransformer(
        [
            (
                "categories",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                CATEGORICAL,
            )
        ],
        remainder="passthrough",
    )
    regressor = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL))),
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=50,
        random_state=0,
    )
    return Pipeline([("encode", encode), ("regressor", regressor)])


def bundled_races(root: Path = races.DATA_DIR) -> pd.DataFrame:
    saved = races.saved_races(root)
    saved = saved[saved["Year"] >= degradation.FIRST_SEASON].reset_index(drop=True)
    locations = {
        year: seasons.load_season(year, root).schedule.set_index("Round")["Location"]
        for year in saved["Year"].unique()
    }
    return saved.assign(
        Race=[
            degradation.race_id(y, r) for y, r in zip(saved["Year"], saved["Round"], strict=True)
        ],
        Location=[locations[y].get(r) for y, r in zip(saved["Year"], saved["Round"], strict=True)],
    )


def training_table(bundled: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in bundled.itertuples():
        race = races.read_race(row.Path)
        losses = degradation.stint_losses(race.laps, race.total_laps)
        if not losses.empty:
            frames.append(losses.assign(Race=row.Race, Year=row.Year, Location=row.Location))
    return pd.concat(frames, ignore_index=True)


def _laps_into_stint(rows: pd.DataFrame) -> pd.DataFrame:
    return (rows["TyreLife"] - rows["StartAge"]).to_frame("LapsIntoStint")


def baseline_errors(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    errors = []
    for compound, rows in test.groupby("Compound"):
        fitted = train[train["Compound"] == compound]
        line = LinearRegression().fit(_laps_into_stint(fitted), fitted["Loss"])
        errors.append(np.abs(line.predict(_laps_into_stint(rows)) - rows["Loss"].to_numpy()))
    return np.concatenate(errors)


def curve_grid(bundled: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    longest = table.groupby("Compound")["TyreLife"].quantile(LONGEST_STINT_QUANTILE)
    ages = {
        compound: np.arange(degradation.FRESH_TYRE_AGE, int(longest[compound]) + 1, dtype=float)
        for compound in degradation.DRY_COMPOUNDS
        if compound in longest
    }
    return pd.DataFrame(
        [
            {
                "Race": race.Race,
                "Year": race.Year,
                "Location": race.Location,
                "Compound": compound,
                "TyreLife": age,
                "StartAge": float(degradation.FRESH_TYRE_AGE),
            }
            for race in bundled.itertuples()
            for compound, compound_ages in ages.items()
            for age in compound_ages
        ]
    )


def cross_validate(
    table: pd.DataFrame, grid: pd.DataFrame, folds: int = FOLDS
) -> tuple[pd.DataFrame, dict[str, float]]:
    curves, errors = [], {"model": [], "baseline": [], "no_wear": []}
    # Grouped by race so laps from the same race never end up on both sides.
    for train_rows, test_rows in GroupKFold(n_splits=folds).split(table, groups=table["Race"]):
        train, test = table.iloc[train_rows], table.iloc[test_rows]
        model = pipeline().fit(train[FEATURES], train["Loss"])
        errors["model"].append(np.abs(model.predict(test[FEATURES]) - test["Loss"].to_numpy()))
        errors["baseline"].append(baseline_errors(train, test))
        errors["no_wear"].append(np.abs(test["Loss"].to_numpy()))
        held_out = grid[grid["Race"].isin(test["Race"].unique())]
        curves.append(held_out.assign(Predicted=model.predict(held_out[FEATURES])))

    unseen = grid[~grid["Race"].isin(table["Race"].unique())]
    if not unseen.empty:
        model = pipeline().fit(table[FEATURES], table["Loss"])
        curves.append(unseen.assign(Predicted=model.predict(unseen[FEATURES])))
    mae = {name: float(np.concatenate(values).mean()) for name, values in errors.items()}
    return pd.concat(curves, ignore_index=True)[degradation.CURVE_COLUMNS], mae


def build(root: Path = races.DATA_DIR) -> Path:
    bundled = bundled_races(root)
    table = training_table(bundled)
    curves, mae = cross_validate(table, curve_grid(bundled, table))
    path = degradation.model_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    curves.astype({"TyreLife": "float32", "Predicted": "float32"}).to_parquet(
        path / "curves.parquet", index=False
    )
    metrics = {
        "laps": len(table),
        "stints": int(table.groupby(["Race", "Driver", "Stint"]).ngroups),
        "races": int(table["Race"].nunique()),
        "first_season": int(table["Year"].min()),
        "last_season": int(table["Year"].max()),
        "folds": FOLDS,
        "mae": {name: round(value, 3) for name, value in mae.items()},
    }
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def main() -> int:
    path = build()
    metrics = json.loads((path / "metrics.json").read_text())
    print(f"saved  {path.relative_to(races.ROOT)}: {json.dumps(metrics['mae'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
