from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from src import predictions, races
from src.predictions import FEATURES

FOLDS = 5


# Every feature can only push a driver further back: a worse grid slot, a bigger qualifying
# gap, slower long runs or weaker form. Without that the trees chase noise in 190 races.
def regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
        monotonic_cst=[1] * len(FEATURES),
        random_state=0,
    )


def predict_races(table: pd.DataFrame, folds: int = FOLDS) -> pd.DataFrame:
    """Predicted finishing order for every race, each from a model that never saw it."""
    known = table.dropna(subset=["Position"])
    upcoming = table[table["Position"].isna()]
    score = pd.Series(np.nan, index=table.index)
    weekend = known["Year"] * 100 + known["Round"]
    for train, test in GroupKFold(n_splits=folds).split(known, groups=weekend):
        fitted = regressor().fit(known.iloc[train][FEATURES], known.iloc[train]["Position"])
        score[known.index[test]] = fitted.predict(known.iloc[test][FEATURES])
    if not upcoming.empty:
        fitted = regressor().fit(known[FEATURES], known["Position"])
        score[upcoming.index] = fitted.predict(upcoming[FEATURES])
    predicted = score.groupby([table["Year"], table["Round"]]).rank(method="first")
    return table.assign(Predicted=predicted.astype(int))[predictions.PREDICTION_COLUMNS]


def build(root: Path = races.DATA_DIR) -> Path:
    table = predictions.feature_table(root)
    predicted = predict_races(table)
    path = predictions.model_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    predicted.to_parquet(path / "predictions.parquet", index=False)
    known = predicted.dropna(subset=["Position"])
    metrics = {
        "races": int(known.groupby(["Year", "Round"]).ngroups),
        "folds": FOLDS,
        "model": predictions.scores(known, "Predicted"),
        "qualifying": predictions.scores(known, "QualiPosition"),
    }
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def main() -> int:
    path = build()
    metrics = json.loads((path / "metrics.json").read_text())
    print(f"saved  {path.relative_to(races.ROOT)}: {json.dumps(metrics['model'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
