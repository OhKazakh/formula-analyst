from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src import predictions, races, seasons
from src.charts import Styles
from views import data, logos, theme
from views.links import requested_int
from views.state import (
    favourite,
    initial,
    pick_favourite,
    remember,
    remembered_round,
    remembered_season,
)
from views.text import plural

BADGE = st.column_config.ImageColumn(" ", width=44)
PLACE = st.column_config.NumberColumn(width=84)
NAME = st.column_config.TextColumn(width=180)
OUTCOME_WIDTH = 104
# formula1.com's purple for spot on, green for places gained and red for places lost.
OUTCOMES = {
    "exact": ("#5300A6", "#FFFFFF"),
    "better": ("#28973E", "#15151E"),
    "worse": ("#E10600", "#FFFFFF"),
}
RESULT_WIDTH = 84
# The same green and red for a winner called or missed.
RESULTS = {True: ("Called it", "#28973E", "#15151E"), False: ("Missed", "#E10600", "#FFFFFF")}


@dataclass(frozen=True)
class Weekend:
    """A race weekend as the prediction views need it, from a race page or a season."""

    year: int
    round_number: int
    styles: Styles


def weekend_predictions(weekend: Weekend, bundle_version: str) -> pd.DataFrame:
    table = data.race_predictions(bundle_version)
    return table[(table["Year"] == weekend.year) & (table["Round"] == weekend.round_number)]


def badges(weekend: Weekend, table: pd.DataFrame) -> list[str]:
    return [
        logos.badge(str(team), weekend.styles.get(driver, {}).get("color"), weekend.year)
        for driver, team in zip(table["Driver"], table["Team"], strict=True)
    ]


def names(year: int, bundle_version: str, drivers: pd.Series) -> list[str]:
    known = {}
    if year in data.bundled_seasons(bundle_version):
        results = data.season(year, bundle_version).results
        known = results.drop_duplicates("Driver", keep="last").set_index("Driver")["Name"]
    return [known.get(driver, driver) for driver in drivers]


def outcome(places: int) -> str:
    """How the finish compared with the prediction, as a coloured pill."""
    if not places:
        return theme.pill("Spot on", *OUTCOMES["exact"], OUTCOME_WIDTH)
    if places > 0:
        return theme.pill(f"▲ {plural(places, 'place')}", *OUTCOMES["better"], OUTCOME_WIDTH)
    return theme.pill(f"▼ {plural(-places, 'place')}", *OUTCOMES["worse"], OUTCOME_WIDTH)


def show_table(table: pd.DataFrame, marked: pd.Series, columns: dict) -> None:
    st.dataframe(
        theme.favourite_rows(table, marked),
        hide_index=True,
        width="content",
        height=35 * (len(table) + 1) + 3,
        column_config=columns,
    )


def next_race(bundle_version: str) -> tuple[Weekend, str, pd.Series] | None:
    """The race the model has predicted that hasn't been run yet, and its pick to win."""
    table = data.race_predictions(bundle_version)
    found = predictions.upcoming_weekend(table)
    if found is None:
        return None
    year, round_number = found
    saved = data.bundled_sessions(bundle_version)
    events = saved.loc[(saved["Year"] == year) & (saved["Round"] == round_number), "EventName"]
    if events.empty:
        return None
    season = (
        data.season(year, bundle_version) if year in data.bundled_seasons(bundle_version) else None
    )
    weekend = Weekend(year, round_number, season.driver_styles if season else {})
    pick = weekend_predictions(weekend, bundle_version).sort_values("Predicted").iloc[0]
    return weekend, str(events.iloc[0]), pick


def model_record(bundle_version: str) -> str:
    metrics = data.prediction_metrics(bundle_version)
    if not metrics:
        return ""
    model, grid = metrics["model"], metrics["qualifying"]
    return (
        f"Over {metrics['races']} races it picked the winner {model['winner']:.0%} of the time, "
        f"against {grid['winner']:.0%} for pole position, and was {model['position_error']:.2f} "
        f"places off on average, against {grid['position_error']:.2f} for the qualifying order: "
        "qualifying decides most races."
    )


def how_it_works(bundle_version: str) -> str:
    return (
        "The model reads each driver's qualifying position and gap to pole, best practice "
        "long run and average finish over the last five races, for the driver and the team. "
        "Every race is predicted by a model trained on the other races. "
        + model_record(bundle_version)
    )


def review(weekend: Weekend, bundle_version: str, heading: str) -> None:
    """How a race went against its prediction."""
    predicted = weekend_predictions(weekend, bundle_version)
    st.subheader(heading)
    # The pick stands even if that driver didn't start.
    picked = predicted.loc[predicted["Predicted"].idxmin()]
    raced = predicted.dropna(subset=["Position"]).sort_values("Position")
    winner = raced.iloc[0]
    if picked["Driver"] == winner["Driver"]:
        text = (
            f"The model picked **{winner['Driver']}** to win from P{winner['QualiPosition']:.0f} "
            "in qualifying, and called it."
        )
    else:
        text = (
            f"The model picked **{picked['Driver']}** (P{picked['QualiPosition']:.0f} in "
            f"qualifying) to win. **{winner['Driver']}** won from "
            f"P{winner['QualiPosition']:.0f}."
        )
    podium = set(predicted.loc[predicted["Predicted"] <= 3, "Driver"]) & set(
        raced.loc[raced["Position"] <= 3, "Driver"]
    )
    model_error = (raced["Predicted"] - raced["Position"]).abs().mean()
    grid_error = (raced["QualiPosition"] - raced["Position"]).abs().mean()
    text += (
        f" It had {len(podium)} of the podium in its top three and was {model_error:.1f} places "
        f"off on average, against {grid_error:.1f} for the qualifying order."
    )
    st.markdown(text)

    beat = (raced["Predicted"] - raced["Position"]).astype(int)
    table = pd.DataFrame(
        {
            "Finished": raced["Position"].astype(int).to_numpy(),
            "Badge": badges(weekend, raced),
            "Driver": names(weekend.year, bundle_version, raced["Driver"]),
            "Predicted": raced["Predicted"].astype(int).to_numpy(),
            "Qualified": raced["QualiPosition"].astype("Int64").to_numpy(),
            "Result": [outcome(places) for places in beat],
        }
    )
    show_table(
        table,
        raced["Driver"].eq(favourite()).reset_index(drop=True),
        {
            "Finished": PLACE,
            "Badge": BADGE,
            "Driver": NAME,
            "Predicted": PLACE,
            "Qualified": PLACE,
            "Result": st.column_config.ImageColumn(
                "Against the prediction", width=OUTCOME_WIDTH + 52
            ),
        },
    )
    st.caption(
        "Green gained places on the prediction, red lost them, purple finished exactly where "
        "predicted. " + how_it_works(bundle_version)
    )


def forecast(weekend: Weekend, bundle_version: str, heading: str) -> None:
    """The predicted race result, made after qualifying and shown until the race is run."""
    predicted = weekend_predictions(weekend, bundle_version).sort_values("Predicted")
    st.subheader(heading)
    pick = predicted.iloc[0]
    st.markdown(f"The model picks **{pick['Driver']}** to win from P{pick['QualiPosition']:.0f}.")
    table = pd.DataFrame(
        {
            "Predicted": predicted["Predicted"].astype(int).to_numpy(),
            "Badge": badges(weekend, predicted),
            "Driver": names(weekend.year, bundle_version, predicted["Driver"]),
            "Qualified": predicted["QualiPosition"].astype("Int64").to_numpy(),
        }
    )
    show_table(
        table,
        predicted["Driver"].eq(favourite()).reset_index(drop=True),
        {"Predicted": PLACE, "Badge": BADGE, "Driver": NAME, "Qualified": PLACE},
    )
    st.caption(how_it_works(bundle_version))


def points(share: float) -> str:
    """A difference between two shares in percentage points, as in "+3 points"."""
    count = round(share * 100)
    return f"{count:+d} point{'' if abs(count) == 1 else 's'}"


def overall(bundle_version: str) -> None:
    metrics = data.prediction_metrics(bundle_version)
    if not metrics:
        return
    model, grid = metrics["model"], metrics["qualifying"]
    winners, podium, places = st.columns(3)
    winners.metric(
        "Winners called",
        f"{model['winner']:.0%}",
        f"{points(model['winner'] - grid['winner'])} on pole position",
        help="How often the model's pick won, against how often the pole sitter won.",
    )
    podium.metric(
        "Podium places",
        f"{model['podium']:.0%}",
        f"{points(model['podium'] - grid['podium'])} on the qualifying order",
        help="How much of the podium was in the model's top three.",
    )
    places.metric(
        "Places off on average",
        f"{model['position_error']:.2f}",
        f"{model['position_error'] - grid['position_error']:+.2f} on the qualifying order",
        delta_color="inverse",
        help="How far each finish was from the predicted place.",
    )
    st.caption(
        f"Over {metrics['races']} races since 2018, each predicted by a model that never saw it."
    )


def record_table(record: pd.DataFrame, season: seasons.Season, table: pd.DataFrame) -> pd.DataFrame:
    teams = table[table["Year"] == season.year].set_index(["Round", "Driver"])["Team"]

    def badge(round_number: int, driver: str) -> str:
        color = season.driver_styles.get(driver, {}).get("color")
        return logos.badge(str(teams.get((round_number, driver), "")), color, season.year)

    return pd.DataFrame(
        {
            "Round": record["Round"],
            "Grand Prix": [
                season.event_name(r).removesuffix(" Grand Prix") for r in record["Round"]
            ],
            "Pick badge": [
                badge(r, d) for r, d in zip(record["Round"], record["Picked"], strict=True)
            ],
            "Picked": [
                f"{driver} from P{grid:.0f}"
                for driver, grid in zip(record["Picked"], record["PickedFrom"], strict=True)
            ],
            "Winner badge": [
                badge(r, d) for r, d in zip(record["Round"], record["Winner"], strict=True)
            ],
            "Winner": [
                f"{driver} from P{grid:.0f}"
                for driver, grid in zip(record["Winner"], record["WinnerFrom"], strict=True)
            ],
            "Result": [
                theme.pill(*RESULTS[bool(called)], RESULT_WIDTH) for called in record["Called"]
            ],
            "Podium": [f"{count} of 3" for count in record["Podium"]],
            "Places off": record["ModelError"].round(1),
            "Grid places off": record["GridError"].round(1),
        }
    )


def season_record(season: seasons.Season, table: pd.DataFrame) -> None:
    record = predictions.season_record(table, season.year)
    st.subheader(f"How did it do in {season.year}?")
    if record.empty:
        st.info("No race of this season has a prediction yet.")
        return
    pole_wins = int((record["WinnerFrom"] == 1).sum())
    st.markdown(
        f"It called **{int(record['Called'].sum())} of {len(record)}** winners, against "
        f"{pole_wins} for the pole sitter, and was {record['ModelError'].mean():.2f} places off "
        f"on average, against {record['GridError'].mean():.2f} for the qualifying order."
    )
    st.dataframe(
        record_table(record, season, table),
        hide_index=True,
        width="content",
        height=35 * (len(record) + 1) + 3,
        column_config={
            "Round": st.column_config.NumberColumn(width=60),
            "Grand Prix": st.column_config.TextColumn(width=150),
            "Pick badge": st.column_config.ImageColumn(" ", width=44),
            "Picked": st.column_config.TextColumn(width=110),
            "Winner badge": st.column_config.ImageColumn(" ", width=44),
            "Winner": st.column_config.TextColumn(width=110),
            "Result": st.column_config.ImageColumn("Result", width=RESULT_WIDTH + 8),
            "Podium": st.column_config.TextColumn(
                width=70, help="Podium finishers it had in its top three"
            ),
            "Places off": st.column_config.NumberColumn(
                width=80, help="How far each finish was from the predicted place, on average"
            ),
            "Grid places off": st.column_config.NumberColumn(
                width=110, help="The same for the qualifying order"
            ),
        },
    )


def render() -> None:
    bundle_version = races.bundle_version()
    table = data.race_predictions(bundle_version)
    st.title("Race predictions")
    if table.empty:
        st.info("No predictions are bundled.")
        return
    st.markdown(
        "After qualifying, a model trained on every race since 2018 predicts the finishing order "
        "from the grid, practice long runs and recent form. The forecast goes up within a few "
        "hours of qualifying and is checked against the result after the race."
    )
    overall(bundle_version)

    upcoming = next_race(bundle_version)
    if upcoming is None:
        st.subheader("What's next?")
        st.info(
            "No race is waiting to be run. The next forecast appears within a few hours of the "
            "next qualifying session."
        )
    else:
        weekend, name, _ = upcoming
        forecast(weekend, bundle_version, f"How will the {name} go?")

    years = sorted(table["Year"].unique().tolist(), reverse=True)
    bundled = data.bundled_seasons(bundle_version)
    years = [year for year in years if year in bundled]

    def choose_year() -> int:
        wanted = remembered_season() or requested_int("season")
        return years.index(wanted) if wanted in years else 0

    year = st.sidebar.selectbox(
        "Season", years, index=initial("predictions_season", choose_year), key="predictions_season"
    )
    season = data.season(year, bundle_version)
    pick_favourite(sorted(season.results["Driver"].unique().tolist()))
    season_record(season, table)

    rounds = predictions.season_record(table, year)["Round"].tolist()
    if not rounds:
        return

    def choose_round() -> int:
        wanted = remembered_round(year) or requested_int("round")
        return rounds.index(wanted) if wanted in rounds else len(rounds) - 1

    key = f"predictions_round-{year}"
    chosen = st.selectbox(
        "Race",
        rounds,
        index=initial(key, choose_round),
        format_func=lambda r: f"Round {r} · {season.event_name(r)}",
        key=key,
    )
    remember(year, chosen)
    st.query_params.from_dict({"season": str(year), "round": str(chosen)})
    review(
        Weekend(year, chosen, season.driver_styles),
        bundle_version,
        f"How predictable was the {season.event_name(chosen)}?",
    )
