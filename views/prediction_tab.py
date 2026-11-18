from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src import predictions, races
from src.charts import Styles
from views import data, logos, theme
from views.state import favourite
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


def next_race_banner(bundle_version: str, year: int, event: str) -> None:
    """Points to the next race's prediction from any other race."""
    found = next_race(bundle_version)
    if found is None or (found[0].year, found[1]) == (year, event):
        return
    # Imported here: the pages module imports this one.
    from views import pages

    _, name, pick = found
    with st.container(border=True):
        st.markdown(
            f":material/insights: **{name}**: the model picks **{pick['Driver']}** to win from "
            f"P{pick['QualiPosition']:.0f}."
        )
        st.page_link(pages.PREDICTIONS, label="See the prediction", icon=":material/arrow_forward:")


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


def prediction_tab(race: races.Race, bundle_version: str) -> None:
    """The forecast after qualifying, and once the race is run, how close it came."""
    weekend = Weekend(race.year, race.round_number, race.driver_styles)
    predicted = weekend_predictions(weekend, bundle_version)
    if predicted.empty:
        st.info("There's no prediction for this race: it needs its qualifying session.")
    elif predicted["Position"].isna().all():
        forecast(weekend, bundle_version, "How will the race go?")
    else:
        review(weekend, bundle_version, "How predictable was it?")
