import pandas as pd
import streamlit as st

from src import races
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


def weekend_predictions(race: races.Race, bundle_version: str) -> pd.DataFrame:
    table = data.race_predictions(bundle_version)
    return table[(table["Year"] == race.year) & (table["Round"] == race.round_number)]


def badges(race: races.Race, table: pd.DataFrame) -> list[str]:
    return [
        logos.badge(str(team), race.driver_styles.get(driver, {}).get("color"), race.year)
        for driver, team in zip(table["Driver"], table["Team"], strict=True)
    ]


def names(race: races.Race, bundle_version: str, drivers: pd.Series) -> list[str]:
    known = {}
    if race.year in data.bundled_seasons(bundle_version):
        results = data.season(race.year, bundle_version).results
        known = results.drop_duplicates("Driver", keep="last").set_index("Driver")["Name"]
    return [known.get(driver, driver) for driver in drivers]


def outcome(places: int) -> str:
    """How the finish compared with the prediction, as a coloured pill."""
    if not places:
        return theme.pill("Spot on", *OUTCOMES["exact"], OUTCOME_WIDTH)
    if places > 0:
        return theme.pill(f"\u25b2 {plural(places, 'place')}", *OUTCOMES["better"], OUTCOME_WIDTH)
    return theme.pill(f"\u25bc {plural(-places, 'place')}", *OUTCOMES["worse"], OUTCOME_WIDTH)


def show_table(table: pd.DataFrame, marked: pd.Series, columns: dict) -> None:
    st.dataframe(
        theme.favourite_rows(table, marked),
        hide_index=True,
        width="content",
        height=35 * (len(table) + 1) + 3,
        column_config=columns,
    )


def next_race(bundle_version: str) -> tuple[int, str, pd.Series] | None:
    """The race the model has predicted that hasn't been run yet, and its pick to win."""
    table = data.race_predictions(bundle_version)
    upcoming = table[table["Position"].isna()]
    if upcoming.empty:
        return None
    year, round_number = upcoming[["Year", "Round"]].max()
    saved = data.bundled_sessions(bundle_version)
    names = saved.loc[(saved["Year"] == year) & (saved["Round"] == round_number), "EventName"]
    if names.empty:
        return None
    weekend = upcoming[(upcoming["Year"] == year) & (upcoming["Round"] == round_number)]
    return int(year), str(names.iloc[0]), weekend.sort_values("Predicted").iloc[0]


def next_race_banner(bundle_version: str, year: int, event: str) -> None:
    """Points to the next race's prediction from the race the app opens on."""
    found = next_race(bundle_version)
    if found is None or found[:2] == (year, event):
        return
    next_year, name, pick = found
    link = f"?season={next_year}&race={races.slugify(name)}&session=qualifying&tab=prediction"
    st.info(
        f"**{name}**: the model picks **{pick['Driver']}** to win from "
        f"P{pick['QualiPosition']:.0f}. [See the prediction]({link})",
        icon=":material/insights:",
    )


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


def prediction_tab(race: races.Race, bundle_version: str) -> None:
    """The forecast after qualifying, and once the race is run, how close it came."""
    weekend = weekend_predictions(race, bundle_version)
    if weekend.empty:
        st.info("There's no prediction for this race: it needs its qualifying session.")
        return
    if weekend["Position"].isna().all():
        upcoming_prediction(race, bundle_version)
        return
    st.subheader("How predictable was it?")
    weekend = weekend.dropna(subset=["Position"])
    weekend = weekend.sort_values("Position")
    picked = weekend.loc[weekend["Predicted"].idxmin()]
    winner = weekend.iloc[0]
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
    podium = set(weekend.loc[weekend["Predicted"] <= 3, "Driver"]) & set(
        weekend.loc[weekend["Position"] <= 3, "Driver"]
    )
    model_error = (weekend["Predicted"] - weekend["Position"]).abs().mean()
    grid_error = (weekend["QualiPosition"] - weekend["Position"]).abs().mean()
    text += (
        f" It had {len(podium)} of the podium in its top three and was {model_error:.1f} places "
        f"off on average, against {grid_error:.1f} for the qualifying order."
    )
    st.markdown(text)

    beat = (weekend["Predicted"] - weekend["Position"]).astype(int)
    table = pd.DataFrame(
        {
            "Finished": weekend["Position"].astype(int).to_numpy(),
            "Badge": badges(race, weekend),
            "Driver": names(race, bundle_version, weekend["Driver"]),
            "Predicted": weekend["Predicted"].astype(int).to_numpy(),
            "Qualified": weekend["QualiPosition"].astype("Int64").to_numpy(),
            "Result": [outcome(places) for places in beat],
        }
    )
    show_table(
        table,
        weekend["Driver"].eq(favourite()).reset_index(drop=True),
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


def upcoming_prediction(race: races.Race, bundle_version: str) -> None:
    """The predicted race result, made after qualifying and shown until the race is run."""
    weekend = weekend_predictions(race, bundle_version)
    st.subheader("How will the race go?")
    weekend = weekend.sort_values("Predicted")
    favourite_pick = weekend.iloc[0]
    st.markdown(
        f"The model picks **{favourite_pick['Driver']}** to win from "
        f"P{favourite_pick['QualiPosition']:.0f}."
    )
    table = pd.DataFrame(
        {
            "Predicted": weekend["Predicted"].astype(int).to_numpy(),
            "Badge": badges(race, weekend),
            "Driver": names(race, bundle_version, weekend["Driver"]),
            "Qualified": weekend["QualiPosition"].astype("Int64").to_numpy(),
        }
    )
    show_table(
        table,
        weekend["Driver"].eq(favourite()).reset_index(drop=True),
        {"Predicted": PLACE, "Badge": BADGE, "Driver": NAME, "Qualified": PLACE},
    )
    st.caption(how_it_works(bundle_version))
