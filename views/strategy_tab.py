import pandas as pd
import streamlit as st

from src import analysis, degradation, races, strategy
from src.charts import FALLBACK_COLOR
from views import data, tyres
from views.data import show
from views.state import race_key
from views.text import listing, plural

TYRE = st.column_config.ImageColumn(" ", width="small")
PER_LAP = st.column_config.NumberColumn(format="%.3f s/lap")


def with_badges(table: pd.DataFrame, colors: dict[str, str]) -> pd.DataFrame:
    badges = [tyres.badge(compound, colors) for compound in table["Compound"]]
    position = table.columns.get_loc("Compound")
    return table.assign(Tyre=badges)[[*table.columns[:position], "Tyre", *table.columns[position:]]]


def tyre_strategy(race: races.Race) -> None:
    st.subheader("What tyre strategy did everyone run?")
    stints = analysis.stint_summary(race.laps)
    show(analysis.strategy_figure(stints, race.drivers, race.compound_colors))
    with st.expander("Stint table"):
        st.dataframe(
            with_badges(stints, race.compound_colors).rename(
                columns={"StartLap": "Start lap", "EndLap": "End lap"}
            ),
            hide_index=True,
            width="stretch",
            column_config={"Tyre": TYRE},
        )


def pit_stops_section(race: races.Race) -> None:
    st.subheader("How long did the pit stops take?")
    stops = strategy.pit_stops(race.laps)
    if stops.empty:
        st.info("No pit stops were recorded for this race.")
        return
    quickest = stops.loc[stops["PitLane"].idxmin()]
    text = (
        f"The quickest stop was **{quickest['Driver']}**'s on lap {quickest['Lap']}, "
        f"{quickest['PitLane']:.1f} s in the pit lane."
    )
    if (loss := strategy.pit_loss(stops)) is not None:
        text += f" A stop cost about {loss:.0f} s against staying out."
    st.markdown(text)
    show(strategy.pit_stop_figure(stops, analysis.team_colors(race.laps, race.driver_styles)))
    st.caption(
        "Time from the pit entry line to the pit exit line, so slow pit lanes and penalties "
        "served in the box count too."
    )

    attempts = strategy.undercuts(race.laps, stops)
    if attempts.empty:
        return
    worked = int(attempts["Worked"].sum())
    st.markdown(f"**Undercuts.** {worked} of {plural(len(attempts), 'attempt')} worked.")
    st.dataframe(
        attempts.assign(Result=attempts["Worked"].map({True: "Worked", False: "Failed"})).drop(
            columns="Worked"
        ),
        hide_index=True,
        width="stretch",
        column_config={"Gap": st.column_config.NumberColumn("Gap before", format="%.1f s")},
    )
    st.caption(
        f"A driver within {strategy.UNDERCUT_GAP:.0f} s of the car ahead pitted first, and that "
        f"car pitted within {strategy.UNDERCUT_WINDOW} laps. The undercut worked if the driver "
        "came out ahead once both had stopped."
    )


def wear_rates(race: races.Race, clean: pd.DataFrame) -> None:
    st.subheader("How fast did the tyres degrade?")
    left, right = st.columns(2)
    min_laps = left.slider("Minimum laps per stint", 3, 20, 8, key=race_key(race, "min_laps"))
    fuel_effect = right.slider(
        "Fuel effect (s per lap of fuel)",
        0.0,
        0.1,
        analysis.DEFAULT_FUEL_EFFECT,
        0.005,
        key=race_key(race, "fuel_effect"),
    )

    laps = clean.copy()
    laps["FuelCorrected"] = analysis.fuel_corrected(laps, race.total_laps, fuel_effect)
    deg = analysis.degradation(laps, "FuelCorrected", min_laps)
    if deg.empty:
        st.info("No stints long enough to fit.")
        return

    summary, table = st.columns([2, 3])
    summary.markdown("**By compound**")
    summary.dataframe(
        with_badges(
            analysis.degradation_by_compound(deg)
            .reset_index()
            .rename(columns={"count": "Stints", "mean": "Mean", "median": "Median"}),
            race.compound_colors,
        ),
        hide_index=True,
        width="stretch",
        column_config={"Tyre": TYRE, "Mean": PER_LAP, "Median": PER_LAP},
    )
    table.markdown("**By stint**")
    table.dataframe(
        with_badges(deg.rename(columns={"DegPerLap": "Degradation"}), race.compound_colors),
        hide_index=True,
        width="stretch",
        column_config={"Tyre": TYRE, "Degradation": PER_LAP},
    )
    st.caption(
        "Seconds lost per lap of tyre age, after fuel correction. A negative rate means the "
        "laps got quicker through the stint, for example as the track rubbered in or "
        "traffic cleared."
    )

    row = st.selectbox(
        "Inspect stint",
        deg.index,
        format_func=lambda i: (
            f"{deg.at[i, 'Driver']} · stint {deg.at[i, 'Stint']} · {deg.at[i, 'Compound']}"
        ),
    )
    driver, stint = deg.at[row, "Driver"], deg.at[row, "Stint"]
    stint_laps = laps[(laps["Driver"] == driver) & (laps["Stint"] == stint)]
    color = race.driver_styles.get(driver, {}).get("color", FALLBACK_COLOR)
    show(analysis.stint_fit_figure(stint_laps, "FuelCorrected", color))


def model_takeaway(expected: dict[str, float]) -> str:
    phrases = listing([f"{loss:+.1f} s on the {compound}" for compound, loss in expected.items()])
    return (
        f"After {degradation.HORIZON} laps on a fresh set, the model expects lap times to "
        f"change by {phrases}."
    )


def tyre_model(race: races.Race, bundle_version: str) -> None:
    st.subheader("What would a model trained on other races expect?")
    if race.year < degradation.FIRST_SEASON:
        st.info(
            "The tyre model covers 2019 onwards, when every race used tyres named soft, "
            "medium and hard."
        )
        return
    metrics = data.tyre_metrics(bundle_version)
    curves = degradation.race_curves(data.tyre_curves(bundle_version), race.year, race.round_number)
    if metrics is None or curves.empty:
        st.info("The tyre model hasn't been trained on this race yet.")
        return

    losses = degradation.stint_losses(race.laps, race.total_laps)
    used = [
        compound for compound in degradation.DRY_COMPOUNDS if compound in set(losses["Compound"])
    ]
    curves = curves[curves["Compound"].isin(used or degradation.DRY_COMPOUNDS)]
    if expected := degradation.loss_after(curves):
        st.markdown(model_takeaway(expected))
    show(analysis.tyre_model_figure(curves, losses, race.compound_colors))
    mae = metrics["mae"]
    st.caption(
        "Lines come from a gradient-boosted model trained on dry-tyre stints from "
        f"{metrics['first_season']}–{metrics['last_season']} races, with this race held out. "
        "Dots are this race's laps. On races it hasn't seen, the model is off by "
        f"{mae['model']:.2f} s a lap on average, against {mae['baseline']:.2f} s for a "
        f"straight line per compound and {mae['no_wear']:.2f} s for assuming no wear."
    )


def strategy_tab(race: races.Race, clean: pd.DataFrame, bundle_version: str) -> None:
    tyre_strategy(race)
    st.divider()
    pit_stops_section(race)
    st.divider()
    wear_rates(race, clean)
    st.divider()
    tyre_model(race, bundle_version)
