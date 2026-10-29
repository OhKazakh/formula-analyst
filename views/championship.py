import base64
import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src import championship, races, seasons
from views import data, logos, theme
from views.links import requested, requested_int
from views.state import favourite, pick_favourite, with_favourite
from views.teammates import teammates_view
from views.text import surname

PROGRESSION_DRIVERS = 6
VIEWS = ["Drivers", "Teams", "Teammates"]
# The timing screen's colours: purple for the best, green for still in it.
PILLS = {
    "Champion": ("#5300A6", "#FFFFFF"),
    "Leader": ("#E10600", "#FFFFFF"),
    "In the fight": ("#28973E", "#15151E"),
    "Out": ("#67676D", "#FFFFFF"),
}
PILL_WIDTH = 96


def pill(status: str) -> str | None:
    if status not in PILLS:
        return None
    fill, ink = PILLS[status]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PILL_WIDTH}" height="22" '
        f'viewBox="0 0 {PILL_WIDTH} 22"><rect width="{PILL_WIDTH}" height="22" rx="11" '
        f'fill="{fill}"/><text x="{PILL_WIDTH / 2:g}" y="15" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="700" '
        f'fill="{ink}">{status}</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def short_event(season: seasons.Season, round_number: int) -> str:
    return season.event_name(int(round_number)).removesuffix(" Grand Prix") + " GP"


def ordinal(place: int) -> str:
    suffix = "th" if 10 <= place % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(place % 10, "th")
    return f"{place}{suffix}"


# Teams take plural verbs, as in "Mercedes lead".
PLURALS = {
    "is": "are",
    "leads": "lead",
    "needs": "need",
    "gains": "gain",
    "outscores": "outscore",
    "doesn't": "don't",
    "drops": "drop",
}


@dataclass(frozen=True)
class TitleRace:
    """What the rest of the season holds, shared by every line about the title."""

    outlook: pd.DataFrame
    season: seasons.Season
    after: int
    key: str
    remaining: int
    rounds: int
    gains: dict[int, float] | None
    chances: pd.Series

    @property
    def teams(self) -> bool:
        return self.key == "Team"

    def verb(self, singular: str) -> str:
        return PLURALS.get(singular, singular) if self.teams else singular

    def called(self, index: int) -> str:
        row = self.outlook.loc[index]
        return row["Team"] if self.teams else surname(row["Name"])

    def full(self, index: int) -> str:
        return self.outlook.loc[index, "Team" if self.teams else "Name"]

    @property
    def next_event(self) -> str:
        upcoming = self.season.schedule[self.season.schedule["Round"] > self.after]["Round"]
        return short_event(self.season, int(upcoming.min()))

    @property
    def last_races(self) -> str:
        return "the last race" if self.rounds == 1 else f"the last {self.rounds} races"

    @property
    def sweep(self) -> str:
        """Taking everything that's left: "winning all 5", or a one-two in every race."""
        which = {1: "the last race", 2: "both"}.get(self.rounds, f"all {self.rounds}")
        if self.teams:
            return "a one-two in the last race" if self.rounds == 1 else f"one-twos in {which}"
        return f"winning {which}"


def title_race_for(
    outlook: pd.DataFrame, season: seasons.Season, after: int, key: str, cars: int
) -> TitleRace:
    remaining = championship.remaining_points(season.schedule, season.year, after, cars)
    # Odds only mean something while the title is open.
    chances = (
        championship.title_chances(
            season.results, outlook, season.schedule, season.year, after, key
        )
        if outlook["Status"].iloc[0] == "Leader"
        else pd.Series(dtype=float)
    )
    return TitleRace(
        outlook=outlook,
        season=season,
        after=after,
        key=key,
        remaining=remaining,
        rounds=int((season.schedule["Round"] > after).sum()),
        gains=championship.clinch_next(outlook, season.schedule, season.year, after, cars),
        chances=chances,
    )


def points(value: float, unit: str = "point") -> str:
    return f"{value:g} {unit}{'' if value == 1 else 's'}"


def leader_line(race: TitleRace) -> str:
    """The table's short version of what settles the title at the next round."""
    must = {index: math.floor(-gain) + 1 for index, gain in race.gains.items() if gain <= 0}
    unless = {index: math.ceil(gain) for index, gain in race.gains.items() if gain > 0}
    codes = race.outlook[race.key]
    if must:
        index = max(must, key=must.get)
        return f"Clinches at the {race.next_event} by beating {codes[index]} by {must[index]}+"
    index = min(unless, key=unless.get)
    return f"Clinches at the {race.next_event} unless {codes[index]} gains {unless[index]}+"


def leader_condition(race: TitleRace) -> str:
    """What settles the title at the next round, spelled out."""
    me = race.called(race.outlook.index[0])
    must = {index: math.floor(-gain) + 1 for index, gain in race.gains.items() if gain <= 0}
    unless = {index: math.ceil(gain) for index, gain in race.gains.items() if gain > 0}
    win = championship.RACE_POINTS[0]
    clauses = [
        f"{me} {race.verb('outscores')} {race.called(index)} there by {amount} or more"
        for index, amount in must.items()
    ]
    if clauses:
        doesnt = race.verb("doesn't")
        clauses += [
            f"{race.called(index)} {doesnt} gain {amount} or more points on {me}"
            for index, amount in unless.items()
        ]
        return f"The title is settled at the {race.next_event} if " + _join(clauses, "and") + "."
    threats = []
    for number, (index, amount) in enumerate(sorted(unless.items(), key=lambda item: item[1])):
        rival = race.called(index)
        threat = f"{rival} {race.verb('gains')} {amount} or more"
        if not number:
            threat += f" points on {me} there"
        if not race.teams and amount <= win:
            place = championship.best_finish(win - amount, 1)
            finish = f"{ordinal(place)} or lower" if place else "out of the points"
            threat += f" (a {rival} win with {me} {finish})"
        threats.append(threat)
    return f"The title is settled at the {race.next_event} unless " + _join(threats, "or") + "."


def _join(parts: list[str], word: str) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" {word} " + parts[-1]


def allowance(race: TitleRace, allowed: float) -> str:
    """How little the leader may score, with a finishing place that fits."""
    leader = race.called(race.outlook.index[0])
    if allowed <= 0:
        return f"{leader} not scoring another point"
    text = f"{leader} scoring {points(allowed)} or fewer"
    if race.teams:
        return text
    place = championship.best_finish(allowed, race.rounds)
    if place:
        return f"{text}, such as finishing {ordinal(place)} or lower every time"
    if allowed == 1:
        return f"{text}, a single 10th place at most"
    return f"{text}, such as 10th place in {allowed:g} of them"


def outlook_line(row, race: TitleRace) -> str:
    leader = race.outlook[race.key].iloc[0]
    if row.Status == "Champion":
        return (
            f"Clinched at the {short_event(race.season, row.Clinch)}"
            if pd.notna(row.Clinch)
            else ""
        )
    if row.Status == "Leader":
        if race.gains:
            return leader_line(race)
        lead = row.Points - race.outlook["Points"].iloc[1]
        when = f" · earliest clinch: {short_event(race.season, row.Clinch)}"
        return f"Leads by {lead:g}" + (when if pd.notna(row.Clinch) else "")
    order = race.sweep[0].upper() + race.sweep[1:]
    if row.Status == "In the fight":
        if row.InControl:
            return f"Outscore {leader} by {row.Needed:g}+ · {race.sweep} is enough"
        if row.Allowed < 0:
            return f"Only a tie: {race.sweep}, {leader} scoreless"
        if row.Allowed == 0:
            return f"{order} with {leader} scoreless"
        return f"{order}, {leader} on {points(row.Allowed, 'pt')} or fewer"
    if row.Status == "Out":
        since = (
            f"Out since the {short_event(race.season, row.OutSince)}"
            if pd.notna(row.OutSince)
            else "Out"
        )
        return (
            f"{since}: {row.Short:g} behind even {race.sweep.replace('one-twos', 'with one-twos')}"
        )
    return f"{row.Gap:g} points behind" if row.Gap else ""


def chance(row, race: TitleRace) -> str:
    """The simulated chance, kept off 0% and 100% while the maths still allows either."""
    if row.Status not in ("Leader", "In the fight"):
        return ""
    share = race.chances[getattr(row, race.key)]
    if share < 0.005:
        return "<1%"
    if share > 0.995:
        return ">99%"
    return f"{share:.0%}"


def chances_note(race: TitleRace) -> str:
    if race.chances.empty:
        return ""
    return (
        f" Chances come from playing out the rest of the season {championship.SIMULATIONS:,} "
        f"times, replaying one of the last {championship.REPLAYED_RACES} races at random for "
        "each race left, and a recent sprint for each sprint. They follow current form but "
        "can't see upgrades coming."
    )


def title_paths(race: TitleRace) -> list[str]:
    outlook = race.outlook
    if outlook["Status"].iloc[0] != "Leader":
        return []
    first = outlook.index[0]
    me = race.called(first)
    lead = outlook["Points"].iloc[0] - outlook["Points"].iloc[1]
    leads = race.verb("leads")
    if lead:
        opening = f"**{race.full(first)}** {leads} by {lead:g} with {race.remaining} points left."
    else:
        second = race.called(outlook.index[1])
        opening = (
            f"**{race.full(first)}** {leads} on wins, level on points with {second}, with "
            f"{race.remaining} points left."
        )
    clinch = outlook["Clinch"].iloc[0]
    if race.gains:
        opening += " " + leader_condition(race)
    elif pd.notna(clinch):
        opening += (
            f" The earliest it can be settled is the {short_event(race.season, clinch)}, with "
            f"{me} winning every race until then and the rest scoring nothing."
        )
    paths = [opening]
    for index, row in outlook[outlook["Status"] == "In the fight"].iterrows():
        rival = race.called(index)
        behind = f"{row['Gap']:g} behind" if row["Gap"] else "level on points"
        give = "has" if race.rounds == 1 else "have"
        text = (
            f"**{race.full(index)}** {race.verb('is')} {behind}, so {race.verb('needs')} "
            f"{points(row['Needed'])} more than {me} from {race.last_races}, which {give} "
            f"{race.remaining} to give."
        )
        if row["InControl"]:
            text += f" {race.sweep.capitalize()} is enough whatever {me} {race.verb('does')}."
        elif row["Allowed"] < 0:
            text += (
                f" The best possible is a tie on points, {race.sweep} with {me} not scoring "
                "again, and then the one with more wins takes it."
            )
        else:
            text += f" That takes {race.sweep} with {allowance(race, row['Allowed'])}."
            text += f" Each point {rival} {race.verb('drops')} lowers that by one."
        if pd.notna(row["Blocker"]) and row["Blocker"] != outlook[race.key].iloc[0]:
            blocker = outlook.index[outlook[race.key] == row["Blocker"]][0]
            text += f" {race.called(blocker)} also has to finish behind {rival} from here."
        paths.append(text)
    if (outlook["Status"] == "Out").any():
        everyone = "Every other team is" if race.teams else "Everyone else is"
        paths.append(
            f"{everyone} out: even {race.sweep} would leave them behind {me}, by the amount in "
            "the table."
        )
    return paths


def title_race(
    contenders: pd.DataFrame, remaining: int, finished: bool, teams: bool = False
) -> str:
    leader = contenders.iloc[0]
    name = leader["Team"] if teams else leader["Name"]
    leads, has = ("lead", "have") if teams else ("leads", "has")
    title = "the teams' championship" if teams else "the championship"
    if finished:
        return f"**{name}** won {title} with {leader['Points']:g} points."
    if remaining == 0:
        return f"**{name}** {leads} with {leader['Points']:g} points."
    challengers = int(contenders["CanWin"].sum()) - 1
    if challengers == 0:
        return (
            f"**{name}** {has} clinched the title: nobody can catch "
            f"{leader['Points']:g} points with {remaining} still available."
        )
    rivals = ("team" if teams else "driver") + ("" if challengers == 1 else "s")
    return (
        f"**{name}** {leads} with {leader['Points']:g} points. With {remaining} points "
        f"still available, {challengers} other {rivals} can still take the title."
    )


def outlook_table(race: TitleRace, colors: dict[str, str]) -> pd.DataFrame:
    outlook, season = race.outlook, race.season
    columns = {
        "Pos": outlook["Position"],
        "Badge": [logos.badge(team, colors.get(team), season.year) for team in outlook["Team"]],
    }
    # A driver's team is already in the badge, which leaves room for the explanation.
    if race.teams:
        columns["Team"] = outlook["Team"]
    else:
        columns["Driver"] = outlook["Name"]
    columns |= {
        "Points": outlook["Points"],
        "Gap": ["–" if not gap else f"{gap:g}" for gap in outlook["Gap"]],
        "Wins": outlook["Wins"],
        "Title": [pill(status) for status in outlook["Status"]],
    }
    if not race.chances.empty:
        columns["Chance"] = [chance(row, race) for row in outlook.itertuples()]
    columns["What it takes"] = [outlook_line(row, race) for row in outlook.itertuples()]
    return pd.DataFrame(columns)


# Pixel widths keep the whole table, explanation included, on a 1440-pixel laptop screen with
# the sidebar open; the explanation gets whatever the other columns leave.
TABLE_WIDTH = 980
WIDTHS = {
    "Pos": 40,
    "Badge": 40,
    "Driver": 144,
    "Team": 116,
    "Points": 56,
    "Gap": 50,
    "Wins": 48,
    "Title": PILL_WIDTH + 8,
    "Chance": 60,
}


def table_config(columns: list[str]) -> dict:
    explanation = TABLE_WIDTH - sum(WIDTHS.get(column, 0) for column in columns)
    return {
        "Pos": st.column_config.NumberColumn(width=WIDTHS["Pos"]),
        "Badge": st.column_config.ImageColumn(" ", width=WIDTHS["Badge"]),
        "Driver": st.column_config.TextColumn(width=WIDTHS["Driver"]),
        "Team": st.column_config.TextColumn(width=WIDTHS["Team"]),
        "Points": st.column_config.Column(width=WIDTHS["Points"]),
        "Gap": st.column_config.Column(width=WIDTHS["Gap"]),
        "Wins": st.column_config.Column(width=WIDTHS["Wins"]),
        "Title": st.column_config.ImageColumn("Title", width=WIDTHS["Title"]),
        "Chance": st.column_config.TextColumn(
            width=WIDTHS["Chance"],
            help="How often the title ends up here when the rest of the season is played out",
        ),
        "What it takes": st.column_config.TextColumn(width=explanation),
    }


def show_outlook(race: TitleRace, colors: dict[str, str], marked: pd.Series) -> None:
    table = outlook_table(race, colors)
    st.dataframe(
        # A styled table shows six decimals unless told otherwise; half points stay visible.
        theme.favourite_rows(table, marked).format({"Points": "{:g}"}),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config=table_config(list(table.columns)),
    )
    paths = title_paths(race)
    if paths:
        st.markdown("\n".join(f"- {path}" for path in paths))


def drivers_view(season: seasons.Season, after: int, colors: dict[str, str]) -> None:
    tables = {
        round_number: championship.standings(season.results, round_number)
        for round_number in season.completed_rounds
        if round_number <= after
    }
    outlook = championship.title_outlook(tables, season.schedule, season.year, after, "Driver")
    table = tables[after]
    remaining = championship.remaining_points(season.schedule, season.year, after)
    finished = after == season.schedule["Round"].max()
    st.markdown(title_race(outlook, remaining, finished))
    race = title_race_for(outlook, season, after, "Driver", cars=1)
    show_outlook(race, colors, outlook["Driver"].eq(favourite()))
    st.caption(
        "Points left assume a win and, where it exists, the fastest-lap point at every remaining "
        "race, plus a win in every remaining sprint. A tie on points goes to whoever has more "
        "wins." + chances_note(race)
    )

    st.subheader("Points progression")
    leaders = with_favourite(
        table["Driver"].head(PROGRESSION_DRIVERS).tolist(), table["Driver"].tolist()
    )
    drivers = st.multiselect(
        "Drivers",
        table["Driver"].tolist(),
        default=leaders,
        key=f"progression-{season.year}-{favourite() or ''}",
    )
    st.plotly_chart(
        championship.progression_figure(
            season.results, season.schedule, drivers, season.driver_styles, after
        ),
        config={"displayModeBar": False},
    )

    st.subheader("Points per round")
    st.plotly_chart(
        championship.points_heatmap(
            season.results, season.schedule, table["Driver"].tolist(), after
        ),
        config={"displayModeBar": False},
    )
    st.caption("Race and sprint points combined. Hover a cell for the finishing positions.")


def penalties_text(penalties: dict[str, float]) -> str:
    notes = [
        f"{team}'s total includes a {abs(points):g}-point penalty"
        if points < 0
        else f"{team}'s total includes {points:g} points that weren't scored on track"
        for team, points in penalties.items()
    ]
    return " ".join(f"{note}." for note in notes)


def teams_view(season: seasons.Season, after: int, colors: dict[str, str]) -> None:
    table = championship.team_standings(season.team_standings, after)
    if table.empty:
        st.info("Team standings aren't available for this season.")
        return
    remaining = championship.remaining_points(season.schedule, season.year, after, cars=2)
    tables = {
        round_number: championship.team_standings(season.team_standings, round_number)
        for round_number in season.completed_rounds
        if round_number <= after
    }
    outlook = championship.title_outlook(
        tables, season.schedule, season.year, after, "Team", cars=2
    )
    finished = after == season.schedule["Round"].max()
    st.markdown(title_race(outlook, remaining, finished, teams=True))
    race = title_race_for(outlook, season, after, "Team", cars=2)
    show_outlook(race, colors, pd.Series(False, index=outlook.index))
    caption = (
        "The official standings. Points left assume a one-two and, where it exists, the "
        "fastest-lap point at every remaining race, plus a one-two in every remaining sprint."
    )
    penalties = championship.penalties(season.team_standings, season.results, after)
    if penalties:
        caption += " " + penalties_text(penalties)
    st.caption(caption + chances_note(race))

    st.subheader("Points progression")
    teams = table["Team"].tolist()
    chosen = st.multiselect("Teams", teams, default=teams, key=f"team_progression-{season.year}")
    st.plotly_chart(
        championship.team_progression_figure(
            season.team_standings, season.schedule, chosen, colors, after
        ),
        config={"displayModeBar": False},
    )

    st.subheader("Points per round")
    st.plotly_chart(
        championship.points_heatmap(season.results, season.schedule, teams, after, by="Team"),
        config={"displayModeBar": False},
    )
    st.caption(
        "Points scored by both cars, race and sprint combined, so penalties applied to the "
        "standings don't show here. Hover a cell for the finishing positions."
    )


def requested_view() -> str | None:
    return {label.lower(): label for label in VIEWS}.get(requested("view"))


def render() -> None:
    bundle_version = races.bundle_version()
    years = sorted(data.bundled_seasons(bundle_version), reverse=True)
    if not years:
        st.info("No championship data is bundled.")
        return
    requested_year = requested_int("season")
    year = st.sidebar.selectbox(
        "Season",
        years,
        index=years.index(requested_year) if requested_year in years else 0,
        key="championship_season",
    )
    season: seasons.Season = data.season(year, bundle_version)
    rounds = season.completed_rounds
    st.title(f"{year} Championship")
    if not rounds:
        st.info("No races have been run this season yet.")
        return

    requested_round = requested_int("round") if requested_year == year else None
    after = (
        st.select_slider(
            "Standings after",
            options=rounds,
            value=requested_round if requested_round in rounds else rounds[-1],
            format_func=lambda r: f"Round {r} · {season.event_name(r)}",
            key=f"standings_after-{year}",
        )
        if len(rounds) > 1
        else rounds[0]
    )
    drivers = championship.standings(season.results, after)["Driver"].tolist()
    liked = pick_favourite(drivers)
    colors = championship.team_colors(season.results, season.driver_styles)

    drivers_tab, teams_tab, teammates_tab = st.tabs(
        VIEWS, default=requested_view(), key="championship_view", on_change="rerun"
    )
    params = {
        "season": str(year),
        "round": str(after),
        "view": st.session_state.get("championship_view", VIEWS[0]).lower(),
    }
    st.query_params.from_dict(params | ({"driver": liked} if liked else {}))
    if drivers_tab.open:
        with drivers_tab:
            drivers_view(season, after, colors)
    if teams_tab.open:
        with teams_tab:
            teams_view(season, after, colors)
    if teammates_tab.open:
        with teammates_tab:
            teammates_view(season, after, colors, bundle_version)
