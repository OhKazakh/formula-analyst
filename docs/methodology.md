# Methodology

How the app works out the numbers it shows.

## Data

- Every race weekend from 2018 to the latest 2026 round is bundled in `data/`. Each race has the lap table with sector times and speed traps, each driver's fastest-lap telemetry, the track outline, the weather, race control messages and the list of team radio clips. Practice, qualifying and sprint sessions sit next to it with the same files, except that practice skips the telemetry and every session shares the race's track outline.
- The 2018 Italian Grand Prix has its qualifying but not the race, whose tyre data FastF1 can't process.
- Formula 1 put its timing archive behind an F1 TV login in October 2026, so sessions from the 2026 Singapore Grand Prix on come from OpenF1, which records the same live timing feed, with the official classification from Jolpica. Rebuilt that way, the 2025 Italian Grand Prix matched FastF1's lap and sector times, tyre stints, pit laps and track status on every lap, and its qualifying results to the millisecond. OpenF1 has no track status of its own, so it is rebuilt from race control messages, which can arrive a few seconds before or after the change: a safety car can start or end a lap apart from FastF1's data. OpenF1 also misses some speed trap readings and has no finish-line speed trap.
- Radio audio isn't bundled; the browser streams it from the official timing service when a clip is played.
- Each season also has its race and sprint results, the official team standings after every round and its calendar, from the Jolpica F1 API.

## Race

- **Race summary.** Safety car and virtual safety car periods are counted in the winner's laps that ran under them. Pit stops leave out pit lane entries during a red flag, when every car waits in the pit lane. Laps led and lead changes come from the running order at the end of each lap.
- **Positions** are the order in which drivers completed each lap. They match FastF1's own position data.
- **Gap to the leader** is the time between each driver and the leader crossing the timing line at the end of a lap. Safety car and virtual safety car laps are the leader's laps that ran under them.
- **Timing screen.** Each car's state is taken when it completes the chosen lap, so lapped cars show their previous lap, and the order is the order on track. A car is out once it stops completing laps before the winner takes the flag. Best laps only count laps that stood, so deleted times are left out.
- **Race replay** positions come from each lap's start and end times. Within a lap, cars follow the speed profile of the winner's fastest lap, so they slow down in corners rather than moving at constant speed. The order shown is the order on track, before any penalties. Track outlines come from the MultiViewer circuit data that FastF1 uses; the 2020 Sakhir, 2026 Spanish and 2026 Bahrain Grand Prix have none.

## Pace and telemetry

- **Fuel correction** subtracts `fuel_effect × laps remaining` from each lap time, normalising every lap to an empty tank. The default is 0.055 s per lap of fuel and can be changed in the app.
- **Pace comparison** uses representative laps within 107% of the fastest one, the same quick-lap threshold as FastF1.
- **Sectors.** A theoretical best adds up a driver's best first, second and third sectors from any accurate lap. Top speed is the highest speed trap reading of the race, and corner speed is the mean minimum speed through every corner on the driver's fastest lap.
- **Corner by corner.** Each corner covers the track halfway to its neighbours. The minimum speed is the slowest point in that stretch, and the braking point is how far before the corner marker the driver last went on the brakes before the apex. Car data is sampled about four times a second, so short brake taps can be missed.
- **Mini-sectors** split the fastest lap into 25 equal distances. A driver's time through each one comes from integrating their speed trace.

## Strategy

- **Stints follow tyre sets.** The timing data starts a new stint whenever a car passes through the pit lane, even without new tyres, as when the safety car led the field through the pit lane in the 2024 Miami sprint. A stint here only ends when the compound changes or the tyre age starts again, and only those visits count as pit stops.
- **Pit stops.** Time in the pit lane runs from the pit entry line on the in-lap to the pit exit line on the out-lap, so it includes the drive through the lane and any penalty served in the box. The cost of a stop is the in-lap plus the out-lap minus two of the driver's typical green-flag laps, and the race's pit loss is the median of those.
- **Undercuts.** A driver within 3 s of the car ahead pits first, and that car pits within the next 5 laps. The undercut worked if the driver was ahead once both had stopped.
- **Degradation** is the slope of a least-squares line through fuel-corrected lap time against tyre age, fitted per stint. Stints shorter than the configurable minimum (8 laps by default) are skipped. A straight line is a simplification: tyre warm-up at the start of a stint and the drop-off at the end are averaged into one number.
- **Tyre model.** For every dry stint from 2019 onwards, when the compounds were named soft, medium and hard, each representative lap's fuel-corrected time is compared with the median of the stint's first three laps. That gives the time lost since the stint began, which includes tyre wear and the track rubbering in. A gradient-boosted tree model (scikit-learn's `HistGradientBoostingRegressor`) predicts it from the compound, tyre age, the age of the set when the stint began, the circuit and the season. It is trained on 128,900 laps from 6,571 stints in 164 races. Cross-validated in five folds grouped by race, so every race is predicted by a model that never saw it, it is off by 0.59 s a lap on average, against 0.63 s for a straight line per compound and 0.75 s for assuming no wear. Lap-to-lap variation from traffic and driving limits how close any model can get. I also tried the race's median track temperature as a feature. At first it looked like a small gain, but that disappeared once stints followed tyre sets instead of pit lane visits: the error was 0.585 s with or without it, so the model leaves it out. The curves shown for each race come from the fold that held it out.
- **Strategy simulator.** Each compound's fresh-tyre pace comes from a least-squares fit of this race's representative laps on compound, tyre age per compound, driver and lap number, so a faster driver or a lighter car doesn't make a compound look quicker. Wear comes from the tyre model's curve for this race. Every plan with one or two stops and at least two dry compounds is timed, with stints of at least 5 laps and no longer than the model's curve reaches. Traffic, safety cars and tyre warm-up aren't modelled.

## Race predictions

- **The prediction** for each race comes from a gradient-boosted tree model (scikit-learn's `HistGradientBoostingRegressor`) that only uses what is known before the start: the qualifying position, the gap to pole in percent, the best long run in practice as percent off the quickest, and the average finishing position over the last five races for the driver and for their team. Every feature is only allowed to push a driver further back, which keeps the trees from chasing noise, and the predicted positions are ranked within each race. Grid penalties aren't in the timing data, so the model starts from the qualifying order.
- **How good it is.** Cross-validated in five folds grouped by race over 191 races since 2018, so every race is predicted by a model that never saw it, it picks the winner 53% of the time against 56% for pole position, gets two thirds of the podium right, as does the qualifying order, and is 3.39 places off on average against 3.45. Qualifying decides most of a race, so the model adds little to it on average; where it helps is with drivers who qualified out of position. For those who qualified at least five places behind their recent average finish, it is 3.9 places off against 5.0. A race that hasn't been run yet is predicted by a model trained on every race.

## Practice and qualifying

- **Qualifying parts** come from FastF1's split of the session into Q1, Q2 and Q3 (or SQ1 to SQ3). A driver's time in each part is their best lap that the stewards didn't delete, and the classification uses the time from the last part they reached.
- **Practice** is ranked by each driver's fastest lap that counted.
- **Long runs** are at least 5 unbroken green-flag laps of one stint, each within 2.5% of that stint's median lap and within 112% of the session's fastest, which leaves out cool-down and push laps. Fuel loads aren't public, so long-run pace is a guide rather than a ranking.

## Conditions

- **Weather** for each lap is the last reading from the circuit's weather station before the leader completed it.
- **Race control** messages are grouped by keyword into penalties, investigations, track limits, flags, safety car and DRS, and blue flags are left out.
- **Penalties and track limits** are read from race control's messages. Served penalties are left out, and each deleted lap time counts once for its driver.

## Championship

- **Championship standings** add up race and sprint points after each round and break ties by countback over classified finishes. They match the official standings for every bundled season.
- **Team standings** come straight from the official classification after every round rather than being added up from driver points, so penalties such as Racing Point's 15-point deduction in 2020 are included.
- **Championship after the race** compares the standings after the race with those after the previous round.
- **Who can still win** compares the leader's points with each driver's points plus the most still available: a race win (and the fastest-lap point from 2019 to 2024) for every remaining round, plus a sprint win where there is one. For teams it's a one-two at every remaining race and sprint. A driver in the fight has to outscore the leader by one point more than the gap, since a tie goes to whoever has more wins. It's in their own hands when winning every remaining race would be enough even if the closest rival finished second every time; otherwise the app gives the most the leader can score for winning everything to still be enough. When the next round can settle the title, it says what each rival has to gain on the leader there to keep it open, or by how much the leader has to beat them; otherwise it gives the earliest round the leader could clinch, by winning every race while the rest score nothing. A driver is out from the first round after which even winning everything couldn't catch the leader.
- **Title chances** come from playing out the rest of the season 10,000 times. Each remaining race replays the points of one of the last ten races, picked at random, and each remaining sprint those of a recent sprint, so form, retirements and the gaps between teams come with them. A driver who missed the latest race scores nothing from there, and a tie on points goes to whoever is ahead now. Chances show from the tenth race on, while the title is open, and stay off 0% and 100% while the maths still allows either. The swings are only those of the last ten races, and upgrades and penalties are unknown, so it can be overconfident: with six races left in 2024 it gave McLaren the teams' title at over 99%, and Ferrari finished 14 points short. Before the 2025 finale it gave Norris 59%, Verstappen 31% and Piastri 10%, and Norris won by two points.
- **Teammate head-to-heads** pair the drivers who raced for the same team at each round, so a stand-in gets a row of their own. Qualifying counts the Grand Prix sessions in which both set a time, by the official classification, and the gap is the median difference in the last part both reached, so one wet session or crash doesn't swing it. The race counts the races both were classified in, and the points include sprints.
