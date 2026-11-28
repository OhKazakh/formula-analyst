import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "research" / "inje-amr25" / "afterwards.py"


@pytest.fixture(scope="module")
def afterwards():
    spec = importlib.util.spec_from_file_location("inje_afterwards", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_course_a_cuts_the_full_course_short_and_keeps_the_pit_straight(afterwards):
    p = afterwards.p
    geometry = p.osm_geometry()
    loop = afterwards.course_a(geometry["course"], afterwards.a_link())
    origin = geometry["course"][0]
    full = p.resample(p.metres(geometry["course"], origin))[0][-1]
    short = p.resample(p.metres(loop, origin))[0][-1]

    assert loop[0] == loop[-1]
    assert short / full * p.INJE_LENGTH == pytest.approx(afterwards.COURSE_A_LENGTH, rel=0.02)
    straight = p.main_straight({**geometry, "course": loop}, afterwards.COURSE_A_LENGTH)
    assert straight["straight_to_pit_lane_m"] < 60


def test_the_bridge_scales_with_the_layout(afterwards):
    full = afterwards.ranges_on(1.0, 0.0)
    short = afterwards.ranges_on(0.5, 0.01)

    assert afterwards.p.lap_time(full["pole_level"][0], 1) == "1:07.8"
    assert full["demo"][0] == pytest.approx(sum(full["pole_level"]) / 2 + 8)
    assert short["pole_level"][1] == pytest.approx(full["pole_level"][1] / 2)
    assert short["amr25"][0] == pytest.approx(short["pole_level"][0] * 1.01)
    assert short["demo"][1] == pytest.approx(full["demo"][1] / 2)
