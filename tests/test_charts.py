import pytest

from src import charts


def test_contrast_matches_wcag_extremes():
    assert charts.contrast("#FFFFFF", "#000000") == pytest.approx(21.0)
    assert charts.contrast("#E10600", "#FFFFFF") == pytest.approx(4.97, abs=0.01)


@pytest.mark.parametrize(
    ("color", "expected"),
    [("#ffffff", True), ("#fff500", True), ("#1c1c25", True), ("#ff8000", False)],
)
def test_hard_to_see_on_light_or_dark_backgrounds(color, expected):
    assert charts.hard_to_see(color) is expected
