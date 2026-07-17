"""Regression test for the breadcrumb-staleness bug found while grooming
task:99d98198: the old update_breadcrumb()/Header.sub_title approach only
refreshed on_mount, so popping ResultsScreen back to AnomalyScreen left the
breadcrumb showing the stale "... > Results" trail. BreadcrumbBar (task:e6b4e888)
fixes this by composing the active chip fresh at render time instead of
mutating state imperatively -- this test exercises exactly that pop path.
"""
import pytest

from bee_bug_hunter.tui import AnomalyScreen, BugHunterApp, FlowSelectScreen, ResultsScreen
from bee_bug_hunter.tui_widgets import BreadcrumbBar, STEP_NAMES


def _active_chip_label(pilot) -> str:
    bar = pilot.app.screen.query_one(BreadcrumbBar)
    chip = bar.query_one(".breadcrumb-chip.active")
    return str(chip.render())


@pytest.mark.asyncio
async def test_breadcrumb_reflects_current_step_and_survives_pop():
    app = BugHunterApp()
    async with app.run_test() as pilot:
        app.push_screen(FlowSelectScreen(kind="ui"))
        await pilot.pause()
        assert _active_chip_label(pilot) == f"1 · {STEP_NAMES[0]}"

        app.push_screen(AnomalyScreen())
        await pilot.pause()
        assert _active_chip_label(pilot) == f"2 · {STEP_NAMES[1]}"

        app.push_screen(ResultsScreen())
        await pilot.pause()
        assert _active_chip_label(pilot) == f"3 · {STEP_NAMES[2]}"

        # The regression case: popping back to AnomalyScreen must not leave
        # the "Results" chip active -- BreadcrumbBar is recomposed on push,
        # not mutated in place, so there's nothing to go stale here, but a
        # regression to the old sub_title-mutation approach would fail this.
        app.pop_screen()
        await pilot.pause()
        assert _active_chip_label(pilot) == f"2 · {STEP_NAMES[1]}"
