"""Shared Textual building blocks, vendored from the house TUI style used
by docker_log_analyzer/tui_widgets.py and SeniorDevAgent's tui/widgets.py.

Vendored rather than imported cross-repo: this repo has no shared dependency
on those packages. Keep this file's contents identical to the other repos'
copies when updating any of them.
"""
from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header, RichLog, Static


def bordered(widget: Widget, title: str) -> Widget:
    """Set `widget.border_title` and return it, so every bordered picker/
    content box in every screen gets its purpose labeled the same way,
    inline in a `yield` or `with` statement instead of a separate
    query_one(...).border_title = ... follow-up line."""
    widget.border_title = title
    return widget


def step_prefix(index: int, total: int) -> str:
    """"[2/6] " style progress counter for a multi-step intake flow."""
    return f"[{index + 1}/{total}] "


# Event-kind -> (icon, rich markup color) for EventFeed.write_event.
_EVENT_STYLES: dict[str, tuple[str, str]] = {
    "tool_call": ("▶", "bold cyan"),
    "tool_done": ("✓", "green"),
    "tool_crashed": ("✗", "bold red"),
    "context_log": ("", "dim"),
}


class EventFeed(RichLog):
    """RichLog specialized for a live progress-event feed — color/icon-codes
    each event kind so a run can be scanned at a glance, and escapes every
    interpolated field via rich.markup.escape first.

    Caller supplies already-formatted, already-escaped text per event; kind
    controls the icon/color prefix. Falls back to plain text for an
    unrecognized kind rather than raising.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("highlight", False)
        kwargs.setdefault("markup", True)
        kwargs.setdefault("wrap", True)
        super().__init__(*args, **kwargs)

    @staticmethod
    def escape(text: str) -> str:
        return escape_markup(text)

    def write_event(self, kind: str, text: str) -> None:
        icon, color = _EVENT_STYLES.get(kind, ("", ""))
        prefix = f"[{color}]{icon}[/] " if icon else (f"[{color}]" if color else "")
        suffix = "[/]" if color and not icon else ""
        self.write(f"{prefix}{text}{suffix}")


# Step names for the 3-screen flow-run flow (Select -> Anomalies -> Results).
# HomeScreen is a hub, not a stage in this flow, so it has no entry here.
STEP_NAMES = ["Select Flows", "Anomaly Signals", "Results"]


class BreadcrumbBar(Horizontal):
    """"1 · Select Flows › 2 · Anomaly Signals › 3 · Results" stepper bar,
    current step highlighted — composed fresh into a screen's compose() and
    keyed off a current_index passed at construction time.

    Deliberately NOT driven by App.sub_title mutated imperatively from
    on_mount/on_screen_resume (the prior approach in this repo): a screen
    that's popped back to rather than freshly pushed may never re-fire
    on_screen_resume reliably across Textual versions, leaving an imperative
    breadcrumb stale. Composing the bar fresh at render time sidesteps that
    lifecycle-hook dependency entirely -- there's nothing to go stale.
    """

    def __init__(self, current_index: int) -> None:
        self._current_index = current_index
        super().__init__(classes="breadcrumb-bar")

    def compose(self) -> ComposeResult:
        for i, name in enumerate(STEP_NAMES):
            classes = "breadcrumb-chip active" if i == self._current_index else "breadcrumb-chip"
            yield Static(f"{i + 1} · {name}", classes=classes)
            if i < len(STEP_NAMES) - 1:
                yield Static("›", classes="breadcrumb-sep")


class CustomScreen(Screen):
    """Base class for every screen in this app -- factors out the
    `yield Header(...)` / `yield Footer()` pair every compose() otherwise
    hand-repeats at its start/end, and gives flow-run screens a one-line way
    to include a correctly-highlighted BreadcrumbBar via compose_head(step_index).

    Deliberately NOT a place for shared business logic or a full compose()
    template -- screens differ too much in what follows (buttons, tables,
    scrollable panels) for one template to fit all of them.
    """

    @staticmethod
    def compose_head(step_index: int | None = None) -> ComposeResult:
        yield Header(show_clock=True)
        if step_index is not None:
            yield BreadcrumbBar(step_index)

    @staticmethod
    def compose_foot() -> ComposeResult:
        yield Footer()
