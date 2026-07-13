"""Textual TUI entry point for bee_bug_hunter.

Four screens:
  1. HomeScreen     - what this app does, the crew's agent roster, and the
                       active LLM/MySQL config plus per-flow docker_host/mysql
                       overrides from manifest.yaml.
  2. FlowSelectScreen - pick which manifest flows to run, then run them
                        sequentially with a live EventFeed inline.
  3. AnomalyScreen  - deterministic anomaly signals (anomaly_detector.py) per
                       flow, computed from the real Flow Runner / Log Capturer
                       output regardless of whether the manager called its
                       optional check_anomalies tool.
  4. ResultsScreen  - per-flow Bug Analyst / SQL Performance Agent reports
                       (falling back to the manager's own summary when neither
                       specialist was escalated to).

Runs orchestrator.run_flow_once per selected flow in a worker thread (it is a
blocking sync call that runs the supervisor's own asyncio loop internally, which
must not share Textual's loop) and forwards its logging.log() calls into the
FlowSelectScreen's EventFeed via a logging.Handler, so the same JSONL-emitting
code path that feeds logs/bee_bug_hunter.jsonl also drives the live view
without any changes to orchestrator.py.

Screens 3 and 4 read their data from the richer per-flow dict run_flow_once now
returns (anomaly/bug_report/perf_report), sourced via delegation_capture.py --
see that module's docstring for why a plain tool wrap was needed to get at each
coworker's own output instead of just the manager's synthesized final answer.

Visual theme: dark terminal palette with a single cyan accent reused for the
"active/primary" signal everywhere (header title, primary buttons, focused
panel border, tool_call feed lines), plus a small fixed semantic set --
green/red/amber -- for done/error/perf signals. Every bordered content panel
carries the shared ".panel" class (see the `.add_class("panel")` calls below)
so the whole app's box style stays consistent as more screens/panels are
added; new screens should follow the same pattern rather than styling panels
one-off.
"""
import logging
import os

# Must run before any beeai_framework import (including transitively, via the
# bee_bug_hunter.* imports below): beeai_framework's own Logger class attaches
# a raw StreamHandler(sys.stdout) directly to itself at construction time,
# completely bypassing our own configure_logging() (which only controls the
# root logger) and, worse, bypassing Textual's alternate-screen rendering --
# any stray write straight to sys.stdout from a background thread races with
# Textual's own redraws and shows up as flashing/garbled terminal output.
# CONFIG.log_level is read once at import time from this env var, so setting
# it any later (e.g. in main()) would be too late for loggers already built
# during the import chain below. WARNING keeps the JSONL trail clean too.
os.environ.setdefault("BEEAI_LOG_LEVEL", "WARNING")

import yaml
from dotenv import load_dotenv
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Markdown, SelectionList, Static
from textual.widgets.selection_list import Selection

from bee_bug_hunter.agents import agent_summaries
from bee_bug_hunter.config import (
    APP_DB_CONN,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MANIFEST,
    LLM_MODEL_ENV_VAR,
)
from bee_bug_hunter.logging_config import configure_logging, get_logger, log
from bee_bug_hunter.orchestrator import run_flow_once
from bee_bug_hunter.tui_widgets import EventFeed, bordered

# Shared palette -- keep in sync with the .panel / component-class rules in
# BugHunterApp.CSS below. Defined once here so screen code (e.g. the
# Anomaly/Results cell coloring) never hardcodes a hex a second time.
ACCENT = "#22d3ee"
GREEN = "#4ade80"
RED = "#f87171"
AMBER = "#fbbf24"
MUTED = "#5a6472"

ui_logger = get_logger("bee_bug_hunter.tui")

# Textual's Header has no built-in breadcrumb; this fakes one via sub_title,
# rebuilt from the live screen stack whenever a screen is shown/resumed.
BREADCRUMB_LABELS = {
    "HomeScreen": "Home",
    "FlowSelectScreen": "Select Flows to Run",
    "AnomalyScreen": "Anomaly Signals",
    "ResultsScreen": "Results",
}


def update_breadcrumb(app: "BugHunterApp") -> None:
    # screen_stack's bottom entry is Textual's own implicit default screen, not
    # one of ours -- skip anything not in BREADCRUMB_LABELS rather than showing it.
    app.sub_title = " › ".join(
        BREADCRUMB_LABELS[s.__class__.__name__] for s in app.screen_stack if s.__class__.__name__ in BREADCRUMB_LABELS
    )


def log_ui(event: str, **fields) -> None:
    """Every screen change / button click / keypress goes through here so the
    TUI's own interaction trail lands in the same JSONL stream as flow-run
    events, instead of being invisible outside the terminal session."""
    log(ui_logger, logging.INFO, event, **fields)


class FeedLogHandler(logging.Handler):
    """Forwards structured log records to an EventFeed instead of stdout/file."""

    _KIND_BY_MESSAGE = {
        "flow_run_started": "tool_call",
        "flow_run_completed": "tool_done",
        "flow_run_skipped_after_failure": "tool_crashed",
        "supervisor_run_failed": "tool_crashed",
        "delegation_captured": "tool_call",
        "anomaly_signals_computed": "context_log",
        # API Flow Runner (Playwright)
        "playwright_flow_started": "tool_call",
        "playwright_flow_finished": "tool_done",
        "playwright_step_failed": "tool_crashed",
        "playwright_flow_file_missing": "tool_crashed",
        # API Flow Runner (requests, non-browser)
        "api_flow_started": "tool_call",
        "api_flow_finished": "tool_done",
        "api_flow_raised": "tool_crashed",
        "api_flow_not_registered": "tool_crashed",
        # Docker Log Capturer
        "docker_capture_started": "tool_call",
        "docker_capture_container_finished": "tool_done",
        "docker_capture_process_did_not_exit": "tool_crashed",
        # DB Query Agent / SQL Performance Agent
        "mysql_query_ok": "tool_done",
        "mysql_query_failed": "tool_crashed",
        "mysql_query_refused": "tool_crashed",
    }

    # Logged for the JSONL audit trail (see logging_memory.LoggingMemory) but far
    # too high-volume to forward into the live feed -- every message added to
    # any of the 6 agents' memory (system/user/tool-call/tool-result/final-answer,
    # per agent, per turn) would otherwise burst in and flood/flicker the RichLog.
    _SKIP_MESSAGES = {"memory_message_added", "memory_message_removed", "memory_reset"}

    def __init__(self, feed: EventFeed) -> None:
        super().__init__()
        self.feed = feed

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message in self._SKIP_MESSAGES:
            return
        kind = self._KIND_BY_MESSAGE.get(message, "context_log")
        extra = getattr(record, "extra_fields", None) or {}
        detail = " ".join(f"{k}={v}" for k, v in extra.items())
        text = f"{record.getMessage()} {detail}".strip()
        self.feed.write_event(kind, EventFeed.escape(text))


class HomeScreen(Screen):
    """Landing screen: explains the app and lists the crew's agent roster."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # VerticalScroll (not a plain Vertical) + the two action buttons placed
        # right after the About panel: the Agent Config panel below can grow
        # with an arbitrary number of flows, and a plain Vertical would silently
        # push "Select Flows to Run" out of the visible area with no scrollbar.
        with VerticalScroll(id="home-body"):
            yield bordered(
                Static(
                    "bee_bug_hunter drives a BeeAI supervisor (Investigation Manager) "
                    "over a batch of flows -- UI flows via Playwright (flows/<name>.yaml) or "
                    "pure JSON API flows via Python/requests (api_flows.py) -- capturing docker "
                    "logs, inspecting DB queries, and producing a root-cause bug/perf report.\n\n"
                    f"Manifest: {DEFAULT_MANIFEST}\n\n"
                    "Press 'r' for UI flows, 'a' for API flows, or click below.",
                    id="intro",
                ),
                "About",
            ).add_class("panel")
            yield Horizontal(
                Button("Select UI Flows to Run", id="select-ui-flows", variant="primary"),
                Button("Select API Flows to Run", id="select-api-flows", variant="primary"),
                id="home-flow-buttons",
            )
            yield bordered(Static(self._agents_text(), id="agents-list"), "Agents").add_class("panel")
            yield bordered(Static(self._config_text(), id="config-details"), "Agent Config").add_class("panel")
        yield Footer()

    def on_mount(self) -> None:
        update_breadcrumb(self.app)

    def on_screen_resume(self) -> None:
        update_breadcrumb(self.app)
        # Refresh in case .env/manifest.yaml were hand-edited externally
        # while this screen was in the background.
        self.query_one("#config-details", Static).update(self._config_text())

    @staticmethod
    def _agents_text() -> str:
        # Static renders Rich markup by default, so [bold] here needs no extra
        # wiring -- just bolds the role name, matching how EventFeed already
        # uses inline markup for its icon/color prefixes.
        return "\n".join(f"• [bold]{a['role']}[/bold]: {a['goal']}" for a in agent_summaries())

    def _config_text(self) -> str:
        """Read-only: surfaces the values that actually decide which LLM and
        which Docker/MySQL targets the agents will hit, plus exactly where to
        edit each one, so this is visible before a run starts instead of only
        discoverable by reading files. There's no in-app editor -- edit the
        named file/var directly and relaunch (or, for env vars, the running
        process's os.environ) to pick up the change."""
        provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
        model = os.getenv(LLM_MODEL_ENV_VAR.get(provider, ""), "")

        lines = [
            f"LLM provider: {provider} ({model})" if model else f"LLM provider: {provider}",
            "  edit: LLM_PROVIDER/*_MODEL in .env overrides bee_bug_hunter/config.py's "
            "DEFAULT_LLM_PROVIDER/DEFAULT_*_MODEL (read by llm.py:get_llm())",
            f"MySQL default: {os.getenv('MYSQL_USER', APP_DB_CONN['user'])}@{os.getenv('MYSQL_HOST', APP_DB_CONN['host'])}:"
            f"{os.getenv('MYSQL_PORT', str(APP_DB_CONN['port']))}/{os.getenv('MYSQL_DATABASE', APP_DB_CONN['database'])}",
            "  edit: MYSQL_* in .env overrides bee_bug_hunter/config.py's APP_DB_CONN "
            "(read by tools/mysql_tool.py:MySQLQueryTool._run())",
            "",
            "Per-flow overrides -- edit bee_bug_hunter/manifest.yaml's docker_host/mysql: keys:",
        ]
        for flow_cfg in self.app.manifest.get("flows", []):
            docker_host = flow_cfg.get("docker_host")
            mysql_cfg = flow_cfg.get("mysql") or {}
            if not docker_host and not mysql_cfg:
                lines.append(f"  • {flow_cfg['name']}: (none — uses defaults above)")
                continue
            override_bits = []
            if docker_host:
                override_bits.append(f"docker_host={docker_host}")
            if mysql_cfg:
                override_bits.append("mysql=" + ",".join(f"{k}={v}" for k, v in mysql_cfg.items() if k != "password"))
            lines.append(f"  • {flow_cfg['name']}: {'; '.join(override_bits)}")
        lines.append("  threaded via orchestrator.run_flow_once -> manager.build_supervisor -> agents.build_agents")
        return "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        log_ui("ui_button_pressed", screen="HomeScreen", button_id=event.button.id)
        if event.button.id == "select-ui-flows":
            self.app.start_flow_select("ui")
        elif event.button.id == "select-api-flows":
            self.app.start_flow_select("api")

    def key_r(self) -> None:
        log_ui("ui_key_pressed", screen="HomeScreen", key="r")
        self.app.start_flow_select("ui")

    def key_a(self) -> None:
        log_ui("ui_key_pressed", screen="HomeScreen", key="a")
        self.app.start_flow_select("api")


class FlowSelectScreen(Screen):
    """Pick a subset of the manifest's flows of one kind (UI or API), then run
    them sequentially.

    Both the picker and the live EventFeed are composed up front and toggled
    via `.display`, rather than pushing a separate "monitor" screen, so the
    app has exactly the four screens the user asked for while still keeping
    live progress visible during a run.
    """

    BINDINGS = [("escape", "pop_screen", "Back")]

    def __init__(self, kind: str = "ui") -> None:
        super().__init__()
        self.kind = kind

    def compose(self) -> ComposeResult:
        title = "Select UI Flows to Run" if self.kind == "ui" else "Select API Flows to Run"
        yield Header(show_clock=True)
        yield Vertical(
            bordered(SelectionList(id="flow-picker"), title).add_class("panel"),
            Button("Run Selected", id="run-selected", variant="primary"),
            bordered(EventFeed(id="event-feed"), "Live Run").add_class("panel"),
            id="flow-select-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        log_ui("ui_screen_shown", screen="FlowSelectScreen", kind=self.kind)
        update_breadcrumb(self.app)
        picker = self.query_one("#flow-picker", SelectionList)
        for flow_cfg in self.app.manifest.get("flows", []):
            if flow_cfg.get("kind", "ui") != self.kind:
                continue
            containers = ", ".join(flow_cfg.get("containers", []))
            label = f"{flow_cfg['name']}  [{containers}]" if containers else flow_cfg["name"]
            picker.add_option(Selection(label, flow_cfg["name"], True))
        self.query_one("#event-feed", EventFeed).display = False

    def on_screen_resume(self) -> None:
        update_breadcrumb(self.app)

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        log_ui("ui_selection_changed", screen="FlowSelectScreen", selected=list(event.selection_list.selected))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run-selected":
            return
        picker = self.query_one("#flow-picker", SelectionList)
        selected = list(picker.selected)
        if not selected:
            log_ui("ui_button_pressed", screen="FlowSelectScreen", button_id=event.button.id, flows=[], rejected="no_flow_selected")
            self.notify("Select at least one flow to run.", severity="warning")
            return

        log_ui("ui_button_pressed", screen="FlowSelectScreen", button_id=event.button.id, flows=selected)
        picker.display = False
        event.button.display = False
        feed = self.query_one("#event-feed", EventFeed)
        feed.display = True
        feed.border_title = "Live Run — running…"
        self.app.begin_batch_run(selected, feed)

    def action_pop_screen(self) -> None:
        log_ui("ui_key_pressed", screen="FlowSelectScreen", key="escape")
        self.app.pop_screen()


class AnomalyScreen(Screen):
    """Deterministic anomaly signals per flow, from the last completed batch."""

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("n", "show_results", "Next: Bug/Perf Report"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield bordered(DataTable(id="anomaly-table"), "Anomaly Signals").add_class("panel")
        yield Static(
            "Signals are computed from the API Flow Runner / Docker Log Capturer's own "
            "reported output (anomaly_detector.py), independent of the manager's report. "
            "Press 'n' for the bug analysis / SQL performance report.",
            id="anomaly-hint",
        )
        yield Footer()

    def on_mount(self) -> None:
        log_ui("ui_screen_shown", screen="AnomalyScreen", result_count=len(self.app.last_results))
        update_breadcrumb(self.app)
        table = self.query_one("#anomaly-table", DataTable)
        table.add_columns("Flow", "Bug", "Perf", "HTTP Errors", "Failed Steps", "Error Containers", "Slow Containers")
        for row in self.app.last_results:
            a = row.get("anomaly") or {}
            bug_signal = bool(a.get("bug_signal"))
            perf_signal = bool(a.get("perf_signal"))
            table.add_row(
                row["flow"],
                Text("yes" if bug_signal else "no", style=f"bold {RED}" if bug_signal else MUTED),
                Text("yes" if perf_signal else "no", style=f"bold {AMBER}" if perf_signal else MUTED),
                str(len(a.get("http_errors", []))),
                str(len(a.get("failed_steps", []))),
                ", ".join(a.get("error_containers", [])) or "-",
                ", ".join(a.get("slow_containers", [])) or "-",
            )

    def action_pop_screen(self) -> None:
        log_ui("ui_key_pressed", screen="AnomalyScreen", key="escape")
        self.app.pop_screen()

    def action_show_results(self) -> None:
        log_ui("ui_key_pressed", screen="AnomalyScreen", key="n")
        self.app.push_screen(ResultsScreen())


class ResultsScreen(Screen):
    """Per-flow bug analysis / SQL performance report from the last batch."""

    BINDINGS = [("escape", "pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="results-body"):
            if not self.app.last_results:
                yield Static("No results yet.")
            for row in self.app.last_results:
                yield bordered(
                    Markdown(self._report_markdown(row)),
                    f"{row['flow']}  (run {row['run_id']})",
                ).add_class("panel")
        yield Footer()

    @staticmethod
    def _report_markdown(row: dict) -> str:
        bug = row.get("bug_report")
        perf = row.get("perf_report")
        parts = []
        if bug:
            parts.append(f"### Bug Analysis\n\n{bug}")
        if perf:
            parts.append(f"### SQL Performance\n\n{perf}")
        if not parts:
            parts.append("_No specialist was escalated to for this flow — the manager reported no actionable issue._")
        parts.append(f"---\n\n### Manager Summary\n\n{row.get('response', '')}")
        if row.get("report_path"):
            parts.append(f"_Saved to `{row['report_path']}`_")
        return "\n\n".join(parts)

    def on_mount(self) -> None:
        log_ui("ui_screen_shown", screen="ResultsScreen", result_count=len(self.app.last_results))
        update_breadcrumb(self.app)

    def action_pop_screen(self) -> None:
        log_ui("ui_key_pressed", screen="ResultsScreen", key="escape")
        self.app.pop_screen()


class BugHunterApp(App):
    """Textual app tying the four screens together and driving flow runs."""

    # App-level (not per-screen) binding, so Quit shows in the Footer and works
    # from every screen -- this is navigation chrome, not an action specific to
    # whatever screen happens to be on top.
    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    # Single cyan accent carries the "active/primary" meaning everywhere
    # (title, primary buttons, focused panel border, table header); green /
    # red / amber are the fixed done/error/perf signal colors used across
    # the feed, table cells, and report headings. Every content panel shares
    # the ".panel" class so new screens stay visually consistent for free --
    # give a new bordered widget `.add_class("panel")` and it matches.
    CSS = f"""
    Screen {{
        background: #0b0e14;
        color: #e6e9ef;
    }}
    Header {{
        background: #12161f;
        color: {ACCENT};
        text-style: bold;
    }}
    Footer {{
        background: #0d1017;
        color: #8891a1;
    }}
    Footer > .footer--key {{
        background: #1c2130;
        color: {ACCENT};
        text-style: bold;
    }}
    Button {{
        background: #1c2130;
        color: #e6e9ef;
        border: none;
        min-width: 3;
    }}
    Button:hover {{
        background: #232a3a;
    }}
    Button.-primary {{
        background: {ACCENT};
        color: #04141a;
        text-style: bold;
    }}
    Button.-primary:hover {{
        background: #67e8f9;
    }}
    .panel {{
        border: round #2a3040;
        background: #11151d;
        padding: 1 2;
    }}
    .panel:focus-within {{
        border: round {ACCENT};
    }}
    DataTable {{
        background: #11151d;
    }}
    DataTable > .datatable--header {{
        background: #161b26;
        color: {ACCENT};
        text-style: bold;
    }}
    DataTable > .datatable--cursor {{
        background: {ACCENT} 25%;
    }}
    SelectionList {{
        background: #11151d;
    }}
    /* SelectionList always renders an "X" glyph inside the toggle button for
       both selected and unselected rows -- the checked/unchecked distinction
       has to come entirely from styling here, not the glyph itself. Unselected
       is a dim X on the panel background (reads as an empty box); selected is
       a solid accent-filled box (dark X on cyan, reads as checked). */
    SelectionList > .selection-list--button {{
        color: {MUTED};
        background: #11151d;
    }}
    SelectionList > .selection-list--button-selected {{
        color: #04141a;
        background: {ACCENT};
        text-style: bold;
    }}
    Markdown {{
        background: #11151d;
    }}
    EventFeed {{
        background: #0d1119;
    }}
    #home-body {{ padding: 1 2; }}
    #intro {{ padding: 1; }}
    #home-flow-buttons {{ height: auto; margin-bottom: 1; }}
    #home-flow-buttons Button {{ margin-right: 2; }}
    #agents-list {{ padding: 1; }}
    #config-details {{ padding: 1; }}
    #flow-select-body {{ padding: 1 2; }}
    #anomaly-hint {{ padding: 0 2; color: {MUTED}; }}
    #results-body {{ padding: 1 2; }}
    """
    TITLE = "Bee Bug Hunter"

    def __init__(self, manifest_path: str = DEFAULT_MANIFEST) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        with open(manifest_path) as f:
            self.manifest: dict = yaml.safe_load(f)
        self.last_results: list[dict] = []
        self._feed_handler: logging.Handler | None = None
        self._selected_flow_names: list[str] = []

    def on_mount(self) -> None:
        log_ui("ui_screen_shown", screen="HomeScreen")
        self.push_screen(HomeScreen())

    def start_flow_select(self, kind: str = "ui") -> None:
        log_ui("ui_navigate", to="FlowSelectScreen", kind=kind)
        self.push_screen(FlowSelectScreen(kind=kind))

    def action_quit(self) -> None:
        log_ui("ui_key_pressed", screen=self.screen.__class__.__name__, key="q")
        self.exit()

    def begin_batch_run(self, flow_names: list[str], feed: EventFeed) -> None:
        """Called by FlowSelectScreen once the user picks flows and presses Run,
        so the worker only starts once the live-feed widget actually exists."""
        log_ui("ui_batch_run_requested", flows=flow_names)
        self._selected_flow_names = flow_names
        self._feed_handler = FeedLogHandler(feed)
        logging.getLogger().addHandler(self._feed_handler)

        self.run_worker(self._run_batch, thread=True, exclusive=True)

    def _run_batch(self) -> None:
        duration_seconds = self.manifest.get("duration_seconds", 30)
        flows_by_name = {f["name"]: f for f in self.manifest.get("flows", [])}
        results = []
        try:
            for name in self._selected_flow_names:
                flow_cfg = flows_by_name.get(name)
                if flow_cfg is None:
                    continue
                try:
                    results.append(run_flow_once(flow_cfg, duration_seconds))
                except Exception:
                    # one flow's failure shouldn't take down the rest of the
                    # selected batch; run_flow_once already logs the root cause,
                    # but log here too so the TUI's own event trail shows which
                    # selected flow got skipped and why, not just that a flow failed.
                    log_ui("ui_flow_run_skipped", flow=name)
                    ui_logger.exception("flow run raised in TUI batch worker")
                    continue
        finally:
            self.last_results = results
            if self._feed_handler is not None:
                logging.getLogger().removeHandler(self._feed_handler)
                self._feed_handler = None
            self.call_from_thread(self._show_anomalies)

    def _show_anomalies(self) -> None:
        log_ui("ui_batch_run_finished", result_count=len(self.last_results))
        if isinstance(self.screen, FlowSelectScreen):
            self.screen.query_one("#event-feed", EventFeed).border_title = "Live Run — done"
        self.push_screen(AnomalyScreen())


def main() -> None:
    # Mirrors main.py's setup: running `python -m bee_bug_hunter.tui` directly
    # (rather than through main.py) previously skipped both of these, so .env
    # overrides silently had no effect and nothing was ever written to
    # logs/bee_bug_hunter.jsonl -- a live TUI run left no JSONL trail at all.
    load_dotenv()
    log_file = os.getenv("LOG_FILE", DEFAULT_LOG_FILE)
    configure_logging(
        level=getattr(logging, os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(), logging.INFO),
        log_file=log_file if log_file.lower() != "none" else None,
    )
    BugHunterApp().run()


if __name__ == "__main__":
    main()
