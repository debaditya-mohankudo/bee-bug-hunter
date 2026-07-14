"""Shared constants used across main.py, tui.py, manager.py, and orchestrator.py.
Import from here instead of re-declaring any of these locally.
"""

DEFAULT_MANIFEST = "bee_bug_hunter/manifest.yaml"

# manifest.yaml's `kind:` default -- both manager.build_supervisor's
# flow_kind param and orchestrator.py's flow_cfg.get("kind", ...) read this.
DEFAULT_FLOW_KIND = "ui"

# Maps an LLM_PROVIDER value (see llm.py:get_chat_model()) to the .env var
# holding that provider's model name -- used by tui.py's Agent Config panel.
LLM_MODEL_ENV_VAR = {
    "ollama": "OLLAMA_MODEL",
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "claude_cli": "CLAUDE_CLI_MODEL",
    "copilot_cli": "COPILOT_CLI_MODEL",
}

DEFAULT_LLM_PROVIDER = "ollama"

# Non-secret LLM defaults -- llm.py:get_chat_model() falls back to these when
# the matching env var isn't set. Real credentials (OPENAI_API_KEY,
# ANTHROPIC_API_KEY) stay .env-only and are never given a default here.
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# Ollama-only: context window size in tokens, forwarded to Ollama's
# options.num_ctx. Not applicable to the other providers.
DEFAULT_OLLAMA_NUM_CTX = 16384
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_CLAUDE_CLI_MODEL = "sonnet"
DEFAULT_COPILOT_CLI_MODEL = "claude-sonnet-4.5"

DEFAULT_LOG_FILE = "logs/bee_bug_hunter.jsonl"
DEFAULT_LOG_LEVEL = "INFO"

# claude_cli only: where role -> claude -p session_id is persisted (see
# claude_cli_llm.py's ClaudeCLIChatModel.for_role()), so a session survives process
# restarts instead of living only in an in-memory dict that resets every run.
DEFAULT_CLAUDE_CLI_SESSION_STORE = ".claude_cli_sessions.json"

# copilot_cli only: where role -> `copilot -p` session_id is persisted (see
# copilot_cli_llm.py's CopilotCLIChatModel.for_role()) -- same crash/restart
# rationale as DEFAULT_CLAUDE_CLI_SESSION_STORE above.
DEFAULT_COPILOT_CLI_SESSION_STORE = ".copilot_cli_sessions.json"

# Where run_flow_once() writes one markdown report per flow run (see reports.py).
DEFAULT_REPORTS_DIR = "reports"

# Fallback values when a MYSQL_* env var (or a per-flow mysql: override) isn't set.
APP_DB_CONN = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "",
}

# Fallback when a flow's docker_host: override isn't set (see manifest.yaml's
# header comment) -- DockerLogCaptureTool doesn't set DOCKER_HOST at all in this
# case; "local" here is purely a display label, not a value passed to Docker.
APP_DOCKER_CONN = {
    "host": "local",
}
