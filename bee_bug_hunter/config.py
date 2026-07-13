"""Shared constants used across main.py, tui.py, manager.py, and orchestrator.py.
Import from here instead of re-declaring any of these locally.
"""

DEFAULT_MANIFEST = "bee_bug_hunter/flows_manifest.yaml"

# flows_manifest.yaml's `kind:` default -- both manager.build_supervisor's
# flow_kind param and orchestrator.py's flow_cfg.get("kind", ...) read this.
DEFAULT_FLOW_KIND = "ui"

# Maps an LLM_PROVIDER value (see llm.py:get_chat_model()) to the .env var
# holding that provider's model name -- used by tui.py's Agent Config panel.
LLM_MODEL_ENV_VAR = {
    "ollama": "OLLAMA_MODEL",
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "claude_cli": "CLAUDE_CLI_MODEL",
}

DEFAULT_LLM_PROVIDER = "ollama"

# Non-secret LLM defaults -- llm.py:get_chat_model() falls back to these when
# the matching env var isn't set. Real credentials (OPENAI_API_KEY,
# ANTHROPIC_API_KEY) stay .env-only and are never given a default here.
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# Ollama-only: context window size in tokens, forwarded to Ollama's
# options.num_ctx. Not applicable to the other providers.
DEFAULT_OLLAMA_NUM_CTX = 16384
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_CLAUDE_CLI_MODEL = "sonnet"

DEFAULT_LOG_FILE = "logs/bee_bug_hunter.jsonl"
DEFAULT_LOG_LEVEL = "INFO"

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

# Fallback when a flow's docker_host: override isn't set (see flows_manifest.yaml's
# header comment) -- DockerLogCaptureTool doesn't set DOCKER_HOST at all in this
# case; "local" here is purely a display label, not a value passed to Docker.
APP_DOCKER_CONN = {
    "host": "local",
}
