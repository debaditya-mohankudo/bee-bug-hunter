"""Reads application source files out of the running container itself, rather than a
hardcoded local checkout -- a real debugging target may run on a remote host with no
local source available, but the container that produced the logs always has both the
code and the process. Discovers the source root via `docker inspect` (WorkingDir),
copies it out with `docker cp` into a scratch dir, and serves reads from that copy."""
import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from pydantic import BaseModel, Field

from bee_bug_hunter.logging_config import get_logger, log

SCRATCH_ROOT = Path(tempfile.gettempdir()) / "bee_bug_hunter_source_copies"
MAX_LINES = 400

logger = get_logger(__name__)


class ReadSourceFileInput(BaseModel):
    container: str = Field(..., description="Name of the running container whose source code to read")
    path: str = Field(..., description="Path to the file, relative to the container's WORKDIR, e.g. 'app.py'")


def _run_docker(args: list[str], env: dict | None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, env=env)


def _ensure_source_copy_sync(container: str, docker_host: str | None) -> Path | str:
    """Returns the local scratch dir holding the container's source, or an error string."""
    dest = SCRATCH_ROOT / container
    if dest.exists():
        return dest

    import os
    env = os.environ.copy()
    if docker_host:
        env["DOCKER_HOST"] = docker_host

    inspect = _run_docker(["inspect", "--format", "{{.Config.WorkingDir}}", container], env)
    if inspect.returncode != 0:
        return f"docker inspect failed for container '{container}': {inspect.stderr.strip()}"
    workdir = inspect.stdout.strip()
    if not workdir:
        return f"container '{container}' has no WorkingDir set; cannot locate its source root"

    dest.parent.mkdir(parents=True, exist_ok=True)
    cp = _run_docker(["cp", f"{container}:{workdir}", str(dest)], env)
    if cp.returncode != 0:
        return f"docker cp failed for container '{container}:{workdir}': {cp.stderr.strip()}"

    log(logger, logging.INFO, "read_source_copied", container=container, workdir=workdir, dest=str(dest))
    return dest


class ReadSourceFileTool(Tool[ReadSourceFileInput, ToolRunOptions, StringToolOutput]):
    name = "read_source_file"
    description = (
        "Reads a source file from the container's own code (copied out via `docker cp` on first "
        "use). Use this to confirm a hypothesis against the real implementation -- e.g. a column "
        "name in a SQL query -- rather than only inferring root cause from logs. Path is relative "
        "to the container's WORKDIR."
    )
    input_schema = ReadSourceFileInput

    def __init__(self, docker_host: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.docker_host = docker_host

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "read_source_file"], creator=self)

    async def _run(self, input: ReadSourceFileInput, options, context) -> StringToolOutput:
        root = await asyncio.to_thread(_ensure_source_copy_sync, input.container, self.docker_host)
        if isinstance(root, str):
            return StringToolOutput(f"error: {root}")

        target = (root / input.path).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            return StringToolOutput(f"error: path '{input.path}' escapes the container's source root")

        if not target.exists():
            return StringToolOutput(f"error: '{input.path}' not found in container '{input.container}'")
        if not target.is_file():
            return StringToolOutput(f"error: '{input.path}' is not a file")

        lines = target.read_text(errors="replace").splitlines()
        truncated = len(lines) > MAX_LINES
        content = "\n".join(lines[:MAX_LINES])
        if truncated:
            content += f"\n... (truncated, {len(lines)} lines total)"
        return StringToolOutput(content)


def clear_scratch_cache() -> None:
    """Wipes SCRATCH_ROOT so cached container source copies don't leak stale
    state. Called from orchestrator.run_batch_once at the top of every batch
    pass (a container could be rebuilt/restarted with a real fix between poll
    cycles -- see that call site) and from tests, for the same isolation
    reason claude_cli_llm.py's clear_persisted_sessions() exists."""
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
