"""Starts/stops background `docker logs -f` capture for a set of containers."""
import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from pydantic import BaseModel, Field

from bee_bug_hunter.config import APP_DOCKER_CONN
from bee_bug_hunter.logging_config import get_logger, log

CAPTURE_DIR = Path(__file__).parent.parent.parent / ".captures"
logger = get_logger(__name__)


class CaptureDockerLogsInput(BaseModel):
    containers: str = Field(..., description="Comma-separated container names to capture, e.g. 'api,worker'")
    duration_seconds: int = Field(..., description="How long to capture logs for, in seconds")
    run_name: str = Field(..., description="Label for this capture run, used to name output files")


def _capture_sync(containers: str, duration_seconds: int, run_name: str, docker_host: str | None) -> str:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    container_list = [c.strip() for c in containers.split(",") if c.strip()]
    log(
        logger, logging.INFO, "docker_capture_started",
        containers=container_list, duration_seconds=duration_seconds, run_name=run_name,
        docker_host=docker_host or APP_DOCKER_CONN["host"],
    )

    env = os.environ.copy()
    if docker_host:
        env["DOCKER_HOST"] = docker_host

    procs = {}
    out_files = {}
    for container in container_list:
        out_path = CAPTURE_DIR / f"{run_name}_{container}.log"
        out_files[container] = out_path
        f = out_path.open("w")
        # `--since 5m` (not `0s`) so this still captures a flow's output even when the
        # flow ran and finished before this tool was invoked -- the supervisor delegates
        # Flow Runner and Log Capturer sequentially, not concurrently, so "from now on"
        # would miss logs the flow already emitted.
        proc = subprocess.Popen(
            ["docker", "logs", "-f", "--since", "5m", container],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        procs[container] = (proc, f)

    time.sleep(duration_seconds)

    for container, (proc, f) in procs.items():
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log(logger, logging.WARNING, "docker_capture_process_did_not_exit", container=container)
            proc.kill()
        f.close()

    results = {}
    for container, path in out_files.items():
        content = path.read_text()[-8000:]
        results[container] = {"log_path": str(path), "content": content}
        log(logger, logging.INFO, "docker_capture_container_finished", container=container, log_path=str(path), bytes_captured=len(content))

    return json.dumps(results, indent=2)


class DockerLogCaptureTool(Tool[CaptureDockerLogsInput, ToolRunOptions, StringToolOutput]):
    name = "capture_docker_logs"
    description = (
        "Captures `docker logs` output for the given containers over a fixed window. "
        "Call this to record what containers logged while a flow was being executed elsewhere. "
        "Returns the file paths of the captured logs and their contents."
    )
    input_schema = CaptureDockerLogsInput

    def __init__(self, docker_host: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # Per-flow override (see manifest.yaml's docker_host comment) for which
        # docker engine `docker logs` talks to; None means "whatever this process's
        # ambient docker context/DOCKER_HOST already points at".
        self.docker_host = docker_host

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "capture_docker_logs"], creator=self)

    async def _run(self, input: CaptureDockerLogsInput, options, context) -> StringToolOutput:
        result = await asyncio.to_thread(
            _capture_sync, input.containers, input.duration_seconds, input.run_name, self.docker_host,
        )
        return StringToolOutput(result)
