"""Tests for bee_bug_hunter.tools.read_source_tool.ReadSourceFileTool. Docker
subprocess calls are monkeypatched so no live container is needed; the scratch
copy is faked by writing files directly to where `docker cp` would have placed
them, since `_ensure_source_copy_sync` short-circuits on an existing dest dir.
"""
import pytest

from bee_bug_hunter.tools.read_source_tool import (
    ReadSourceFileInput,
    ReadSourceFileTool,
    SCRATCH_ROOT,
    clear_scratch_cache,
)


@pytest.fixture(autouse=True)
def _clean_scratch():
    clear_scratch_cache()
    yield
    clear_scratch_cache()


@pytest.fixture
def fake_container_source():
    """Pre-seeds SCRATCH_ROOT/<container> as if `docker cp` already ran, so
    _ensure_source_copy_sync's cache-hit path is exercised without shelling out."""
    dest = SCRATCH_ROOT / "demo-api"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "app.py").write_text("print('hello from app.py')\n")
    return dest


@pytest.mark.asyncio
async def test_reads_file_from_cached_scratch_copy(fake_container_source):
    tool = ReadSourceFileTool()
    result = await tool._run(ReadSourceFileInput(container="demo-api", path="app.py"), None, None)
    assert "hello from app.py" in result.get_text_content()


@pytest.mark.asyncio
async def test_rejects_path_traversal(fake_container_source):
    tool = ReadSourceFileTool()
    result = await tool._run(ReadSourceFileInput(container="demo-api", path="../../etc/passwd"), None, None)
    assert "error" in result.get_text_content()
    assert "escapes" in result.get_text_content()


@pytest.mark.asyncio
async def test_missing_file_returns_error_not_raise(fake_container_source):
    tool = ReadSourceFileTool()
    result = await tool._run(ReadSourceFileInput(container="demo-api", path="nope.py"), None, None)
    assert "error" in result.get_text_content()
    assert "not found" in result.get_text_content()


@pytest.mark.asyncio
async def test_inspect_failure_surfaces_as_error(monkeypatch):
    def fake_run_docker(args, env):
        class _Result:
            returncode = 1
            stderr = "no such container"
            stdout = ""

        return _Result()

    monkeypatch.setattr("bee_bug_hunter.tools.read_source_tool._run_docker", fake_run_docker)
    tool = ReadSourceFileTool()
    result = await tool._run(ReadSourceFileInput(container="missing-container", path="app.py"), None, None)
    assert "error" in result.get_text_content()
    assert "docker inspect failed" in result.get_text_content()
