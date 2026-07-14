"""Tests for MySQLQueryTool's EXPLAIN cache: a query PLAN (index used, access
type, estimated rows) is driven by schema/index/table stats, not live row
content, so caching it across a batch pass is safe -- unlike a plain SELECT's
data, which can legitimately change between calls. The cache is module-level
(shared by every MySQLQueryTool instance/flow in the same run_batch_once
pass), so each test clears it via clear_explain_cache() for isolation.
pymysql is monkeypatched so no live MySQL is needed.
"""
from unittest.mock import MagicMock, patch

import pytest

from bee_bug_hunter.tools.mysql_tool import MySQLQueryTool, RunQueryInput, clear_explain_cache


@pytest.fixture(autouse=True)
def _clean_explain_cache():
    clear_explain_cache()
    yield
    clear_explain_cache()


def _fake_connect(rows: list[dict]):
    """Returns a MagicMock standing in for pymysql.connect(...)'s connection,
    wired so `with conn.cursor() as cur: cur.execute(...); cur.fetchmany(200)`
    returns `rows`."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchmany.return_value = rows
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


async def _current_cache_size() -> int:
    # clear_explain_cache() rebinds the module-level name to a fresh instance,
    # so re-import it fresh via the module rather than the name captured at
    # import time in this test file.
    from bee_bug_hunter.tools import mysql_tool

    return await mysql_tool._explain_cache.size()


@pytest.mark.asyncio
async def test_explain_cache_hit_skips_pymysql_connect():
    tool = MySQLQueryTool()
    conn, cursor = _fake_connect([{"id": 1, "select_type": "SIMPLE", "table": "orders", "key": None}])

    with patch("bee_bug_hunter.tools.mysql_tool.pymysql.connect", return_value=conn) as mock_connect:
        first = await tool._run(RunQueryInput(query="EXPLAIN SELECT * FROM orders"), None, None)
        assert mock_connect.call_count == 1

        second = await tool._run(RunQueryInput(query="EXPLAIN SELECT * FROM orders"), None, None)
        # Second call is a cache hit -- pymysql.connect must not be called again.
        assert mock_connect.call_count == 1

    assert first.get_text_content() == second.get_text_content()


@pytest.mark.asyncio
async def test_explain_cache_miss_populates_cache():
    tool = MySQLQueryTool()
    conn, _cursor = _fake_connect([{"id": 1}])

    with patch("bee_bug_hunter.tools.mysql_tool.pymysql.connect", return_value=conn):
        assert await _current_cache_size() == 0
        await tool._run(RunQueryInput(query="EXPLAIN SELECT * FROM orders"), None, None)
        assert await _current_cache_size() == 1


@pytest.mark.asyncio
async def test_two_tool_instances_share_the_module_level_cache():
    """Simulates the real topology: DB Query Agent's and SQL Performance
    Agent's MySQLQueryTool are separate instances, but both must hit the same
    cache -- this is the entire point of making it module-level rather than
    per-instance."""
    db_query_agent_tool = MySQLQueryTool()
    sql_perf_agent_tool = MySQLQueryTool()
    conn, _cursor = _fake_connect([{"id": 1}])

    with patch("bee_bug_hunter.tools.mysql_tool.pymysql.connect", return_value=conn) as mock_connect:
        first = await db_query_agent_tool._run(RunQueryInput(query="EXPLAIN SELECT * FROM orders"), None, None)
        second = await sql_perf_agent_tool._run(RunQueryInput(query="EXPLAIN SELECT * FROM orders"), None, None)

    # Second instance's identical EXPLAIN is a cache hit -- only one real call,
    # and both instances see the same cached result.
    assert mock_connect.call_count == 1
    assert first.get_text_content() == second.get_text_content()


@pytest.mark.asyncio
async def test_plain_select_never_touches_explain_cache():
    tool = MySQLQueryTool()
    conn, _cursor = _fake_connect([{"id": 1}])

    with patch("bee_bug_hunter.tools.mysql_tool.pymysql.connect", return_value=conn) as mock_connect:
        await tool._run(RunQueryInput(query="SELECT * FROM orders"), None, None)
        await tool._run(RunQueryInput(query="SELECT * FROM orders"), None, None)

    # Not an EXPLAIN -- both calls must hit the DB, no caching applied.
    assert mock_connect.call_count == 2
    assert await _current_cache_size() == 0


@pytest.mark.asyncio
async def test_failed_explain_is_not_cached():
    # _query_sync's try/except only wraps cursor.execute (not pymysql.connect
    # itself), so a query-execution failure -- not a connection failure -- is
    # what actually reaches the {"error": ...} JSON path this test targets.
    tool = MySQLQueryTool()
    conn = MagicMock()
    cursor = MagicMock()
    cursor.execute.side_effect = RuntimeError("table doesn't exist")
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch("bee_bug_hunter.tools.mysql_tool.pymysql.connect", return_value=conn):
        result = await tool._run(RunQueryInput(query="EXPLAIN SELECT * FROM missing_table"), None, None)

    assert "table doesn't exist" in result.get_text_content()
    assert await _current_cache_size() == 0


@pytest.mark.asyncio
async def test_explain_cache_is_keyed_on_exact_query_text():
    tool = MySQLQueryTool()
    conn, _cursor = _fake_connect([{"id": 1}])

    with patch("bee_bug_hunter.tools.mysql_tool.pymysql.connect", return_value=conn) as mock_connect:
        await tool._run(RunQueryInput(query="EXPLAIN SELECT * FROM orders"), None, None)
        await tool._run(RunQueryInput(query="EXPLAIN SELECT * FROM users"), None, None)

    # Different query text -- both are cache misses, both hit the DB.
    assert mock_connect.call_count == 2
    assert await _current_cache_size() == 2


@pytest.mark.asyncio
async def test_explain_cache_does_not_collide_across_different_databases():
    """Two MySQLQueryTool instances pointed at different databases (as two
    flows in the same batch pass could be, via manifest.yaml's mysql:
    override) must not share a cache entry for the same query text."""
    tool_db_a = MySQLQueryTool(database="app_a")
    tool_db_b = MySQLQueryTool(database="app_b")
    conn, _cursor = _fake_connect([{"id": 1}])

    with patch("bee_bug_hunter.tools.mysql_tool.pymysql.connect", return_value=conn) as mock_connect:
        await tool_db_a._run(RunQueryInput(query="EXPLAIN SELECT * FROM orders"), None, None)
        await tool_db_b._run(RunQueryInput(query="EXPLAIN SELECT * FROM orders"), None, None)

    # Same query text, different database -- both must be real cache misses.
    assert mock_connect.call_count == 2
    assert await _current_cache_size() == 2


@pytest.mark.asyncio
async def test_write_keyword_refusal_still_works_and_is_not_cached():
    tool = MySQLQueryTool()
    result = await tool._run(RunQueryInput(query="DROP TABLE orders"), None, None)
    assert "refused" in result.get_text_content()
    assert await _current_cache_size() == 0
