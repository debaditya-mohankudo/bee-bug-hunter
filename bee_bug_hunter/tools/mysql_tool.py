"""Formulates and runs read-only SQL queries against MySQL, given a log excerpt."""
import asyncio
import json
import logging
import os
import re
import time

import pymysql
from beeai_framework.cache.base import BaseCache
from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from pydantic import BaseModel, Field

from bee_bug_hunter.config import APP_DB_CONN
from bee_bug_hunter.logging_config import get_logger, log

WRITE_KEYWORDS = re.compile(r"^\s*(insert|update|delete|drop|alter|truncate|create)\b", re.IGNORECASE)
# Schema-introspection queries: EXPLAIN (query plan) and SHOW COLUMNS/SHOW
# TABLES/SHOW INDEX/SHOW CREATE TABLE/DESCRIBE/DESC (schema shape) all
# describe schema/index/table structure, never row content -- same
# cacheability argument for all of them, see _SchemaOnlyCache's comment below.
SCHEMA_INTROSPECTION_KEYWORDS = re.compile(
    r"^\s*(explain|show\s+(columns|tables|index|create\s+table)|desc(ribe)?)\b", re.IGNORECASE,
)
logger = get_logger(__name__)

# Key prefix _generate_key tags schema-introspection queries with, so a
# BaseCache implementation can decide cacheability from the key string alone
# (Tool.run()'s built-in cache.get/set calls have no other visibility into
# what kind of query this is).
_SCHEMA_KEY_PREFIX = "schema::"


class _SchemaOnlyCache(BaseCache[StringToolOutput]):
    """Wraps UnconstrainedCache, but only actually stores/returns entries whose
    key was tagged by MySQLQueryTool._generate_key as schema-introspection --
    every other key (plain SELECTs, refused writes) always misses. This is what
    makes BeeAI's native per-Tool caching (options={"cache": ...}, which is
    otherwise unconditional per input) safe to attach here at all: unlike
    schema shape, row content can legitimately change between calls within the
    same investigation and must never be served stale.

    Also refuses to persist a result whose JSON body carries an "error" key --
    native Tool.run() caches whatever _run() returns on any non-exception path
    (this project's tools deliberately never raise ToolError, see
    orchestrator-framework-error-boundary), so without this a transient query
    failure would otherwise get cached as permanent for the rest of the batch
    pass instead of being retried on the next call.

    Module-level singleton, shared by every MySQLQueryTool instance -- both the
    DB Query Agent's and SQL Performance Agent's, across every flow in the same
    orchestrator.run_batch_once pass -- because schema-introspection results
    aren't scoped to one agent or one flow. Reset once per batch pass via
    clear_schema_cache().
    """

    def __init__(self) -> None:
        super().__init__()
        self._inner: UnconstrainedCache[StringToolOutput] = UnconstrainedCache()

    def _cacheable(self, key: str) -> bool:
        return key.startswith(_SCHEMA_KEY_PREFIX)

    async def size(self) -> int:
        return await self._inner.size()

    async def set(self, key: str, value: StringToolOutput) -> None:
        if not self._cacheable(key):
            return
        try:
            body = json.loads(value.get_text_content())
        except (json.JSONDecodeError, TypeError):
            body = {}
        if isinstance(body, dict) and body.get("error") is not None:
            return
        await self._inner.set(key, value)

    async def get(self, key: str) -> StringToolOutput | None:
        if not self._cacheable(key):
            return None
        result = await self._inner.get(key)
        if result is not None:
            log(logger, logging.INFO, "mysql_query_schema_cached", key=key)
        return result

    async def has(self, key: str) -> bool:
        if not self._cacheable(key):
            return False
        return await self._inner.has(key)

    async def delete(self, key: str) -> bool:
        return await self._inner.delete(key)

    async def clear(self) -> None:
        await self._inner.clear()


_schema_cache = _SchemaOnlyCache()


def clear_schema_cache() -> None:
    """Call at the top of every run_batch_once pass -- see _SchemaOnlyCache's
    class docstring for why. Rebinds to a fresh instance rather than awaiting
    BaseCache.clear() (async), since callers here are sync. Every
    MySQLQueryTool constructed after this call (build_agents() runs fresh per
    flow) picks up the new instance via get_schema_cache()."""
    global _schema_cache
    _schema_cache = _SchemaOnlyCache()


def get_schema_cache() -> _SchemaOnlyCache:
    """Fetched fresh at MySQLQueryTool construction time (agents.py's
    build_agents(), called once per flow) so a clear_schema_cache() rebind
    earlier in the same batch pass is picked up by every flow's tools --
    passing a value captured at import time would pin every tool to whichever
    instance existed first and never see a later rebind."""
    return _schema_cache


class RunQueryInput(BaseModel):
    query: str = Field(..., description="A read-only SQL SELECT query to run against the configured MySQL database")


class MySQLQueryTool(Tool[RunQueryInput, ToolRunOptions, StringToolOutput]):
    name = "run_mysql_query"
    description = (
        "Executes a read-only SQL SELECT query against the app's MySQL database and returns the rows. "
        "Use this to inspect the actual data state referenced by a SQL statement found in captured logs. "
        "EXPLAIN is allowed. Write operations (INSERT/UPDATE/DELETE/DDL) are rejected."
    )
    input_schema = RunQueryInput

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        **kwargs,
    ) -> None:
        # options={"cache": ...} may already be supplied by the caller (agents.py
        # passes get_schema_cache()); default to the shared schema cache if not,
        # so any ad-hoc MySQLQueryTool() construction still gets schema caching
        # instead of silently running uncached.
        kwargs.setdefault("options", {})
        kwargs["options"].setdefault("cache", get_schema_cache())
        super().__init__(**kwargs)
        # Per-flow override (see manifest.yaml's mysql: block); any field
        # left None falls back to the matching MYSQL_* env var.
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def _resolved_connection(self) -> dict:
        return {
            "host": self.host or os.getenv("MYSQL_HOST", APP_DB_CONN["host"]),
            "port": self.port or int(os.getenv("MYSQL_PORT", str(APP_DB_CONN["port"]))),
            "database": self.database or os.getenv("MYSQL_DATABASE", APP_DB_CONN["database"]),
        }

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "run_mysql_query"], creator=self)

    def _generate_key(self, input, options=None) -> str:
        # Tagging with the resolved connection preserves the pre-migration
        # invariant: two flows both running "EXPLAIN SELECT * FROM orders"
        # against different databases (manifest.yaml's mysql: override) must
        # not collide. The schema::/norow:: prefix is what lets _SchemaOnlyCache
        # decide cacheability from the key alone, since Tool.run()'s generic
        # cache.get/set calls don't otherwise carry the query's own shape.
        query = input.query if hasattr(input, "query") else input["query"]
        connection = self._resolved_connection()
        prefix = _SCHEMA_KEY_PREFIX if SCHEMA_INTROSPECTION_KEYWORDS.match(query) else "norow::"
        return f"{prefix}{connection['host']}:{connection['port']}/{connection['database']}::{query.strip()}"

    def _query_sync(self, query: str) -> str:
        if WRITE_KEYWORDS.match(query):
            log(logger, logging.WARNING, "mysql_query_refused", reason="write_keyword_detected", query=query)
            return json.dumps({"error": "refused: only read-only SELECT queries are allowed"})

        connection = self._resolved_connection()
        conn = pymysql.connect(
            **connection,
            user=self.user or os.getenv("MYSQL_USER", APP_DB_CONN["user"]),
            password=self.password if self.password is not None else os.getenv("MYSQL_PASSWORD", APP_DB_CONN["password"]),
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
        )
        started = time.monotonic()
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchmany(200)
            elapsed_ms = round((time.monotonic() - started) * 1000, 1)
            # row contents may hold customer/PII data — log query shape and timing only, never row values.
            log(logger, logging.INFO, "mysql_query_ok", query=query, row_count=len(rows), elapsed_ms=elapsed_ms)
            return json.dumps({"query": query, "row_count": len(rows), "rows": rows}, indent=2, default=str)
        except Exception as e:
            log(logger, logging.ERROR, "mysql_query_failed", query=query, error=str(e))
            return json.dumps({"query": query, "error": str(e)})
        finally:
            conn.close()

    async def _run(self, input: RunQueryInput, options, context) -> StringToolOutput:
        # Caching (schema-introspection only, error results excluded) is now
        # handled entirely by Tool.run()'s built-in cache wrapping around this
        # method, via _generate_key + the options={"cache": _SchemaOnlyCache()}
        # passed at construction -- no manual cache check/set here anymore.
        result = await asyncio.to_thread(self._query_sync, input.query)
        return StringToolOutput(result)
