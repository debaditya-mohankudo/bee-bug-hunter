"""Formulates and runs read-only SQL queries against MySQL, given a log excerpt."""
import asyncio
import json
import logging
import os
import re
import time

import pymysql
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
# cacheability argument for all of them, see _schema_cache's comment below.
SCHEMA_INTROSPECTION_KEYWORDS = re.compile(
    r"^\s*(explain|show\s+(columns|tables|index|create\s+table)|desc(ribe)?)\b", re.IGNORECASE,
)
logger = get_logger(__name__)

# Module-level, shared by every MySQLQueryTool instance -- both the DB Query
# Agent's and SQL Performance Agent's, across every flow in the same
# orchestrator.run_batch_once pass. Schema-introspection results (a query
# PLAN's index/access-type/estimated-rows, or a table's column/index
# structure) are driven by schema state, not live row content, so they can't
# go stale within a batch pass the way a plain SELECT's *data* could --
# keyed on (resolved connection, exact query text), not just query text,
# because different flows can point at different databases (see
# manifest.yaml's mysql: override) and two flows both running
# "EXPLAIN SELECT * FROM orders" against different DBs must not collide.
# Cleared at the top of every run_batch_once pass via clear_schema_cache()
# (mirroring claude_cli_llm.py/copilot_cli_llm.py's clear_persisted_sessions()
# -- nothing should assume a fresh poll cycle's schema is unchanged from the
# last one). Plain SELECTs deliberately bypass this cache (see
# MySQLQueryTool._run): unlike schema shape, row content can legitimately
# change between calls within the same investigation.
_schema_cache: UnconstrainedCache[str] = UnconstrainedCache()


def clear_schema_cache() -> None:
    """Call at the top of every run_batch_once pass -- see _schema_cache's
    module-level comment for why. Rebinds to a fresh instance rather than
    awaiting UnconstrainedCache.clear() (async), since callers here are sync."""
    global _schema_cache
    _schema_cache = UnconstrainedCache()


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
        query = input.query
        is_schema_introspection = bool(SCHEMA_INTROSPECTION_KEYWORDS.match(query))
        connection = self._resolved_connection()
        cache_key = f"{connection['host']}:{connection['port']}/{connection['database']}::{query.strip()}"

        cached_result = await _schema_cache.get(cache_key) if is_schema_introspection else None
        if cached_result is not None:
            log(logger, logging.INFO, "mysql_query_schema_cached", query=query)
            return StringToolOutput(cached_result)

        result = await asyncio.to_thread(self._query_sync, query)

        if is_schema_introspection and json.loads(result).get("error") is None:
            await _schema_cache.set(cache_key, result)

        return StringToolOutput(result)
