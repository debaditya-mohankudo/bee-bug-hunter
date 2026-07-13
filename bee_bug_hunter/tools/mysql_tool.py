"""Formulates and runs read-only SQL queries against MySQL, given a log excerpt."""
import asyncio
import json
import logging
import os
import re
import time

import pymysql
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from pydantic import BaseModel, Field

from bee_bug_hunter.config import APP_DB_CONN
from bee_bug_hunter.logging_config import get_logger, log

WRITE_KEYWORDS = re.compile(r"^\s*(insert|update|delete|drop|alter|truncate|create)\b", re.IGNORECASE)
logger = get_logger(__name__)


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
        # Per-flow override (see flows_manifest.yaml's mysql: block); any field
        # left None falls back to the matching MYSQL_* env var.
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "run_mysql_query"], creator=self)

    def _query_sync(self, query: str) -> str:
        if WRITE_KEYWORDS.match(query):
            log(logger, logging.WARNING, "mysql_query_refused", reason="write_keyword_detected", query=query)
            return json.dumps({"error": "refused: only read-only SELECT queries are allowed"})

        conn = pymysql.connect(
            host=self.host or os.getenv("MYSQL_HOST", APP_DB_CONN["host"]),
            port=self.port or int(os.getenv("MYSQL_PORT", str(APP_DB_CONN["port"]))),
            user=self.user or os.getenv("MYSQL_USER", APP_DB_CONN["user"]),
            password=self.password if self.password is not None else os.getenv("MYSQL_PASSWORD", APP_DB_CONN["password"]),
            database=self.database or os.getenv("MYSQL_DATABASE", APP_DB_CONN["database"]),
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
        result = await asyncio.to_thread(self._query_sync, input.query)
        return StringToolOutput(result)
