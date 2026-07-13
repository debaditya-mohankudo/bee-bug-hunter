"""Minimal demo app for crew_bug_hunter to investigate.

Serves crew_bug_hunter/flows/example_login.yaml exactly:
  GET  /login              -> #email, #password, #login-submit
  POST /api/auth/login     -> seeded bug: queries a column that doesn't exist,
                               so valid creds 500 with a full traceback in the logs
  GET  /dashboard           -> #dashboard, fires the seeded slow query on load
  GET  /api/orders/<user_id> -> seeded perf issue: a self cross-join that's slow
                                 and explicitly logs its duration in ms
"""
import logging
import os
import sys
import time

import pymysql
from flask import Flask, jsonify, request

app = Flask(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logger = app.logger


def db():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "db"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
        database=os.getenv("MYSQL_DATABASE", "demo"),
        cursorclass=pymysql.cursors.DictCursor,
    )


@app.get("/health")
def health():
    return "ok"


@app.get("/login")
def login_page():
    return """
    <html><body>
      <input id="email" />
      <input id="password" type="password" />
      <button id="login-submit" onclick="submit()">Log in</button>
      <div id="dashboard" style="display:none"></div>
      <script>
        function submit() {
          fetch('/api/auth/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              email: document.getElementById('email').value,
              password: document.getElementById('password').value,
            }),
          }).then(r => r.json()).then(data => {
            if (data.user_id) {
              document.getElementById('dashboard').style.display = 'block';
              fetch('/api/orders/' + data.user_id);
            }
          });
        }
      </script>
    </body></html>
    """


@app.post("/api/auth/login")
def api_login():
    body = request.get_json(force=True)
    email = body.get("email")
    password = body.get("password")
    conn = db()
    try:
        with conn.cursor() as cur:
            # BUG: this table's column is `password`, not `passwd` -- every login
            # fails with a MySQL error, not a credentials check failure.
            cur.execute(
                "SELECT id FROM users WHERE email=%s AND passwd=%s", (email, password)
            )
            row = cur.fetchone()
    except Exception:
        logger.exception("login query failed")
        return jsonify({"error": "internal error"}), 500
    finally:
        conn.close()

    if not row:
        return jsonify({"error": "invalid credentials"}), 401
    return jsonify({"user_id": row["id"]})


@app.get("/api/orders/<int:user_id>")
def api_orders(user_id):
    conn = db()
    started = time.monotonic()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, item, amount FROM orders WHERE user_id = %s LIMIT 3000",
                (user_id,),
            )
            order_ids = [r["id"] for r in cur.fetchall()]

            # BUG: N+1 -- one round trip per order to look up its item detail,
            # instead of a single batched query with an IN (...) clause.
            rows = []
            for order_id in order_ids:
                cur.execute(
                    "SELECT id, item, amount FROM orders WHERE id = %s", (order_id,)
                )
                rows.append(cur.fetchone())
    finally:
        conn.close()
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    logger.info("orders query for user_id=%s took %sms", user_id, elapsed_ms)
    return jsonify({"rows": rows, "elapsed_ms": elapsed_ms})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
