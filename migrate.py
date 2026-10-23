"""Database migration script for AnemiaVision AI.

Run this once before starting the Flask app (or include it in the Docker
entrypoint) to ensure all tables exist in the configured database.

On Render the DATABASE_URL env-var points to a persistent Postgres instance,
so patient history survives every redeploy and container restart.
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# Quick check: verify we can reach the database before Flask starts up.
# On Render the Postgres DB may take a few seconds to accept connections
# after a cold container start.
# ---------------------------------------------------------------------------


def wait_for_db(max_retries: int = 10, delay: float = 2.0) -> None:
    """Poll the database until it accepts connections or the retry limit is hit."""

    from config import DATABASE_URI, DATABASE_BACKEND

    if DATABASE_BACKEND == "sqlite":
        print("[migrate] Using SQLite — no wait required.", flush=True)
        return

    # For Postgres we attempt a raw psycopg connection before handing off to
    # SQLAlchemy so we get a clean error rather than a cryptic ORM traceback.
    try:
        import psycopg  # psycopg v3
    except ImportError:
        try:
            import psycopg2 as psycopg  # type: ignore[no-redef]
        except ImportError:
            print("[migrate] psycopg not available — skipping DB wait.", flush=True)
            return

    # Strip the SQLAlchemy driver prefix so psycopg can parse the DSN.
    dsn = DATABASE_URI
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if dsn.startswith(prefix):
            dsn = "postgresql://" + dsn[len(prefix):]
            break

    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg.connect(dsn, connect_timeout=5)
            conn.close()
            print(f"[migrate] Database is ready (attempt {attempt}).", flush=True)
            return
        except Exception as exc:
            print(
                f"[migrate] Waiting for database (attempt {attempt}/{max_retries}): {exc}",
                flush=True,
            )
            time.sleep(delay)

    print("[migrate] WARNING: Database did not become ready in time. Proceeding anyway.", flush=True)


def run_migrations() -> None:
    """Create all tables using the Flask-SQLAlchemy metadata."""

    from app import create_app, db
    from schema_patches import apply_schema_patches

    app = create_app()
    with app.app_context():
        print("[migrate] Running db.create_all() …", flush=True)
        db.create_all()
        apply_schema_patches(db)
        print("[migrate] Tables created / verified successfully.", flush=True)

        # Report which backend is active so we can confirm Postgres is used.
        from config import DATABASE_BACKEND, DATABASE_URI
        masked = DATABASE_URI[:40] + "..." if len(DATABASE_URI) > 40 else DATABASE_URI
        print(f"[migrate] Active DB backend : {DATABASE_BACKEND}", flush=True)
        print(f"[migrate] Active DB URI     : {masked}", flush=True)

        if DATABASE_BACKEND == "sqlite":
            print(
                "[migrate] WARNING: Using SQLite. Patient history can be LOST on redeploys "
                "unless the DB file is on a persistent disk. Set DATABASE_URL to Postgres for "
                "durable storage (required for long-term retention on Vercel serverless).",
                flush=True,
            )


if __name__ == "__main__":
    wait_for_db()
    run_migrations()
    print("[migrate] Done.", flush=True)
