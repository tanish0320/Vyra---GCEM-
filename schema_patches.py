"""Idempotent SQL tweaks after deploy so existing databases gain new columns safely.

`db.create_all()` does not add columns to tables that already exist. These patches
keep patient rows compatible across upgrades without a separate Alembic setup.
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from config import DATABASE_BACKEND


def apply_schema_patches(db) -> None:
    """Apply additive schema changes required by the current application code."""

    try:
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        if "scans" not in tables:
            return
        column_names = {column["name"] for column in inspector.get_columns("scans")}
        if "deleted_at" in column_names:
            return

        if DATABASE_BACKEND == "sqlite":
            db.session.execute(text("ALTER TABLE scans ADD COLUMN deleted_at DATETIME"))
        else:
            db.session.execute(
                text("ALTER TABLE scans ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE")
            )
        db.session.commit()
        print("[schema] Added scans.deleted_at for soft-delete retention.", flush=True)
    except Exception as exc:
        db.session.rollback()
        print(f"[schema] Patch skipped or failed (non-fatal): {exc}", flush=True)
