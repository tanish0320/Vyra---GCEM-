"""Idempotent SQL tweaks after deploy so existing databases gain new columns safely.

`db.create_all()` does not add columns to tables that already exist. These patches
keep patient rows compatible across upgrades without a separate Alembic setup.
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from config import DATABASE_BACKEND


_SCAN_COLUMN_PATCHES: tuple[tuple[str, str, str], ...] = (
    ("deleted_at", "DATETIME", "TIMESTAMP WITH TIME ZONE"),
    ("is_mock", "BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("model_version", "VARCHAR(120)", "VARCHAR(120)"),
    ("calibration_status", "VARCHAR(40)", "VARCHAR(40)"),
    ("hb_estimate_g_dl", "FLOAT", "DOUBLE PRECISION"),
    ("regression_status", "VARCHAR(40)", "VARCHAR(40)"),
    ("anatomical_modality", "VARCHAR(40)", "VARCHAR(40)"),
    ("triage_level", "VARCHAR(32)", "VARCHAR(32)"),
    ("ita_angle", "FLOAT", "DOUBLE PRECISION"),
    ("ita_category", "VARCHAR(40)", "VARCHAR(40)"),
    ("glare_fraction", "FLOAT", "DOUBLE PRECISION"),
    ("normalized_image_path", "VARCHAR(500)", "VARCHAR(500)"),
    ("quality_warnings_json", "TEXT", "TEXT"),
)


def apply_schema_patches(db) -> None:
    """Apply additive schema changes required by the current application code."""

    try:
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        if "scans" not in tables:
            return
        column_names = {column["name"] for column in inspector.get_columns("scans")}
        added_columns: list[str] = []

        for col_name, sqlite_type, pg_type in _SCAN_COLUMN_PATCHES:
            if col_name in column_names:
                continue
            if DATABASE_BACKEND == "sqlite":
                db.session.execute(text(f"ALTER TABLE scans ADD COLUMN {col_name} {sqlite_type}"))
            else:
                db.session.execute(
                    text(f"ALTER TABLE scans ADD COLUMN IF NOT EXISTS {col_name} {pg_type}")
                )
            added_columns.append(col_name)

        if added_columns:
            db.session.commit()
            print(
                f"[schema] Added scans columns: {', '.join(added_columns)}",
                flush=True,
            )
    except Exception as exc:
        db.session.rollback()
        print(f"[schema] Patch skipped or failed (non-fatal): {exc}", flush=True)
