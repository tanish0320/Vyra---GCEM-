"""Thread-safe ephemeral and persisted sidecar job store for VYRA React UI integration.

Stores asynchronous screening jobs without modifying the core Scan database schema.
Preserves idempotency keys, stages, detection proposals, and user confirmed ROIs.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import UPLOADS_DIR

logger = logging.getLogger("vyra.job_store")


class JobStore:
    """Thread-safe sidecar store for screening jobs."""

    def __init__(self, persistence_file: Path | None = None):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._idempotency_keys: dict[str, str] = {}  # key -> job_id
        self._persistence_file = persistence_file or (UPLOADS_DIR / "jobs_store.json")
        self._load()

    def _load(self) -> None:
        """Load persisted jobs from JSON file on disk."""
        if not self._persistence_file.exists():
            return
        try:
            with open(self._persistence_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._jobs = data.get("jobs", {})
                self._idempotency_keys = data.get("idempotency_keys", {})
        except Exception as exc:
            logger.warning("Could not read jobs store from %s: %s", self._persistence_file, exc)

    def _persist(self) -> None:
        """Write current jobs state atomically to disk."""
        try:
            self._persistence_file.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._persistence_file.with_suffix(".tmp")
            payload = {
                "jobs": self._jobs,
                "idempotency_keys": self._idempotency_keys,
            }
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            temp_path.replace(self._persistence_file)
        except Exception as exc:
            logger.warning("Could not persist jobs store: %s", exc)

    def create_job(self, job: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        """Save a new job and record its idempotency key if provided."""
        job_id = str(job["id"])
        with self._lock:
            self._jobs[job_id] = job
            if idempotency_key:
                self._idempotency_keys[str(idempotency_key).strip()] = job_id
            self._persist()
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve a job by its unique identifier."""
        with self._lock:
            job = self._jobs.get(str(job_id))
            return dict(job) if job else None

    def get_job_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        """Retrieve an existing job by client Idempotency-Key header."""
        with self._lock:
            clean_key = str(idempotency_key).strip()
            job_id = self._idempotency_keys.get(clean_key)
            if not job_id:
                return None
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update fields on an existing job and refresh updated_at timestamp."""
        with self._lock:
            target_id = str(job_id)
            if target_id not in self._jobs:
                return None
            job = self._jobs[target_id]
            job.update(updates)
            job["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            self._persist()
            return dict(job)

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        """Cancel a pending/in-progress job without persisting screening results."""
        with self._lock:
            target_id = str(job_id)
            if target_id not in self._jobs:
                return None
            job = self._jobs[target_id]
            job["status"] = "cancelled"
            job["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            self._persist()
            return dict(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return all jobs sorted by creation timestamp descending."""
        with self._lock:
            return [dict(j) for j in sorted(self._jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True)]


# Global default instance
job_store = JobStore()
