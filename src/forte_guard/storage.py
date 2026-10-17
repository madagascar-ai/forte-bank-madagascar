from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from pydantic import BaseModel


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


class JobStore:
    def __init__(self, data_dir: Path, recover: bool = False):
        self.root = data_dir.resolve()
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, filename TEXT NOT NULL, status TEXT NOT NULL,
                stage TEXT NOT NULL, progress REAL NOT NULL, created_at REAL NOT NULL,
                error TEXT, zones TEXT)""")
            if recover:
                conn.execute(
                    "UPDATE jobs SET status='failed', stage='interrupted', error=? WHERE status='processing'",
                    ("Сервис перезапущен во время обработки; загрузите видео повторно",),
                )

    def connect(self):
        return sqlite3.connect(self.db, timeout=15)

    def folder(self, job_id: str) -> Path:
        # IDs are also validated at API boundaries, but keep disk access independently safe.
        if not job_id or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Invalid job ID")
        return self.root / "jobs" / job_id

    def create(self, job_id: str, filename: str, zones: dict | None = None):
        self.folder(job_id).mkdir(parents=True, exist_ok=True)
        with self.lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?,?,'processing','queued',0,?,NULL,?)",
                (job_id, filename, time.time(), json.dumps(zones) if zones else None),
            )

    def update(self, job_id: str, stage: str, progress: float, status="processing", error=None):
        with self.lock, self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET stage=?,progress=?,status=?,error=? WHERE job_id=?",
                (stage, max(0, min(1, progress)), status, error, job_id),
            )

    def get(self, job_id: str):
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["zones"] = json.loads(item["zones"]) if item["zones"] else None
        return item

    def list(self):
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT job_id,filename,status,stage,progress,created_at,error FROM jobs ORDER BY created_at DESC LIMIT 100"
                )
            ]
