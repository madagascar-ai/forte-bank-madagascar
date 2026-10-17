from __future__ import annotations

import hmac
import json
import logging
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .contracts import ZoneProfile
from .pipeline import run_pipeline
from .readiness import inspect_readiness
from .settings import Settings
from .storage import JobStore, MaintenanceError

logger = logging.getLogger("forte_guard")
JOB_PATTERN = re.compile(r"^[0-9a-f]{32}$")
FRAME_PATTERN = re.compile(r"^f[0-9]{6}$")
ARTIFACTS = {
    "frames",
    "metadata",
    "zones",
    "object_journal",
    "action_journal",
    "candidates",
    "candidates.rules",
    "candidates.llm",
    "catalog_snapshot",
    "run_config",
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.validate()
    store = JobStore(settings.data_dir, recover=True)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forte-job")

    @asynccontextmanager
    async def lifespan(_app):
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Forte Guard",
        version="0.1.0",
        lifespan=lifespan,
        description="Local video journals and hybrid candidate detection; minimal verification.",
    )
    app.state.store = store
    app.state.settings = settings
    app.state.executor = executor

    if settings.allowed_clients:
        allowed = {ip_address(value) for value in ["127.0.0.1", "::1", *settings.allowed_clients]}

        @app.middleware("http")
        async def restrict_clients(request: Request, call_next):
            try:
                permitted = request.client is not None and ip_address(request.client.host) in allowed
            except ValueError:
                permitted = False
            if not permitted:
                return JSONResponse({"detail": "Доступ с этого устройства не разрешён"}, status_code=403)
            return await call_next(request)

    def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Неверный или отсутствующий X-API-Key")

    def get_job(job_id):
        if not JOB_PATTERN.fullmatch(job_id):
            raise HTTPException(404, "Задача не найдена")
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Задача не найдена")
        if job["status"] == "processing" and time.time() - job["created_at"] > settings.job_timeout_sec:
            store.update(
                job_id, "timeout", job["progress"], status="failed", error="Достигнут лимит времени обработки"
            )
            job = store.get(job_id)
        return job

    def worker(job_id):
        job = store.get(job_id)
        try:
            remaining = settings.job_timeout_sec - (time.time() - job["created_at"])
            if remaining <= 0:
                raise TimeoutError("Лимит времени истёк в очереди")
            zones = ZoneProfile.model_validate(job["zones"]) if job["zones"] else None

            def progress(stage, value):
                current = store.get(job_id)
                if current["status"] != "processing":
                    raise TimeoutError("Обработка задачи уже остановлена")
                # Only the worker publishes the terminal done state after saving results.
                if stage != "done":
                    store.update(job_id, stage, value)

            run_pipeline(
                store.folder(job_id) / "input.mov",
                store.folder(job_id),
                settings,
                job_id,
                progress,
                zones=zones,
                remaining_seconds=remaining,
            )
            if store.get(job_id)["status"] == "processing":
                store.update(job_id, "done", 1, status="done")
        except Exception as exc:
            # Avoid logging frames, prompts, descriptions or keys. Detailed stage artifacts remain local.
            error = (
                str(exc)
                if isinstance(exc, (FileNotFoundError, ValueError, TimeoutError))
                else (f"Ошибка обработки: {type(exc).__name__}")
            )
            logger.error("Job %s failed: %s", job_id, type(exc).__name__)
            store.update(job_id, "failed", 0, status="failed", error=error[:500])

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "detector_ready": settings.model_path.is_file(),
            "local_only": True,
            "verification_mode": "structural_only",
            "release": os.getenv("FORTE_RELEASE_SHA", "workspace"),
        }

    @app.post("/api/v1/analyze", dependencies=[Depends(require_key)], status_code=202)
    async def analyze(file: Annotated[UploadFile, File()], zones: Annotated[str | None, Form()] = None):
        if store.maintenance():
            raise HTTPException(
                503,
                "Сервис обновляется. Повторите загрузку через несколько секунд.",
                headers={"Retry-After": "10"},
            )
        filename = Path((file.filename or "video.mov").replace("\\", "/")).name
        if Path(filename).suffix.lower() not in {".mov", ".mp4"}:
            raise HTTPException(415, "Поддерживаются MOV и MP4")
        profile = None
        if zones:
            try:
                profile = ZoneProfile.model_validate_json(zones)
                if len(profile.zones) > 10:
                    raise ValueError("Too many zones")
                if len({zone.name for zone in profile.zones}) != len(profile.zones):
                    raise ValueError("Duplicate zones")
                profile.source = "manual"
            except (ValueError, ValidationError):
                raise HTTPException(422, "Неверный формат зон") from None
        job_id = uuid.uuid4().hex
        folder = store.folder(job_id)
        folder.mkdir(parents=True)
        size = 0
        try:
            with (folder / "input.mov").open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "Файл превышает допустимый размер")
                    output.write(chunk)
            if size == 0:
                raise HTTPException(422, "Загружен пустой файл")
            try:
                store.create(job_id, filename, profile.model_dump() if profile else None)
            except MaintenanceError:
                raise HTTPException(
                    503,
                    "Сервис обновляется. Повторите загрузку через несколько секунд.",
                    headers={"Retry-After": "10"},
                ) from None
            executor.submit(worker, job_id)
        except Exception:
            if store.get(job_id) is None:
                shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await file.close()
        return {"job_id": job_id, "status": "processing"}

    @app.get("/api/v1/readiness", dependencies=[Depends(require_key)])
    def readiness():
        report = inspect_readiness(settings)
        return JSONResponse(report, status_code=200 if report["status"] == "ready" else 503)

    @app.get("/api/v1/deployment", dependencies=[Depends(require_key)])
    def deployment():
        path = os.getenv("FORTE_DEPLOY_STATE")
        if not path:
            return {"managed": False, "release": os.getenv("FORTE_RELEASE_SHA", "workspace")}
        try:
            report = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            report = {}
        fields = (
            "state",
            "active_sha",
            "pending_sha",
            "previous_sha",
            "checked_at",
            "deployed_at",
            "ci_url",
            "error",
        )
        return {
            "managed": True,
            **{key: report.get(key) for key in fields},
            "maintenance": store.maintenance(),
        }

    @app.get("/api/v1/jobs", dependencies=[Depends(require_key)])
    def jobs():
        return {"jobs": store.list()}

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def status(job_id: str):
        job = get_job(job_id)
        if job["status"] == "done":
            return JSONResponse(json.loads((store.folder(job_id) / "result.json").read_text()))
        return {
            "job_id": job_id,
            "status": job["status"],
            "stage": job["stage"],
            "progress": job["progress"],
            "error": job["error"],
        }

    @app.get("/api/v1/jobs/{job_id}/artifacts/{artifact}", dependencies=[Depends(require_key)])
    def artifact(job_id: str, artifact: str):
        get_job(job_id)
        if artifact not in ARTIFACTS:
            raise HTTPException(404, "Артефакт не найден")
        path = store.folder(job_id) / f"{artifact}.json"
        if not path.is_file():
            raise HTTPException(404, "Этот этап пока не завершён")
        data = json.loads(path.read_text())
        if artifact == "frames":
            data = [{key: value for key, value in item.items() if key != "image_path"} for item in data]
        return JSONResponse(data)

    @app.get("/api/v1/jobs/{job_id}/frames/{frame_id}", dependencies=[Depends(require_key)])
    def frame(job_id: str, frame_id: str):
        get_job(job_id)
        if not FRAME_PATTERN.fullmatch(frame_id):
            raise HTTPException(404, "Кадр не найден")
        path = store.folder(job_id) / "frames" / f"{frame_id}.jpg"
        if not path.is_file():
            raise HTTPException(404, "Кадр не найден")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/v1/jobs/{job_id}/video", dependencies=[Depends(require_key)])
    def video(job_id: str):
        get_job(job_id)
        return FileResponse(store.folder(job_id) / "input.mov", media_type="video/quicktime")

    @app.get("/api/v1/config", dependencies=[Depends(require_key)])
    def config():
        return {
            "sample_fps": settings.sample_fps,
            "detector": settings.detector,
            "model": settings.model_path.name,
            "llm_model": settings.llm_model,
            "llm_enabled": settings.llm_enabled,
            "auto_zones": settings.auto_zones,
            "client_access_restricted": bool(settings.allowed_clients),
            "verification_mode": "structural_only",
            "max_upload_mb": settings.max_upload_mb,
            "job_timeout_sec": settings.job_timeout_sec,
            "zones": json.loads((settings.config_dir / "zones.json").read_text()),
            "rules": json.loads((settings.config_dir / "rules.json").read_text())["rules"],
        }

    web = Path(__file__).parent / "web"
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(
            web / "index.html",
            headers={
                "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; "
                "media-src 'self' blob:; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
                "Referrer-Policy": "no-referrer",
            },
        )

    return app
