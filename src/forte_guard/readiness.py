"""Small local readiness probes; never load video data or return credentials."""

from __future__ import annotations

import importlib.util
import json
import tempfile

import httpx

from .contracts import ZoneProfile
from .reasoning.catalog import load_catalog
from .settings import Settings


def inspect_readiness(settings: Settings) -> dict:
    checks = []

    def check(name, ready, message):
        checks.append({"name": name, "ready": bool(ready), "message": message})

    try:
        settings.validate()
        load_catalog(settings.config_dir)
        zones = json.loads((settings.config_dir / "zones.json").read_text())
        ZoneProfile.model_validate({key: zones[key] for key in ("source", "zones")})
        detector = json.loads((settings.config_dir / "detector.json").read_text())
        check("configuration", True, "Конфигурация загружена")
    except (ValueError, OSError, KeyError):
        check("configuration", False, "Проверьте .env и config/")
        # Invalid locality settings must never cause an outbound probe.
        return {"status": "not_ready", "checks": checks}
    packages = ("torch", "ultralytics", "lap")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    check(
        "vision_dependencies",
        not missing,
        "Установлены" if not missing else "Установите зависимости .[vision]",
    )
    weights_ready = settings.model_path.is_file()
    check(
        "detector_weights", weights_ready, "Веса YOLO доступны" if weights_ready else "Подготовьте веса YOLO"
    )
    if settings.detector == "world":
        try:
            vocabulary = json.loads((settings.model_path.parent / "vocabulary.json").read_text())
            matched = vocabulary == detector["prompts"]
            check(
                "detector_vocabulary",
                matched,
                "Словарь совпадает" if matched else "Словарь не совпадает с config/",
            )
        except (OSError, ValueError):
            check("detector_vocabulary", False, "Выполните scripts/prepare_models.py")
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=settings.data_dir):
            pass
        check("storage", True, "Каталог задач доступен для записи")
    except OSError:
        check("storage", False, "Каталог задач недоступен для записи")
    if settings.llm_enabled:
        try:
            with httpx.Client(trust_env=False, timeout=3) as client:
                response = client.post(
                    settings.ollama_url.rstrip("/") + "/api/show", json={"model": settings.llm_model}
                )
                response.raise_for_status()
                capabilities = response.json().get("capabilities", [])
            check("local_llm", True, "Ollama и локальная модель доступны")
            if settings.auto_zones:
                vision = "vision" in capabilities
                check(
                    "vision_llm",
                    vision,
                    "Изображения поддерживаются" if vision else "Автозонам нужна vision-модель",
                )
        except (httpx.HTTPError, ValueError):
            check("local_llm", False, "Запустите Ollama и загрузите модель из FORTE_LLM_MODEL")
    else:
        check("local_llm", True, "Отключена настройкой FORTE_LLM_ENABLED")
    return {"status": "ready" if all(item["ready"] for item in checks) else "not_ready", "checks": checks}
