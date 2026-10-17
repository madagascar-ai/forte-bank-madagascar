from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


def load_env(path: Path = Path(".env")) -> None:
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key, value.strip().strip("\"'"))


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("FORTE_API_KEY", ""))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("FORTE_DATA_DIR", "data")))
    config_dir: Path = field(default_factory=lambda: Path(os.getenv("FORTE_CONFIG_DIR", "config")))
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("FORTE_MODEL_PATH", "models/cashier-world.pt"))
    )
    detector: str = field(default_factory=lambda: os.getenv("FORTE_DETECTOR", "world"))
    device: str = field(default_factory=lambda: os.getenv("FORTE_DEVICE", "auto"))
    sample_fps: float = field(default_factory=lambda: float(os.getenv("FORTE_SAMPLE_FPS", "5")))
    llm_enabled: bool = field(default_factory=lambda: _bool("FORTE_LLM_ENABLED", True))
    auto_zones: bool = field(default_factory=lambda: _bool("FORTE_AUTO_ZONES", True))
    ollama_url: str = field(default_factory=lambda: os.getenv("FORTE_OLLAMA_URL", "http://127.0.0.1:11434"))
    llm_model: str = field(default_factory=lambda: os.getenv("FORTE_LLM_MODEL", "gemma3:4b"))
    job_timeout_sec: float = field(default_factory=lambda: float(os.getenv("FORTE_JOB_TIMEOUT_SEC", "285")))
    max_upload_mb: int = 256
    max_video_sec: float = 180
    detection_confidence: float = 0.20
    image_size: int = 640
    inference_batch: int = 8

    def validate(self) -> None:
        if not self.api_key or self.api_key == "change-this-local-key":
            raise ValueError("Set FORTE_API_KEY in .env (see scripts/setup.py)")
        if not 0 < self.sample_fps <= 30:
            raise ValueError("FORTE_SAMPLE_FPS must be in (0, 30]")
        if self.detector not in {"world", "yolo", "replay"}:
            raise ValueError("FORTE_DETECTOR must be world, yolo, or replay")
        url = urlparse(self.ollama_url)
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1", "ollama"}:
            raise ValueError("Ollama must be local: localhost, loopback, or Docker service 'ollama'")
        if self.job_timeout_sec <= 0:
            raise ValueError("Job timeout must be positive")
