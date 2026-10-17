from __future__ import annotations

import os
from dataclasses import dataclass, field
from ipaddress import ip_address, ip_network
from pathlib import Path
from urllib.parse import urlparse


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


# Detectors whose classes are the text prompts of config/detector.json baked into the weights.
OPEN_VOCABULARY = {"yoloe", "world"}

# Tailscale/CGNAT range: Python does not mark it private, but it never routes to the internet.
_TEAM_NETWORK = ip_network("100.64.0.0/10")


def is_team_llm_host(hostname: str | None) -> bool:
    """Loopback, the Docker service, or a literal private/tailnet IP of a team-owned machine.

    Hostnames are rejected on purpose: DNS could silently point them at a cloud endpoint.
    """
    if hostname in {"localhost", "ollama"}:
        return True
    try:
        address = ip_address(hostname or "")
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address in _TEAM_NETWORK


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
        default_factory=lambda: Path(os.getenv("FORTE_MODEL_PATH", "models/detector-v2.pt"))
    )
    detector: str = field(default_factory=lambda: os.getenv("FORTE_DETECTOR", "yolo"))
    device: str = field(default_factory=lambda: os.getenv("FORTE_DEVICE", "auto"))
    sample_fps: float = field(default_factory=lambda: float(os.getenv("FORTE_SAMPLE_FPS", "5")))
    llm_enabled: bool = field(default_factory=lambda: _bool("FORTE_LLM_ENABLED", True))
    auto_zones: bool = field(default_factory=lambda: _bool("FORTE_AUTO_ZONES", True))
    # "track": absence only when a cashier track is seen away; "zone": also when the workstation stays empty.
    cashier_presence: str = field(default_factory=lambda: os.getenv("FORTE_CASHIER_PRESENCE", "zone"))
    ollama_url: str = field(default_factory=lambda: os.getenv("FORTE_OLLAMA_URL", "http://127.0.0.1:11434"))
    llm_model: str = field(default_factory=lambda: os.getenv("FORTE_LLM_MODEL", "gemma4:26b"))
    # The zone fallback sends a frame, so it needs a vision-capable model; text-only LLMs cannot do it.
    zone_model: str = field(
        default_factory=lambda: os.getenv("FORTE_ZONE_MODEL") or os.getenv("FORTE_LLM_MODEL", "gemma4:26b")
    )
    # Limits for keyless browser uploads (0 disables); API-key uploads are never limited.
    ui_max_queued: int = field(default_factory=lambda: int(os.getenv("FORTE_UI_MAX_QUEUED", "10")))
    ui_max_uploads_per_hour: int = field(
        default_factory=lambda: int(os.getenv("FORTE_UI_MAX_UPLOADS_PER_HOUR", "60"))
    )
    # Offline benchmark against team labels; off by default so the jury cannot start it.
    benchmark_enabled: bool = field(default_factory=lambda: _bool("FORTE_BENCHMARK_ENABLED", False))
    benchmark_labels: str = field(default_factory=lambda: os.getenv("FORTE_BENCHMARK_LABELS", ""))
    benchmark_videos: Path = field(
        default_factory=lambda: Path(os.getenv("FORTE_BENCHMARK_VIDEOS", "videos"))
    )
    benchmark_tolerance_sec: float = 3.0
    judge_model: str = field(
        default_factory=lambda: os.getenv("FORTE_JUDGE_MODEL") or os.getenv("FORTE_LLM_MODEL", "gemma4:26b")
    )
    job_timeout_sec: float = field(default_factory=lambda: float(os.getenv("FORTE_JOB_TIMEOUT_SEC", "285")))
    max_upload_mb: int = 256
    max_video_sec: float = 180
    detection_confidence: float = 0.20
    image_size: int = 640
    # api_key: the UI asks for the key; open (explicit opt-in): opening the page authorizes the
    # browser; trusted_clients: only FORTE_ALLOWED_CLIENTS get the browser session; guest: any visitor
    # may upload and sees only their own jobs, everything else still needs the key.
    web_access: str = field(default_factory=lambda: os.getenv("FORTE_WEB_ACCESS", "api_key"))
    inference_batch: int = 8
    allowed_clients: list[str] = field(
        default_factory=lambda: [
            value.strip() for value in os.getenv("FORTE_ALLOWED_CLIENTS", "").split(",") if value.strip()
        ]
    )

    def validate(self) -> None:
        if not self.api_key or self.api_key == "change-this-local-key":
            raise ValueError("Set FORTE_API_KEY in .env (see scripts/setup.py)")
        if not 0 < self.sample_fps <= 30:
            raise ValueError("FORTE_SAMPLE_FPS must be in (0, 30]")
        if self.detector not in {*OPEN_VOCABULARY, "yolo", "replay"}:
            raise ValueError("FORTE_DETECTOR must be world, yoloe, yolo, or replay")
        if self.cashier_presence not in {"track", "zone"}:
            raise ValueError("FORTE_CASHIER_PRESENCE must be track or zone")
        url = urlparse(self.ollama_url)
        if url.scheme != "http" or not is_team_llm_host(url.hostname):
            raise ValueError(
                "Ollama must be local: loopback, Docker service 'ollama', or a private/tailnet IP"
            )
        if self.job_timeout_sec <= 0:
            raise ValueError("Job timeout must be positive")
        # Exact addresses only: a typo must never expand access to an entire subnet.
        self.allowed_clients = [str(ip_address(value)) for value in self.allowed_clients]
        if self.web_access not in {"api_key", "open", "trusted_clients", "guest"}:
            raise ValueError("FORTE_WEB_ACCESS must be api_key, open, trusted_clients or guest")
        if self.web_access == "trusted_clients" and not self.allowed_clients:
            raise ValueError("Trusted web access requires an explicit FORTE_ALLOWED_CLIENTS list")
