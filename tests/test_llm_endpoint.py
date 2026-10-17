import json

import httpx
import pytest

from forte_guard.readiness import inspect_readiness
from forte_guard.reasoning.llm import local_chat
from forte_guard.video import Deadline


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434", "http://ollama:11434", "http://192.168.1.20:11434", "http://100.118.149.106:11434"],
)
def test_team_owned_llm_hosts_are_allowed(settings, url):
    settings.ollama_url = url
    settings.validate()


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8:11434",
        "http://gpu.example.com:11434",
        "http://my-box.tailnet.ts.net:11434",
        "https://100.118.149.106:11434",
    ],
)
def test_public_addresses_and_hostnames_are_rejected(settings, url):
    settings.ollama_url = url
    with pytest.raises(ValueError, match="local"):
        settings.validate()


def _mock_client(monkeypatch, handler, target="forte_guard.reasoning.llm.httpx.Client"):
    factory = httpx.Client
    monkeypatch.setattr(target, lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs))


def test_chat_disables_thinking_and_strips_leaked_reasoning(monkeypatch, settings):
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        content = 'synthetic leaked reasoning\n</think>\n\n{"candidates": []}'
        return httpx.Response(200, json={"done_reason": "stop", "message": {"content": content}})

    _mock_client(monkeypatch, handler)
    settings.llm_model = "qwen3:30b"
    assert local_chat(settings, "synthetic", {}, Deadline(10)) == {"candidates": []}
    assert sent["think"] is False
    assert sent["model"] == "qwen3:30b"


def test_readiness_checks_vision_on_the_separate_zone_model(monkeypatch, settings):
    settings.llm_enabled = settings.auto_zones = True
    settings.llm_model, settings.zone_model = "qwen3:30b", "gemma4:26b"
    capabilities = {"qwen3:30b": ["completion", "thinking"], "gemma4:26b": ["completion", "vision"]}
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"capabilities": capabilities[json.loads(request.content)["model"]]}
        ),
        target="forte_guard.readiness.httpx.Client",
    )
    checks = {item["name"]: item["ready"] for item in inspect_readiness(settings)["checks"]}
    assert checks["local_llm"] and checks["vision_llm"]
