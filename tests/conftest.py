from pathlib import Path

import pytest

from examples.make_demo import create_demo
from forte_guard.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    return Settings(
        api_key="test-only-not-a-real-secret",
        data_dir=tmp_path / "data",
        config_dir=ROOT / "config",
        model_path=tmp_path / "absent.pt",
        llm_enabled=False,
        auto_zones=False,
        sample_fps=5,
    )


@pytest.fixture
def demo(tmp_path):
    return create_demo(tmp_path / "synthetic")
