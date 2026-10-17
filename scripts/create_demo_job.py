"""Register an explicitly synthetic pipeline run for inspecting the UI without bank footage."""

import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.make_demo import create_demo  # noqa: E402
from forte_guard.pipeline import run_pipeline  # noqa: E402
from forte_guard.settings import Settings, load_env  # noqa: E402
from forte_guard.storage import JobStore  # noqa: E402

load_env(ROOT / ".env")
settings = Settings()
settings.llm_enabled = False
settings.auto_zones = False
store = JobStore(settings.data_dir)
job_id = uuid.uuid4().hex
store.create(job_id, "Синтетический пример.mov")
source, replay, zones = create_demo(settings.data_dir / "demo_source")
shutil.copyfile(source, store.folder(job_id) / "input.mov")
try:
    run_pipeline(
        store.folder(job_id) / "input.mov", store.folder(job_id), settings, job_id, zones=zones, replay=replay
    )
    store.update(job_id, "done", 1, status="done")
except Exception:
    store.update(job_id, "failed", 0, status="failed", error="Synthetic demo failed")
    raise
print(f"Synthetic UI demo ready: {job_id}")
