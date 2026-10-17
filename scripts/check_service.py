"""Check the actual HTTP upload -> pipeline -> artifacts, without printing video-derived content."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forte_guard.settings import Settings, load_env  # noqa: E402


def main():
    load_env(ROOT / ".env")
    settings = Settings()
    settings.validate()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=f"http://127.0.0.1:{os.getenv('FORTE_PORT', '8000')}")
    parser.add_argument("--video", type=Path, help="Optional full HTTP smoke test with a local video")
    args = parser.parse_args()
    # An explicit URL can be a team server; the key is sent only to that requested server.
    with httpx.Client(
        base_url=args.url, headers={"X-API-Key": settings.api_key}, trust_env=False, timeout=30
    ) as client:
        response = client.get("/api/v1/readiness")
        report = response.json()
        for check in report.get("checks", []):
            print(f"{'OK' if check['ready'] else 'FAIL'} {check['name']}: {check['message']}", flush=True)
        response.raise_for_status()
        if not args.video:
            return
        started = time.monotonic()
        with args.video.open("rb") as source:
            upload = client.post("/api/v1/analyze", files={"file": (args.video.name, source)})
        upload.raise_for_status()
        job_id = upload.json()["job_id"]
        previous = None
        while time.monotonic() - started < settings.job_timeout_sec + 15:
            result = client.get(f"/api/v1/jobs/{job_id}")
            result.raise_for_status()
            value = result.json()
            stage = value.get("stage", value["status"])
            if stage != previous:
                print(f"Stage: {stage}", flush=True)
                previous = stage
            if value["status"] == "failed":
                raise RuntimeError("HTTP pipeline failed; inspect the job locally")
            if value["status"] == "done":
                break
            time.sleep(1)
        else:
            raise TimeoutError("HTTP smoke test exceeded its deadline")
        artifacts = [
            "frames",
            "object_journal",
            "action_journal",
            "candidates.rules",
            "candidates.llm",
            "candidates",
        ]
        for name in artifacts:
            artifact = client.get(f"/api/v1/jobs/{job_id}/artifacts/{name}")
            artifact.raise_for_status()
            if name == "frames":
                frame_id = artifact.json()[0]["frame_id"]
                image = client.get(f"/api/v1/jobs/{job_id}/frames/{frame_id}")
                image.raise_for_status()
                if not image.content.startswith(b"\xff\xd8"):
                    raise RuntimeError("Frame endpoint did not return a JPEG")
        video = client.get(f"/api/v1/jobs/{job_id}/video", headers={"Range": "bytes=0-15"})
        if video.status_code != 206:
            raise RuntimeError("Video endpoint does not support playback range requests")
        if client.get("/api/v1/jobs", headers={"X-API-Key": "invalid"}).status_code != 401:
            raise RuntimeError("Authentication check failed")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "job_id": job_id,
                    "elapsed_sec": round(time.monotonic() - started, 2),
                    "stats": value["stats"],
                    "artifacts_checked": len(artifacts),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
