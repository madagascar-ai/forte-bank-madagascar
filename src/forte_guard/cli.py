from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .contracts import ZoneProfile
from .settings import Settings, load_env


def main():
    load_env()
    parser = argparse.ArgumentParser(prog="forte-guard")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the local API and interface")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    analyze = commands.add_parser("analyze", help="Run the same pipeline without a web server")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--zones", type=Path)
    analyze.add_argument("--replay", type=Path, help="Explicit synthetic/development object journal")
    analyze.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    settings.validate()
    if args.command == "serve":
        import os

        import uvicorn

        from .api import create_app

        uvicorn.run(
            create_app(settings),
            host=args.host or os.getenv("FORTE_HOST", "127.0.0.1"),
            port=args.port or int(os.getenv("FORTE_PORT", "8000")),
        )
    else:
        from .pipeline import run_pipeline

        if args.no_llm:
            settings.llm_enabled = False
        zones = ZoneProfile.model_validate_json(args.zones.read_text()) if args.zones else None
        job_id = uuid.uuid4().hex
        output = args.output or settings.data_dir / "jobs" / job_id
        result = run_pipeline(
            args.video,
            output,
            settings,
            job_id,
            lambda stage, value: print(f"{stage}: {value:.0%}", flush=True),
            zones=zones,
            replay=args.replay,
        )
        # Do not print descriptions, image data or keys into the controlling cloud chat.
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": result.status,
                    "candidates": len(result.violations),
                    "output": str(output),
                }
            )
        )


if __name__ == "__main__":
    main()
