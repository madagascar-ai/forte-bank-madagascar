"""Create local folders and a random API key, without overwriting user configuration."""

import secrets
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for name in ("data/jobs", "models"):
    (root / name).mkdir(parents=True, exist_ok=True)
env = root / ".env"
if not env.exists():
    template = (root / ".env.example").read_text()
    env.write_text(template.replace("change-this-local-key", secrets.token_urlsafe(32)))
    env.chmod(0o600)
print("Local setup ready. API key is stored in .env; it was not printed.")
