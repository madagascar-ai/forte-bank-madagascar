"""Apply FORTE_* settings delivered by the deploy workflow on top of the server .env.

Overrides replace matching lines and new names are appended; everything else on the server (paths,
host-specific values) stays. Only names are printed, never values. The file is written atomically, 0600.
Usage: merge_env.py <server .env> <overrides file>
"""

import os
import re
import sys
import tempfile
from pathlib import Path

NAME = re.compile(r"^FORTE_[A-Z0-9_]+$")


def parse(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            values[name.strip()] = value
    return values


def merge(current: str, overrides: dict[str, str]) -> tuple[str, list[str]]:
    for name, value in overrides.items():
        if not NAME.fullmatch(name) or "\n" in value or "\r" in value:
            raise ValueError(f"refusing setting {name!r}: only single-line FORTE_* values are allowed")
    lines, applied = current.splitlines(), []
    for index, line in enumerate(lines):
        name = line.partition("=")[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and name in overrides:
            if line != f"{name}={overrides[name]}":
                applied.append(name)
            lines[index] = f"{name}={overrides[name]}"
    present = {line.partition("=")[0].strip() for line in lines if "=" in line and not line.lstrip().startswith("#")}
    added = [name for name in overrides if name not in present]
    if added:
        lines += ["", "# Managed in GitHub (Settings -> Secrets and variables -> Actions)"]
        lines += [f"{name}={overrides[name]}" for name in added]
    return "\n".join(lines) + "\n", applied + added


def main():
    target, source = Path(sys.argv[1]), Path(sys.argv[2])
    merged, changed = merge(target.read_text(), parse(source.read_text()))
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".env.")
    with os.fdopen(fd, "w") as handle:
        handle.write(merged)
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    print("settings changed: " + (", ".join(sorted(changed)) if changed else "none"))


if __name__ == "__main__":
    main()
