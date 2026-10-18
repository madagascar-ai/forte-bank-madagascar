"""Aggregate-only diagnostics for a benchmark run: where violations are found or lost.

Prints counts per outcome group (TP/FN/FP/TN) and never prints descriptions, label comments,
judge text, file names or frames, so the output is safe to share with cloud agents (AGENTS.md 2.2).

    python scripts/benchmark_diagnostics.py --data ~/forte-bench-data [--run RUN_ID]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

GROUPS = ("TP", "FN", "FP", "TN", "failed")
RULES = json.loads((Path(__file__).resolve().parents[1] / "config/rules.json").read_text())["rules"]
BRIDGE_SEC = 1.0


def spans(states: list[dict], name: str, value, bridge: float) -> list[tuple[float, float]]:
    """Intervals where the state holds with clear visibility; gaps up to `bridge` seconds are joined."""
    raw = sorted(
        (s["start_sec"], s["end_sec"])
        for s in states
        if s["name"] == name and s.get("value") == value and s.get("visibility", "clear") == "clear"
    )
    joined: list[list[float]] = []
    for start, end in raw:
        if joined and start - joined[-1][1] <= bridge:
            joined[-1][1] = max(joined[-1][1], end)
        else:
            joined.append([start, end])
    return [(a, b) for a, b in joined]


def longest_overlap(states: list[dict], required: dict, bridge: float) -> float:
    current = None
    for name, value in required.items():
        parts = spans(states, name, value, bridge)
        if current is None:
            current = parts
            continue
        current = [(max(a, c), min(b, d)) for a, b in current for c, d in parts if max(a, c) < min(b, d)]
    return max((b - a for a, b in current or []), default=0.0)


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def latest_run(root: Path) -> Path:
    runs = [p for p in (root / "benchmarks").iterdir() if (p / "run.json").is_file()]
    return max(runs, key=lambda p: (load(p / "run.json") or {}).get("started_at") or 0)


def video_stats(folder: Path) -> dict:
    journal = load(folder / "action_journal.json") or {}
    objects = load(folder / "object_journal.json") or {}
    final = load(folder / "candidates.json") or []
    rules = load(folder / "candidates.rules.json") or []
    llm = load(folder / "candidates.llm.json") or []
    zones = load(folder / "zones.json") or {}
    result = load(folder / "result.json") or {}
    true_states = {
        state["name"] for state in journal.get("states", []) if state.get("value") not in (False, None, "closed")
    }
    roles = Counter(role for role in (journal.get("roles") or {}).values() if isinstance(role, str))
    near = {}
    for rule in RULES:
        engine = rule.get("engine", {})
        if engine.get("type") == "overlap":
            states = journal.get("states", [])
            near[rule["id"]] = (
                longest_overlap(states, engine["states"], 0.0),
                longest_overlap(states, engine["states"], BRIDGE_SEC),
                engine.get("min_duration_sec", 0.0),
            )
    return {
        "near": near,
        "labels": {d["label"] for frame in objects.get("frames", []) for d in frame.get("detections", [])},
        "actions": {action["type"] for action in journal.get("actions", [])},
        "states": true_states,
        "roles": set(roles),
        "rule_ids": {c["rule_id"] for c in rules},
        "llm_ids": {c["rule_id"] for c in llm},
        "accepted": {c["rule_id"] for c in final if c.get("status") == "accepted"},
        "rejected": Counter(
            f"{c['rule_id']}:{c.get('verification')}" for c in final if c.get("status") == "rejected"
        ),
        "zones": {zone["name"] for zone in zones.get("zones", [])},
        "zone_source": zones.get("source"),
        "llm_warning": any("LLM" in w for w in result.get("warnings", [])),
        "has_rule": bool(rules),
        "has_llm": bool(llm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="FORTE_DATA_DIR of the service")
    parser.add_argument("--run", help="run_id; latest by default")
    args = parser.parse_args()
    folder = args.data.expanduser() / "benchmarks" / args.run if args.run else latest_run(args.data.expanduser())
    run = load(folder / "run.json")
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in run["items"]:
        outcome = (item.get("score") or {}).get("outcome")
        if outcome in GROUPS:
            groups[outcome].append(video_stats(folder / "videos" / item["video"]))

    print(f"run {run['run_id']} release {str(run['config'].get('release'))[:7]} status {run['status']}")
    print("videos per group:", {g: len(groups[g]) for g in GROUPS})

    def table(title: str, key: str):
        print(f"\n== {title}: number of videos per group where it appears")
        names = sorted({name for g in GROUPS for v in groups[g] for name in v[key]})
        for name in names:
            row = {g: sum(1 for v in groups[g] if name in v[key]) for g in GROUPS if groups[g]}
            print(f"  {name:48s} " + "  ".join(f"{g}={n}" for g, n in row.items()))

    print("\n== pipeline funnel (videos per group)")
    for label, test in [
        ("any rule-engine candidate", lambda v: v["has_rule"]),
        ("any LLM candidate", lambda v: v["has_llm"]),
        ("any accepted candidate", lambda v: bool(v["accepted"])),
        ("candidates but all rejected", lambda v: (v["has_rule"] or v["has_llm"]) and not v["accepted"]),
        ("LLM warning (branch failed)", lambda v: v["llm_warning"]),
        ("cashier role found", lambda v: "cashier" in v["roles"]),
        ("client role found", lambda v: "client" in v["roles"]),
    ]:
        print(f"  {label:48s} " + "  ".join(f"{g}={sum(1 for v in groups[g] if test(v))}" for g in GROUPS if groups[g]))
    print("  zone sources:", {g: dict(Counter(v["zone_source"] for v in groups[g])) for g in GROUPS if groups[g]})
    table("detector labels", "labels")
    table("zones", "zones")
    table("journal actions", "actions")
    table("journal states (true / not closed)", "states")
    table("rule-engine candidates by rule", "rule_ids")
    table("LLM candidates by rule", "llm_ids")
    table("accepted by rule", "accepted")
    print(f"\n== state rules: videos reaching min duration (as is / joining gaps <= {BRIDGE_SEC}s / half of min)")
    for rule in [r for r in RULES if r.get("engine", {}).get("type") == "overlap"]:
        cells = []
        for g in GROUPS:
            if groups[g]:
                values = [v["near"][rule["id"]] for v in groups[g]]
                strict = sum(1 for a, _, m in values if a >= m)
                bridged = sum(1 for _, b, m in values if b >= m)
                half = sum(1 for _, b, m in values if b >= m / 2)
                cells.append(f"{g}={strict}/{bridged}/{half}")
        print(f"  {rule['id']:36s} " + "  ".join(cells))
    print("\n== rejected candidates by rule:reason (count of candidates per group)")
    reasons = sorted({r for g in GROUPS for v in groups[g] for r in v["rejected"]})
    for reason in reasons:
        print(f"  {reason:72s} " + "  ".join(f"{g}={sum(v['rejected'][reason] for v in groups[g])}" for g in GROUPS if groups[g]))


if __name__ == "__main__":
    main()
