import json
from pathlib import Path

from ..contracts import ActionJournal, Rule


def load_catalog(config_dir: Path) -> tuple[list[Rule], dict]:
    rules = [
        Rule.model_validate(item) for item in json.loads((config_dir / "rules.json").read_text())["rules"]
    ]
    if len({rule.id for rule in rules}) != len(rules):
        raise ValueError("Rule IDs must be unique")
    scenarios = json.loads((config_dir / "scenarios.json").read_text())
    return rules, scenarios


def eligible(rule: Rule, journal: ActionJournal) -> bool:
    return rule.enabled and set(rule.required_capabilities).issubset(journal.capabilities)
