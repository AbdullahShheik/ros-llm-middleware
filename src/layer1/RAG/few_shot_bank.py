"""
few_shot_bank.py
-----------------
Loads the few-shot example bank from few_shot_examples.json -- the single
source of truth for Layer 1's examples. Nothing in this repo should keep a
second, hand-maintained copy of the examples any more.

Each entry in the JSON has this shape:

    {
      "id": "plan_001",
      "instruction": "Pick up the red block and place it on the shelf.",
      "output": { "status": "ok", "subtasks": [...] }
    }
    # or, for a rejection example:
    { "id": "...", "instruction": "...",
      "output": { "status": "clarification_needed", "reason": "..." } }

On load, two retrieval-facing fields are DERIVED from the example content
rather than stored in the JSON, so they cannot drift out of sync with the
examples themselves:

  status         -- copied up from output["status"]
  skill_domains  -- computed from the skills actually used in the example's
                    subtasks, via SKILL_DOMAINS below

Both are consumed by the coverage guarantees in few_shot_retriever.py.
"""

import json
import os

EXAMPLES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "few_shot_examples.json",
)

# Skill -> domain. Skills that either arm or mobile robots can run (move_to,
# stop, wait, rotate...) and pure-perception skills are deliberately absent:
# they say nothing about which domain an example demonstrates, so they must
# not make an arm-only example look "combined".
SKILL_DOMAINS = {
    # arm / manipulator
    "pick": "arm",
    "place": "arm",
    "grip": "arm",
    "release": "arm",
    "inspect": "arm",
    "push": "arm",
    "home": "arm",
    # mobile base
    "navigate_to": "mobile",
    "move_base": "mobile",
    "follow_path": "mobile",
    "plan_path": "mobile",
    "avoid_obstacle": "mobile",
    "follow_person": "mobile",
    "dock": "mobile",
    "undock": "mobile",
    "localize": "mobile",
    "create_map": "mobile",
    "update_map": "mobile",
}

COMBINED_DOMAIN = frozenset({"arm", "mobile"})


def derive_skill_domains(output: dict) -> frozenset:
    """Domains demonstrated by an example, derived from its subtasks' skills.

    Rejection examples have no subtasks, so they get an empty set -- correct:
    they demonstrate a classification decision, not a domain.
    """
    domains = set()
    for subtask in output.get("subtasks") or []:
        for skill in subtask.get("required_skills") or []:
            domain = SKILL_DOMAINS.get(skill)
            if domain:
                domains.add(domain)
    return frozenset(domains)


def load_examples(filepath: str = EXAMPLES_FILE) -> list[dict]:
    """Load and enrich the few-shot bank. Returns a list of example dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for entry in raw["examples"]:
        output = entry["output"]
        examples.append({
            "id": entry["id"],
            "instruction": entry["instruction"],
            "output": output,
            "status": output.get("status", "ok"),
            "skill_domains": derive_skill_domains(output),
        })
    return examples


def bank_summary(examples: list[dict]) -> str:
    """One-line breakdown of the bank, for start-up logging."""
    by_status = {}
    for ex in examples:
        by_status[ex["status"]] = by_status.get(ex["status"], 0) + 1
    combined = sum(1 for ex in examples if ex["skill_domains"] == COMBINED_DOMAIN)
    status_str = ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
    return f"{len(examples)} examples ({status_str}; {combined} arm+mobile)"
