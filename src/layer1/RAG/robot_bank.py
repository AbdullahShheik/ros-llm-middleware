"""
robot_bank.py
--------------
Builds the bank of currently active robot TYPES for retrieval -- the
fleet-and-skills analogue of few_shot_bank.py's example bank.

Three files compose into one bank entry per active robot type:

  robot_fleet.json     -- how many units of each robot TYPE are currently
                           deployed. Same type can be zero, one, or many
                           (e.g. two mobile robots) -- this is the only file
                           that changes as the fleet grows or shrinks.
  robots_registry.json -- one entry per unique robot TYPE (populated by
                           onboarding), listing which skill_ids that type
                           supports. Type-level, not instance-level.
  master_skills.json   -- the full skill catalog: every skill_id's
                           description/inputs/preconditions/effects. Used
                           only as a lookup to resolve a type's OWN skill_ids
                           into full definitions -- a bank entry never pulls
                           in skills outside what that specific robot type's
                           registry entry lists.

A robot type only becomes a bank entry if BOTH: the fleet currently has at
least one unit of it (count > 0) AND its registry entry's status is
"active" -- a type that's onboarded but not deployed, or deployed but
decommissioned, shouldn't offer its skills to Layer 1 right now.
"""

import json
import os

_LAYER1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLEET_FILE = os.path.join(_LAYER1_DIR, "robot_fleet.json")
REGISTRY_FILE = os.path.join(_LAYER1_DIR, "robots_registry.json")
MASTER_SKILLS_FILE = os.path.join(_LAYER1_DIR, "master_skills.json")


def _resolve_skills(skill_ids: list, master_by_id: dict, robot_type: str) -> list:
    """Resolves ONLY this robot type's own skill_ids against the master
    catalog into full skill dicts, in the shape build_skill_prompt_block and
    validate_plan already expect (name/description/inputs/preconditions/
    effects). master_by_id is a lookup, not a source of extra skills."""
    resolved, missing = [], []
    for sid in skill_ids:
        skill = master_by_id.get(sid)
        if skill is None:
            missing.append(sid)
            continue
        resolved.append({
            "name": skill["skill_id"],
            "description": skill["description"],
            "inputs": skill["inputs"],
            "preconditions": skill["preconditions"],
            "effects": skill["effects"],
        })
    if missing:
        # A registry entry referencing a skill_id no longer in the master
        # catalog is a data-integrity bug, not a runtime condition to work
        # around silently.
        raise ValueError(
            f"Robot type '{robot_type}' references skill_id(s) not found in "
            f"master_skills.json: {missing}. Registry and master catalog are out of sync."
        )
    return resolved


def _bank_text(robot_type: str, capabilities: list, skills: list,
                shared_skill_names: set) -> str:
    """Text used for cosine matching against instructions: capability
    keywords plus each DISTINGUISHING skill's own description, so wording
    that overlaps with what a skill actually DOES gets matched -- e.g.
    "bring the bottle" ~ "navigate to a named location using path planning".

    Skills in shared_skill_names (identical across every active robot type,
    e.g. move_to/stop/wait/rotate) are excluded from this text even though
    they still fully belong to the type's resolved skill set once selected.
    Their descriptions are verbatim-identical across types, so including
    them adds the same text to every document and only dilutes the signal
    that's supposed to tell types apart -- it never helps discriminate."""
    parts = [robot_type.replace("_", " ")]
    parts.extend(cap.replace("_", " ") for cap in capabilities)
    parts.extend(skill["description"] for skill in skills
                 if skill["name"] not in shared_skill_names)
    return ". ".join(parts)


def load_robot_bank(fleet_file: str = FLEET_FILE,
                     registry_file: str = REGISTRY_FILE,
                     master_skills_file: str = MASTER_SKILLS_FILE) -> list:
    """Returns a list of bank entries, one per active + deployed robot type:
        {"robot_type", "count", "capabilities", "skills", "text"}
    Each entry's "skills" is that type's own skill set only -- never the
    full master catalog.
    """
    with open(fleet_file, "r", encoding="utf-8") as f:
        fleet_counts = json.load(f)["fleet"]

    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)["robots"]

    with open(master_skills_file, "r", encoding="utf-8") as f:
        master_by_id = {s["skill_id"]: s for s in json.load(f)["skills"]}

    active_entries = []
    for entry in registry:
        robot_type = entry["robot_type"]
        count = fleet_counts.get(robot_type, 0)
        if count <= 0 or entry.get("status") != "active":
            continue
        skills = _resolve_skills(entry["skill_ids"], master_by_id, robot_type)
        active_entries.append((entry, count, skills))

    if not active_entries:
        raise ValueError(
            "No active robot types with fleet count > 0 -- check robot_fleet.json "
            "counts against robots_registry.json 'status' fields."
        )

    # Skills every active type has in common (e.g. move_to, stop, wait) carry
    # identical description text on every type -- useless, in fact harmful,
    # for telling types apart, so they're dropped from the retrieval text
    # (not from the resolved skill set) below.
    skill_name_sets = [{s["name"] for s in skills} for _, _, skills in active_entries]
    shared_skill_names = set.intersection(*skill_name_sets) if len(skill_name_sets) > 1 else set()

    bank = []
    for entry, count, skills in active_entries:
        robot_type = entry["robot_type"]
        capabilities = entry.get("capabilities_detected", [])
        bank.append({
            "robot_type": robot_type,
            "count": count,
            "capabilities": capabilities,
            "skills": skills,
            "text": _bank_text(robot_type, capabilities, skills, shared_skill_names),
        })
    return bank


def bank_summary(bank: list) -> str:
    """One-line breakdown of the active fleet, for start-up logging."""
    return ", ".join(
        f"{e['robot_type']} x{e['count']} ({len(e['skills'])} skills)" for e in bank
    )
