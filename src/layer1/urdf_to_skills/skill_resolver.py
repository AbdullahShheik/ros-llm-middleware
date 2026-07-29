"""
skill_resolver.py
--------------------
bridges the registry (which stores skill_ids only) into full skill objects
-- this is what feeds directly into your existing skill_filter.py, which
expects a list of full skill dicts (name/description/inputs/preconditions/
effects), same shape your original robot_skills.json already had.

get_skills_for_robot_type() replaces "load robot_skills.json directly" in
your pipeline with "resolve this robot type's skill_ids against the master
catalog" -- the two-tier filtering logic in skill_filter.py doesn't need
to change at all, it just receives its skill list from here now.
"""

import json


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_skills_for_robot_type(robot_type, master_skills_path="master_skills.json",
                               registry_path="robots_registry.json"):
    """
    Returns a list of full skill dicts (compatible with skill_filter.py's
    expected shape -- note the key is "name" there, so this also renames
    "skill_id" -> "name" to match).
    """
    master_skills = load_json(master_skills_path)["skills"]
    registry = load_json(registry_path)

    robot_entry = next((r for r in registry["robots"] if r["robot_type"] == robot_type), None)
    if robot_entry is None:
        raise ValueError(f"Robot type '{robot_type}' not found in registry.")

    master_by_id = {s["skill_id"]: s for s in master_skills}

    resolved = []
    missing = []
    for sid in robot_entry["skill_ids"]:
        if sid in master_by_id:
            s = master_by_id[sid]
            resolved.append({
                "name": s["skill_id"],
                "description": s["description"],
                "inputs": s["inputs"],
                "preconditions": s["preconditions"],
                "effects": s["effects"],
            })
        else:
            missing.append(sid)

    if missing:
        # A registry entry referencing a skill_id no longer in the master
        # catalog is a real data-integrity problem (e.g. someone renamed or
        # deleted a master skill without updating referencing robots) --
        # surface it loudly rather than silently dropping the robot's
        # capability.
        raise ValueError(
            f"Robot type '{robot_type}' references skill_id(s) not found in "
            f"master_skills.json: {missing}. Registry and master catalog are out of sync."
        )

    return resolved
