"""
Onboard_robot.py
------------------
Full pipeline: URDF -> parsed summary -> LLM skill matching -> validation
-> written into robots_registry.json. Auto-goes-live (per your choice --
no manual approval gate), but the invalid-skill_id check always runs
regardless, since that's a correctness safety net, not an approval step.

Usage:
    python onboard_robot.py --urdf sample_mobile.urdf --robot-type mobile_robot_v2
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from urdf_parser import parse_urdf, summarize_for_llm
from llm_skill_matcher import match_skills_for_robot


def load_master_skills():
    base_dir = os.path.dirname(__file__)  # folder of this script
    path = os.path.join(base_dir, "..", "master_skills.json")
    path = os.path.abspath(path)

    with open(path) as f:
        return json.load(f)["skills"]


def _get_registry_path():
    return Path(__file__).resolve().parent.parent / "robots_registry.json"


def load_registry():
    path = _get_registry_path()
    return json.loads(path.read_text())


def save_registry(registry):
    path = _get_registry_path()
    path.write_text(json.dumps(registry, indent=2))


def onboard_robot_from_urdf(urdf_path, robot_type_name, master_skills_path="master_skills.json",
                             registry_path="robots_registry.json", call_llm_fn=None):
    """
    Runs the full pipeline and returns the new registry entry (also writes
    it to registry_path). Raises ValueError if robot_type_name already
    exists in the registry -- onboarding is for NEW robot types only;
    updating an existing one is a separate, deliberate action.
    """
    master_skills = load_master_skills()
    registry = load_registry()

    existing_types = {r["robot_type"] for r in registry["robots"]}
    if robot_type_name in existing_types:
        raise ValueError(
            f"Robot type '{robot_type_name}' already exists in the registry. "
            f"Onboarding is for new robot types only."
        )

    parsed = parse_urdf(urdf_path)
    urdf_summary = summarize_for_llm(parsed)

    match_result = match_skills_for_robot(urdf_summary, master_skills, call_llm_fn=call_llm_fn)

    if match_result["invalid_skill_ids"]:
        print(f"WARNING: LLM matcher returned {len(match_result['invalid_skill_ids'])} "
              f"skill_id(s) not found in master_skills.json -- dropped: "
              f"{match_result['invalid_skill_ids']}")

    new_entry = {
        "robot_type": robot_type_name,
        "urdf_source": urdf_path,
        "capabilities_detected": match_result["detected_capabilities"],
        "skill_ids": match_result["valid_skill_ids"],
        "date_onboarded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "onboarding_method": "urdf_llm_pipeline",
        "status": "active",
        "llm_reasoning": match_result["reasoning"],
        "llm_dropped_invalid_ids": match_result["invalid_skill_ids"],
    }

    registry["robots"].append(new_entry)
    save_registry(registry, registry_path)

    return new_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Path to the robot's URDF file")
    parser.add_argument("--robot-type", required=True, help="New robot type name to register")
    args = parser.parse_args()

    entry = onboard_robot_from_urdf(args.urdf, args.robot_type)
    print(json.dumps(entry, indent=2))
