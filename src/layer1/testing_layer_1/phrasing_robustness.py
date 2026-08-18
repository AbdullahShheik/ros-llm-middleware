#!/usr/bin/env python3
"""
phrasing_robustness.py
-----------------------
Tests whether Layer 1's decomposition is sensitive to how an instruction is
WORDED, as distinct from the LLM's own inherent run-to-run stochasticity
(temperature-driven landmark/direction choices when multiple options are
equally valid for the same instruction).

Design (confirmed before writing this):
  - 2 canonical instructions, each with a REFERENCE plan pulled from
    src/layer1/eval_runs.jsonl (plans[0].subtasks -- the first LLM
    decomposition of that instruction, before any replan).
  - ONE frozen environment snapshot, built once from panda_world.sdf's own
    declared spawn poses (NOT a live Gazebo/perception read -- this is what
    makes it reproducible offline and identical across all 16 calls). Reused
    unchanged for every call so environment drift can't confound the
    phrasing comparison.
  - Per canonical instruction: 3 baseline-repeat calls (the EXACT canonical
    text, unparaphrased) + 5 paraphrase calls, all compared against the same
    reference plan under the same frozen environment. The baseline-repeat
    arm measures the instruction's own inherent mismatch rate; only a
    paraphrase mismatch rate meaningfully above that baseline counts as
    evidence of genuine phrasing sensitivity, not just LLM noise.
  - decompose_instruction() (from layer1_pipeline.py) is called directly,
    not run_dart_style_evaluation.py's hand-rolled (and currently broken --
    see that file) equivalent.

Structural comparison (the "signature" a plan is reduced to before
comparing -- ignores task-id labels/ordering by construction):
  1. Topologically order the plan's subtasks (ties broken by ascending
     task id).
  2. Keep only pick/place subtasks that carry an object_name. navigate_to
     and detach never carry object identity in their args, so they're
     excluded; attach is ALSO excluded -- whether an object needs
     mobile-robot transport is a fact about the frozen environment (fixed
     identically across every call here), not about how the instruction was
     worded, so it contributes zero phrasing-sensitivity signal and would
     only produce a stale-reference artifact (see: Reference 2's own
     reachable_by_arm tags reflect leftover state from an earlier,
     unrelated test run, not a fresh-spawn environment).
  3. Normalize object names (red_block -> red_cube, etc.) via
     OBJECT_NAME_ALIASES, imported from placement_eval.py -- not
     re-typed here.
  4. Group by normalized object name, preserving topological order, into
     {skill_sequence, robots, place_args}.

Comparison (reference vs. candidate):
  - Object-name sets must match exactly (else: immediate mismatch).
  - For each shared object: skill_sequence list-equal, robots list-equal,
    every place_args key exactly equal (landmark/direction/relative_to are
    NOT fuzzy-matched -- a different anchor or axis is a genuine structural
    difference, not noise).
  - Every mismatch is collected as {object, field, reference_value,
    candidate_value}, not just a bare boolean.

A paraphrase/baseline call whose decompose_instruction() raises (clarification_
needed, infeasible, or retries exhausted) is recorded as a non-match with that
error as the reason -- no signature is computed for it.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import xml.etree.ElementTree as ET

LAYER1_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parents[2]

# Mirrors layer1_pipeline.py's own sys.path.insert for its RAG package --
# makes layer1_pipeline/placement_eval importable regardless of cwd.
sys.path.insert(0, str(LAYER1_DIR))
# world_model is a separate ROS2 (ament_python) package; its own build_environment
# module has no rclpy dependency (just yaml/numpy/PIL/stdlib xml), so adding its
# source directory directly to sys.path makes it importable without sourcing the
# ROS workspace at all -- this harness is fully offline.
sys.path.insert(0, str(SRC_DIR / "world_model"))

from layer1_pipeline import decompose_instruction, get_client, MAP_YAML_PATH, SDF_PATH
from placement_eval import OBJECT_NAME_ALIASES
from world_model.build_environment import build_environment_prompt
from world_model.scene_tracking import get_tracked_names

EVAL_LOG = LAYER1_DIR / "eval_runs.jsonl"

CANONICAL_A = "place green cube on top of red cube"
CANONICAL_B = "Arrange the green cube, the red cube, the blue cube, and the yellow cube in a triangle"

PARAPHRASES_A = [
    "Stack the green cube on top of the red cube.",
    "Put the green cube on the red one.",
    "I'd like the green cube placed above the red cube, please.",
    "Take the green cube and set it down on top of the red cube.",
    "Green cube goes on top of red cube.",
]

PARAPHRASES_B = [
    "Set up the green, red, blue, and yellow cubes in a triangle formation.",
    "Could you arrange the four cubes -- green, red, blue, and yellow -- into a triangle?",
    "Put the red, blue, yellow, and green cubes together to form a triangle shape.",
    "I want the green, red, blue, and yellow cubes arranged as a triangle.",
    "Form a triangle out of the green cube, red cube, blue cube, and yellow cube.",
]


# ---------------------------------------------------------------------------
# Frozen environment snapshot -- SDF-derived, not a live Gazebo/perception
# read, so it's identical every time this script runs.
# ---------------------------------------------------------------------------

def build_spawn_object_map(sdf_path: str) -> dict:
    """Deterministic 'spawn pose' snapshot for every tracked object/zone/robot
    in panda_world.sdf, parsed directly from the SDF's own declared <pose>
    (or <include><pose>). Same SDF file in, same object_map out, every time --
    no running simulation required."""
    tree = ET.parse(sdf_path)
    world = tree.getroot().find("world")
    tracked_objects, tracked_zones, tracked_robots = get_tracked_names(sdf_path)
    tracked = tracked_objects | tracked_zones | tracked_robots

    def _parse_pose(text: str) -> dict:
        x, y, z = (float(v) for v in text.split()[:3])
        return {"x": x, "y": y, "z": z}

    object_map = {}
    for model in world.findall("model"):
        name = model.get("name")
        pose_el = model.find("pose")
        if name in tracked and pose_el is not None and pose_el.text:
            object_map[name] = _parse_pose(pose_el.text)
    for include in world.findall("include"):
        name_el = include.find("name")
        pose_el = include.find("pose")
        name = name_el.text.strip() if name_el is not None and name_el.text else None
        if name in tracked and pose_el is not None and pose_el.text:
            object_map[name] = _parse_pose(pose_el.text)
    return object_map


def build_frozen_environment() -> str:
    object_map = build_spawn_object_map(SDF_PATH)
    return build_environment_prompt(
        map_yaml_path=MAP_YAML_PATH, sdf_path=SDF_PATH, object_map=object_map
    )


# ---------------------------------------------------------------------------
# Reference plans
# ---------------------------------------------------------------------------

def load_reference_subtasks(instruction_text: str) -> list:
    """First eval_runs.jsonl record whose instruction matches exactly ->
    plans[0]['subtasks'] (the first decomposition, before any replan)."""
    with open(EVAL_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("instruction") == instruction_text:
                return rec["plans"][0]["subtasks"]
    raise RuntimeError(f"No eval_runs.jsonl record found for instruction {instruction_text!r}")


# ---------------------------------------------------------------------------
# Structural signature + comparison
# ---------------------------------------------------------------------------

def normalize_object(name):
    return OBJECT_NAME_ALIASES.get(name, name)


def topo_order(subtasks: list) -> list:
    indeg = {t["id"]: 0 for t in subtasks}
    children = {t["id"]: [] for t in subtasks}
    for t in subtasks:
        for dep in t["dependencies"]:
            children[dep].append(t["id"])
            indeg[t["id"]] += 1
    ready = sorted(tid for tid, d in indeg.items() if d == 0)
    order = []
    while ready:
        tid = ready.pop(0)
        order.append(tid)
        for c in sorted(children[tid]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort()
    return order


def compute_signature(subtasks: list) -> dict:
    order = topo_order(subtasks)
    task_map = {t["id"]: t for t in subtasks}
    by_object = {}
    for tid in order:
        t = task_map[tid]
        skill = t["required_skills"][0]
        if skill not in ("pick", "place"):
            continue
        obj = t["args"].get("object_name")
        if not obj:
            continue
        by_object.setdefault(normalize_object(obj), []).append(t)

    sig = {}
    for obj, tasks in by_object.items():
        place_args = None
        for t in tasks:
            if t["required_skills"][0] == "place":
                a = t["args"]
                place_args = {
                    "landmark": a.get("landmark"),
                    "relative_to": normalize_object(a["relative_to"]) if a.get("relative_to") else None,
                    "direction": a.get("direction"),
                    "distance": float(a["distance"]) if a.get("distance") is not None else None,
                    "target_location": a.get("target_location"),
                }
        sig[obj] = {
            "skill_sequence": [t["required_skills"][0] for t in tasks],
            "robots": [t["robot"] for t in tasks],
            "place_args": place_args,
        }
    return sig


def compare_signatures(ref_sig: dict, cand_sig: dict):
    diffs = []
    ref_objects, cand_objects = set(ref_sig), set(cand_sig)
    if ref_objects != cand_objects:
        diffs.append({
            "object": None, "field": "object_set",
            "reference_value": sorted(ref_objects), "candidate_value": sorted(cand_objects),
        })
        return False, diffs

    for obj in sorted(ref_objects):
        r, c = ref_sig[obj], cand_sig[obj]
        if r["skill_sequence"] != c["skill_sequence"]:
            diffs.append({"object": obj, "field": "skill_sequence",
                           "reference_value": r["skill_sequence"], "candidate_value": c["skill_sequence"]})
        if r["robots"] != c["robots"]:
            diffs.append({"object": obj, "field": "robots",
                           "reference_value": r["robots"], "candidate_value": c["robots"]})
        r_pa, c_pa = r["place_args"] or {}, c["place_args"] or {}
        for key in ("landmark", "relative_to", "direction", "distance", "target_location"):
            if r_pa.get(key) != c_pa.get(key):
                diffs.append({"object": obj, "field": f"place_args.{key}",
                               "reference_value": r_pa.get(key), "candidate_value": c_pa.get(key)})
    return (len(diffs) == 0), diffs


# ---------------------------------------------------------------------------
# Call plan
# ---------------------------------------------------------------------------

def build_call_plan() -> list:
    plan = []
    plan += [{"canonical": "A", "type": "baseline", "instruction": CANONICAL_A} for _ in range(3)]
    plan += [{"canonical": "A", "type": "paraphrase", "instruction": p} for p in PARAPHRASES_A]
    plan += [{"canonical": "B", "type": "baseline", "instruction": CANONICAL_B} for _ in range(3)]
    plan += [{"canonical": "B", "type": "paraphrase", "instruction": p} for p in PARAPHRASES_B]
    return plan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("Building frozen environment snapshot (SDF spawn poses, not live Gazebo)...")
    print("=" * 90)
    environment = build_frozen_environment()
    print(environment)

    print("Loading reference plans from", EVAL_LOG)
    ref_sigs = {
        "A": compute_signature(load_reference_subtasks(CANONICAL_A)),
        "B": compute_signature(load_reference_subtasks(CANONICAL_B)),
    }
    canonical_text = {"A": CANONICAL_A, "B": CANONICAL_B}

    client = get_client()
    call_plan = build_call_plan()

    results = []
    for i, call in enumerate(call_plan, 1):
        print(f"\n[{i}/{len(call_plan)}] canonical={call['canonical']} type={call['type']}")
        print(f"  instruction: {call['instruction']!r}")
        stats = {}
        record = {
            "index": i,
            "canonical": call["canonical"],
            "type": call["type"],
            "instruction": call["instruction"],
        }
        start = time.monotonic()
        try:
            plan, _G = decompose_instruction(
                call["instruction"], client, environment=environment, stats=stats
            )
            elapsed = time.monotonic() - start
            cand_sig = compute_signature(plan["subtasks"])
            match, diffs = compare_signatures(ref_sigs[call["canonical"]], cand_sig)
            record.update({
                "status": "ok", "match": match, "diffs": diffs, "signature": cand_sig,
                "plan_subtasks": plan["subtasks"], "llm_calls": stats.get("llm_calls"),
                "elapsed_s": round(elapsed, 3), "error": None,
            })
            print(f"  -> match={match}  llm_calls={stats.get('llm_calls')}  time={elapsed:.1f}s")
            for d in diffs:
                print(f"     DIFF [{d['object']}] {d['field']}: "
                      f"ref={d['reference_value']!r} vs cand={d['candidate_value']!r}")
        except RuntimeError as e:
            elapsed = time.monotonic() - start
            record.update({
                "status": "error", "match": False, "diffs": [], "signature": None,
                "plan_subtasks": None, "llm_calls": stats.get("llm_calls"),
                "elapsed_s": round(elapsed, 3), "error": str(e),
            })
            print(f"  -> FAILED: {e}")
        results.append(record)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    summary = {}
    for canon, label in (("A", "green-on-red"), ("B", "triangle")):
        baseline = [r for r in results if r["canonical"] == canon and r["type"] == "baseline"]
        paraphrase = [r for r in results if r["canonical"] == canon and r["type"] == "paraphrase"]
        b_matched = sum(1 for r in baseline if r["match"])
        p_matched = sum(1 for r in paraphrase if r["match"])
        b_rate = b_matched / len(baseline) if baseline else float("nan")
        p_rate = p_matched / len(paraphrase) if paraphrase else float("nan")

        print(f"\nCanonical {canon} ({label}): {canonical_text[canon]!r}")
        print(f"  {p_matched}/{len(paraphrase)} paraphrases matched, "
              f"{b_matched}/{len(baseline)} baseline-repeats matched")

        if p_rate >= b_rate:
            verdict = "paraphrase rate >= baseline rate -- no evidence of phrasing sensitivity."
        else:
            gap_pp = (b_rate - p_rate) * 100
            verdict = (f"paraphrase rate is {gap_pp:.0f} percentage points below baseline -- "
                       "read as directional evidence of phrasing sensitivity, not a "
                       "statistical conclusion given the n=3/n=5 sample sizes.")
        print(f"  {verdict}")

        summary[canon] = {
            "instruction": canonical_text[canon],
            "paraphrases_matched": p_matched, "paraphrases_total": len(paraphrase),
            "baseline_matched": b_matched, "baseline_total": len(baseline),
            "verdict": verdict,
        }

    out_path = Path(__file__).resolve().parent / (
        f"phrasing_robustness_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "environment": environment,
            "reference_signatures": ref_sigs,
            "results": results,
            "summary": summary,
        }, f, indent=2)
    print(f"\nRaw results (all {len(results)} records + signatures) written to {out_path}")


if __name__ == "__main__":
    main()
