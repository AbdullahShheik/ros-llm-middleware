#!/usr/bin/env python3
"""
aggregate_metrics.py
---------------------
Cross-run aggregation over eval_runs.jsonl, built after the 14-run batch
completed (SESSION_NOTES.md §7/§10). Two reports:

1. PLACEMENT ACCURACY BY TYPE -- every placement_errors entry across every
   run, classified "flat" (landmark / target_location / relative_to with a
   same-plane direction) vs "stacked" (relative_to + direction=='above'),
   with mean/median/min/max execution_error_m and a 5mm-tolerance breach
   count per group. planning_error_m is checked for exactly 0.0 everywhere
   as a sanity check (it's the geometric-resolution error, independent of
   physical execution -- see SESSION_NOTES.md §1).

2. LLM QUERY COUNT PER SUBTASK, BY TIER -- using the corrected 5-tier
   classify_tier() imported from score_subtasks.py (not reimplemented, so
   tier logic stays in exactly one place). Reports llm_calls per run and
   llm_calls normalized by subtasks_dispatched (queries per subtask), since
   raw llm_calls scales with plan size and isn't comparable across tiers on
   its own. replans_used is reported alongside since it's the direct driver
   of any llm_calls beyond the initial decomposition.

A placement_errors entry doesn't carry a task_id/plan_id back-reference to
the place-subtask that produced it (same schema gap score_subtasks.py works
around for stage_failures -- see its module docstring). This script
resolves it the same way: positional pairing, per object, in dispatch
order. Mismatches are surfaced as warnings rather than silently guessed.

Printed report only, prioritizing readability over raw-data completeness --
see score_subtasks.py -v for full per-dispatch detail.
"""

import json
import statistics
import sys
from pathlib import Path

from score_subtasks import classify_tier

LAYER1_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG = LAYER1_DIR / "eval_runs.jsonl"

# Mirrors phrasing_robustness.py's own sys.path.insert for the same import.
sys.path.insert(0, str(LAYER1_DIR))
from placement_eval import OBJECT_NAME_ALIASES  # noqa: E402  (red_block -> red_cube etc.)

TOLERANCE_M = 0.005  # 5mm, matches the place_verify tolerance used throughout the pipeline

# Canonical display order matching the actual experimental tiers
# (SESSION_NOTES.md §6/§10), not alphabetical.
TIER_ORDER = [
    "single_pick_place",
    "multi_object",
    "pattern_line_3obj",
    "pattern_triangle_3obj",
    "pattern_triangle_4obj_ambiguous",
]


def load_runs(log_path: Path) -> list:
    runs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: {log_path}:{line_no} is not valid JSON ({e}), skipping",
                      file=sys.stderr)
    return runs


# ---------------------------------------------------------------------------
# 1. Placement accuracy by type
# ---------------------------------------------------------------------------

def classify_placement_type(args: dict) -> str:
    """stacked iff relative_to is set AND direction == 'above'; everything
    else (landmark, target_location, or relative_to with a same-plane
    direction like left/right/behind) is flat."""
    if args.get("relative_to") and args.get("direction") == "above":
        return "stacked"
    return "flat"


def collect_place_args_by_object(run: dict) -> dict:
    """object_name -> list of place-subtask args, in dispatch order (plan
    order, then subtask order within a plan). Names are normalized through
    OBJECT_NAME_ALIASES ("red_block" -> "red_cube" etc.) since the LLM's own
    plan JSON isn't guaranteed to use the canonical "_cube" naming that
    placement_errors' "object" field uses (confirmed inconsistent in
    practice -- some runs' place args say "red_block", others "red_cube",
    for the same physical object)."""
    by_object = {}
    for plan in run.get("plans", []):
        for t in plan.get("subtasks", []):
            if "place" not in t.get("required_skills", []):
                continue
            args = t.get("args", {})
            obj = args.get("object_name")
            if obj is None:
                continue
            obj = OBJECT_NAME_ALIASES.get(obj, obj)
            by_object.setdefault(obj, []).append(args)
    return by_object


def classify_all_placements(runs: list, warnings: list) -> list:
    rows = []
    for idx, run in enumerate(runs, 1):
        placement_errors = run.get("placement_errors", [])
        if not placement_errors:
            continue
        place_args_by_object = collect_place_args_by_object(run)

        errors_by_object = {}
        for pe in placement_errors:
            obj = OBJECT_NAME_ALIASES.get(pe["object"], pe["object"])
            errors_by_object.setdefault(obj, []).append(pe)

        for obj, pe_list in errors_by_object.items():
            args_list = place_args_by_object.get(obj, [])
            if len(args_list) < len(pe_list):
                warnings.append(
                    f"  [placement-type pairing] run {idx}, object={obj!r}: "
                    f"{len(pe_list)} placement_errors entries but only "
                    f"{len(args_list)} place-subtask args found -- "
                    "classifying the unmatched entries as 'flat' (safe "
                    "default; verify by hand)."
                )
            for i, pe in enumerate(pe_list):
                args = args_list[i] if i < len(args_list) else {}
                rows.append({
                    "run_index": idx,
                    "object": obj,
                    "type": classify_placement_type(args),
                    "planning_error_m": pe.get("planning_error_m"),
                    "execution_error_m": pe.get("execution_error_m"),
                })
    return rows


def print_placement_report(rows: list):
    print("=" * 100)
    print("PLACEMENT ACCURACY BY TYPE")
    print("=" * 100)

    # planning_error_m sanity check across every entry, not just the
    # grouped ones -- flag loudly if this assumption ever breaks.
    bad_planning = [r for r in rows if r["planning_error_m"] != 0.0]
    if bad_planning:
        print(f"\n!!! SANITY CHECK FAILED: {len(bad_planning)} placement_errors "
              f"entries have planning_error_m != 0.0 (expected exactly 0.0 "
              f"everywhere -- geometry resolution should never be the "
              f"source of error). Inspect these before trusting anything "
              f"downstream:")
        for r in bad_planning:
            print(f"    run {r['run_index']}, {r['object']}: "
                  f"planning_error_m={r['planning_error_m']}")
    else:
        print(f"\nSanity check passed: planning_error_m == 0.0 for all "
              f"{len(rows)} placement_errors entries across all runs -- "
              f"geometry resolution is never the source of error.")

    print(f"\n{'type':<10}{'count':>8}{'mean_mm':>12}{'median_mm':>12}"
          f"{'min_mm':>10}{'max_mm':>10}{'breach(>5mm)':>14}")
    print("-" * 76)
    for ptype in ("flat", "stacked"):
        group = [r["execution_error_m"] for r in rows if r["type"] == ptype]
        if not group:
            print(f"{ptype:<10}{0:>8}{'n/a':>12}{'n/a':>12}{'n/a':>10}{'n/a':>10}{0:>14}")
            continue
        breaches = sum(1 for e in group if e > TOLERANCE_M)
        print(f"{ptype:<10}{len(group):>8}"
              f"{statistics.mean(group) * 1000:>12.4f}"
              f"{statistics.median(group) * 1000:>12.4f}"
              f"{min(group) * 1000:>10.4f}"
              f"{max(group) * 1000:>10.4f}"
              f"{breaches:>14}")
    print("\n(execution_error_m shown in mm; breach = execution_error_m > "
          f"{TOLERANCE_M * 1000:.0f}mm, the place_verify tolerance)")


# ---------------------------------------------------------------------------
# 2. LLM query count per subtask, by tier
# ---------------------------------------------------------------------------

def print_llm_report(runs: list):
    print("\n" + "=" * 100)
    print("LLM QUERY COUNT PER SUBTASK, BY TIER")
    print("=" * 100)

    by_tier = {}
    for run in runs:
        tier, _, _ = classify_tier(run.get("instruction", ""))
        llm_calls = run.get("llm_calls")
        dispatched = run.get("subtasks_dispatched")
        replans = run.get("replans_used")
        bucket = by_tier.setdefault(tier, {
            "n_runs": 0, "llm_calls": [], "queries_per_subtask": [], "replans": [],
        })
        bucket["n_runs"] += 1
        if llm_calls is not None:
            bucket["llm_calls"].append(llm_calls)
        if llm_calls is not None and dispatched:
            bucket["queries_per_subtask"].append(llm_calls / dispatched)
        if replans is not None:
            bucket["replans"].append(replans)

    header = (f"{'tier':<34}{'runs':>6}{'mean_llm_calls':>16}"
              f"{'mean_queries/subtask':>22}{'mean_replans_used':>20}")
    print(f"\n{header}")
    print("-" * len(header))

    ordered_tiers = [t for t in TIER_ORDER if t in by_tier]
    ordered_tiers += sorted(t for t in by_tier if t not in TIER_ORDER)

    for tier in ordered_tiers:
        b = by_tier[tier]
        mean_llm = statistics.mean(b["llm_calls"]) if b["llm_calls"] else float("nan")
        mean_qps = statistics.mean(b["queries_per_subtask"]) if b["queries_per_subtask"] else float("nan")
        mean_replans = statistics.mean(b["replans"]) if b["replans"] else float("nan")
        print(f"{tier:<34}{b['n_runs']:>6}{mean_llm:>16.3f}"
              f"{mean_qps:>22.4f}{mean_replans:>20.3f}")

    print("\n(queries/subtask = llm_calls / subtasks_dispatched -- the "
          "cross-tier-comparable figure, since raw llm_calls scales with "
          "plan size; replans_used is the direct driver of any llm_calls "
          "beyond the initial decomposition)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"No such file: {args.log}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(args.log)
    warnings = []

    placement_rows = classify_all_placements(runs, warnings)
    print_placement_report(placement_rows)
    print_llm_report(runs)

    if warnings:
        print("\n" + "=" * 100)
        print("WARNINGS (schema ambiguity encountered while aggregating -- read before trusting the numbers)")
        print("=" * 100)
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
