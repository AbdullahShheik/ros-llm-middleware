#!/usr/bin/env python3
"""
score_subtasks.py
------------------
Subtask-level binary success-rate scoring for src/layer1/eval_runs.jsonl.

DEFINITION: every subtask dispatched across a run counts as one independent
sample -- including every retry attempt after a replan (a task_id repeated
across plan_001/plan_002/... is NOT deduplicated, each is its own dispatch).
Score 1 if that dispatch's terminal feedback was success/complete, 0 if it
was failed/rejected/infeasible/etc. Success rate = successes / (successes +
failures). Anything never resolved to a terminal outcome (the run aborted
before reaching it) is excluded from both and counted separately as
"incomplete".

TWO KNOWN LIMITATIONS OF THE UNDERLYING LOG SCHEMA (read before trusting
the numbers below -- see the printed WARNINGs for exactly which runs they
affected):

1. stage_failures entries carry no plan_id (it was available as a local
   variable in layer1_pipeline.py's _handle_feedback at capture time, but
   never added to the logged dict). task_id alone is NOT unique across
   plans in the same run -- confirmed concretely in the triangle-arrangement
   sample data, where "T6" is a *different* subtask in plan_003 ("Attach the
   blue cube...") than in plan_004 ("Re-place the yellow cube..."). This
   script resolves the ambiguity with a positional heuristic instead:
   stage_failures[i] is paired with plans[i]. That is only valid because
   MAX_REPLAN_ATTEMPTS == 1 in the current layer1_pipeline.py config, which
   guarantees at most one failure-driven plan transition per plan (so
   len(stage_failures) is always len(plans)-1 or len(plans)). If that
   constant ever changes, or a wave ever produces two failures before a
   replan fires, this pairing can silently misattribute. The real fix is a
   one-line addition to layer1_pipeline.py's stage_failures.append(...) call
   (it already has `plan_id` in scope); not applied here since it wasn't
   asked for -- flagging it for a follow-up.

2. The run record stores only an aggregate "subtasks_completed" COUNT, not
   which specific task_ids succeeded. So "not in stage_failures" alone is
   NOT sufficient evidence of success -- a subtask can simply never have
   been dispatched (e.g. blue_cube's own place subtask in the triangle run:
   the plan aborted on an unrelated yellow_cube failure in an earlier wave,
   so blue_cube's place was still further down the DAG and never published).
   To tell "proven successful" apart from "never reached," this script
   reconstructs the actual execution waves using layer1_pipeline.py's own
   build_dag()/build_resource_constrained_waves() logic (copied below, not
   imported -- importing layer1_pipeline.py itself triggers the RAG
   package's sentence-transformers/torch load at module import time just to
   reach two pure-networkx functions, which is unacceptably heavy for a
   scoring script; keep this copy in sync if that logic ever changes) against
   robot_fleet.json's real per-robot-type capacities. Every subtask in a wave
   strictly before the attributed failure's wave is scored 1 (wave-gating
   guarantees the whole wave already reported success, or the plan would
   never have reached the failure's wave at all). The failed task itself is
   0. Everything in the SAME wave as the failure, or in a LATER wave, is
   "incomplete" -- dispatched-but-unrecorded or never-dispatched
   respectively; the schema cannot distinguish which without per-task
   completion logging this repo doesn't have yet.

As a sanity cross-check, each run's own "subtasks_completed" aggregate is
printed next to this script's computed success count -- a run with no
same-wave-as-failure ambiguity should match closely; a gap tells you how
much irrecoverable ambiguity that run's data has.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("This script requires networkx: pip install networkx", file=sys.stderr)
    sys.exit(1)

LAYER1_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG = LAYER1_DIR / "eval_runs.jsonl"
DEFAULT_FLEET_FILE = LAYER1_DIR / "robot_fleet.json"
DEFAULT_CSV = Path(__file__).resolve().parent / "subtask_scores.csv"

FAILED_STATUSES = {"failed", "failure", "error", "rejected", "infeasible"}


# ---------------------------------------------------------------------------
# Wave reconstruction -- mirrors layer1_pipeline.py's build_dag() /
# build_resource_constrained_waves() exactly (see module docstring for why
# this is a copy, not an import).
# ---------------------------------------------------------------------------

def build_dag(plan: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for task in plan["subtasks"]:
        G.add_node(task["id"], **task)
    for task in plan["subtasks"]:
        for dep_id in task["dependencies"]:
            G.add_edge(dep_id, task["id"])
    return G


def build_resource_constrained_waves(
    G: nx.DiGraph, task_map: dict, fleet_counts: dict, default_capacity: int = 1
) -> list:
    completed = set()
    remaining = set(G.nodes)
    waves = []
    while remaining:
        ready = [
            n for n in remaining
            if all(dep in completed for dep in G.predecessors(n))
        ]
        if not ready:
            waves.append(sorted(remaining))
            break
        by_robot: dict = {}
        for task_id in ready:
            by_robot.setdefault(task_map[task_id]["robot"], []).append(task_id)
        wave = []
        for robot, task_ids in by_robot.items():
            capacity = fleet_counts.get(robot, default_capacity)
            if capacity <= 0:
                capacity = default_capacity
            task_ids.sort(key=lambda tid: (-task_map[tid].get("priority", 0), tid))
            wave.extend(task_ids[:capacity])
        wave.sort()
        waves.append(wave)
        completed.update(wave)
        remaining -= set(wave)
    return waves


# ---------------------------------------------------------------------------
# Complexity tier heuristic (instruction #5) -- simple keyword-based, printed
# per run so it can be sanity-checked by eye rather than trusted blindly.
# ---------------------------------------------------------------------------

CUBE_COLOR_RE = re.compile(r"\b(red|green|blue|yellow)\b", re.IGNORECASE)

# Shape words are split by which experimental tier they signal (see
# SESSION_NOTES.md §6) rather than lumped into one flat pattern-word list --
# "line"/"triangle" are the actual experimental variable, not incidental
# phrasing. Tuned against the 14 records in eval_runs.jsonl as run, not
# meant to generalize past what's actually been tested (e.g. "square" has
# never appeared in a real instruction here; it's grouped with triangle-like
# shapes as a reasonable guess, not a validated bucket).
LINE_WORDS = ("line", "row")
TRIANGLE_LIKE_WORDS = ("triangle", "l-shape", "l shape", "square", "corner")
GENERIC_PATTERN_WORDS = ("pattern", "arrange")


def classify_tier(instruction: str):
    colors = {m.lower() for m in CUBE_COLOR_RE.findall(instruction)}
    n_colors = len(colors)
    lowered = instruction.lower()

    is_line = any(w in lowered for w in LINE_WORDS)
    is_triangle_like = any(w in lowered for w in TRIANGLE_LIKE_WORDS)
    has_pattern_word = (
        is_line or is_triangle_like
        or any(w in lowered for w in GENERIC_PATTERN_WORDS)
    )

    if is_line:
        tier = "pattern_line_3obj" if n_colors == 3 else f"pattern_line_{n_colors}obj"
    elif is_triangle_like:
        if n_colors == 3:
            tier = "pattern_triangle_3obj"
        elif n_colors == 4:
            # Tier 5: the deliberately underspecified 4-cube triangle (no
            # diagonal/centroid primitive exists, so this is the one tier
            # expected to show real ambiguity -- see SESSION_NOTES.md §4).
            tier = "pattern_triangle_4obj_ambiguous"
        else:
            tier = f"pattern_triangle_{n_colors}obj"
    elif has_pattern_word:
        tier = f"pattern_arrangement_{n_colors}obj"
    elif n_colors >= 2:
        tier = "multi_object"
    else:
        tier = "single_pick_place"
    return tier, n_colors, has_pattern_word


# ---------------------------------------------------------------------------
# Per-run scoring
# ---------------------------------------------------------------------------

class Dispatch:
    __slots__ = ("plan_id", "task_id", "robot", "score", "reason")

    def __init__(self, plan_id, task_id, robot, score, reason):
        self.plan_id = plan_id
        self.task_id = task_id
        self.robot = robot
        self.score = score      # 1, 0, or None (incomplete)
        self.reason = reason


def score_run(run: dict, fleet_counts: dict, warnings: list) -> list:
    """Return a list of Dispatch objects covering every subtask across every
    plan in this run."""
    plans = run.get("plans", [])
    stage_failures = run.get("stage_failures", [])
    run_status = run.get("status")
    instruction = run.get("instruction", "")

    if not plans:
        return []

    expected_failure_counts = {len(plans) - 1, len(plans)}
    if len(stage_failures) not in expected_failure_counts:
        warnings.append(
            f"  [plan-pairing heuristic] instruction={instruction!r}: "
            f"{len(stage_failures)} stage_failures entries for {len(plans)} plans "
            f"(expected {sorted(expected_failure_counts)}) -- positional pairing "
            "may be unreliable for this run; verify by hand."
        )

    dispatches = []
    for i, plan in enumerate(plans):
        plan_id = plan.get("plan_id", f"<plan index {i}>")
        subtasks = plan.get("subtasks", [])
        if not subtasks:
            continue
        task_map = {t["id"]: t for t in subtasks}
        is_last_plan = (i == len(plans) - 1)
        failure = stage_failures[i] if i < len(stage_failures) else None

        if failure is None:
            if is_last_plan and run_status == "success":
                for tid, t in task_map.items():
                    dispatches.append(Dispatch(
                        plan_id, tid, t.get("robot"), 1,
                        "run completed successfully (final plan, all waves cleared)",
                    ))
            else:
                warnings.append(
                    f"  [no attributable failure] plan_id={plan_id}, "
                    f"is_last_plan={is_last_plan}, run_status={run_status!r} -- "
                    "marking every subtask in this plan incomplete rather than "
                    "guessing."
                )
                for tid, t in task_map.items():
                    dispatches.append(Dispatch(
                        plan_id, tid, t.get("robot"), None,
                        "no stage_failures entry attributed to this plan, and it "
                        "isn't a successful final plan -- outcome unknown",
                    ))
            continue

        failed_task_id = failure.get("task_id")
        if failed_task_id not in task_map:
            warnings.append(
                f"  [pairing mismatch] plan_id={plan_id}: attributed failure "
                f"task_id={failed_task_id!r} is not one of this plan's own "
                f"subtask ids {sorted(task_map)} -- positional heuristic broke "
                "down here; marking every subtask in this plan incomplete."
            )
            for tid, t in task_map.items():
                dispatches.append(Dispatch(
                    plan_id, tid, t.get("robot"), None,
                    "pairing heuristic mismatch -- see WARNING above",
                ))
            continue

        G = build_dag(plan)
        waves = build_resource_constrained_waves(G, task_map, fleet_counts)
        failed_wave_idx = next(
            (w for w, wave in enumerate(waves) if failed_task_id in wave), None
        )
        if failed_wave_idx is None:
            warnings.append(
                f"  [wave reconstruction] plan_id={plan_id}: could not locate "
                f"{failed_task_id!r} in any reconstructed wave -- marking every "
                "subtask in this plan incomplete."
            )
            for tid, t in task_map.items():
                dispatches.append(Dispatch(
                    plan_id, tid, t.get("robot"), None,
                    "wave reconstruction failed to place the failed task",
                ))
            continue

        for w, wave in enumerate(waves):
            for tid in wave:
                t = task_map[tid]
                if w < failed_wave_idx:
                    dispatches.append(Dispatch(
                        plan_id, tid, t.get("robot"), 1,
                        f"wave {w} < failure's wave {failed_wave_idx} -- "
                        "wave-gating guarantees this wave fully succeeded",
                    ))
                elif tid == failed_task_id:
                    dispatches.append(Dispatch(
                        plan_id, tid, t.get("robot"), 0,
                        "matched stage_failures entry for this plan",
                    ))
                elif w == failed_wave_idx:
                    dispatches.append(Dispatch(
                        plan_id, tid, t.get("robot"), None,
                        "same wave as the failure -- dispatched, but this "
                        "schema doesn't record its individual outcome",
                    ))
                else:
                    dispatches.append(Dispatch(
                        plan_id, tid, t.get("robot"), None,
                        f"wave {w} > failure's wave {failed_wave_idx} -- "
                        "never dispatched (plan aborted first)",
                    ))
    return dispatches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_fleet_counts(fleet_file: Path) -> dict:
    try:
        with open(fleet_file, "r", encoding="utf-8") as f:
            return json.load(f)["fleet"]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        print(f"WARNING: could not load {fleet_file} ({e}); defaulting every "
              "robot type to capacity 1.", file=sys.stderr)
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--fleet", type=Path, default=DEFAULT_FLEET_FILE)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("-v", "--verbose", action="store_true",
                     help="print every individual dispatch's score + reason")
    args = ap.parse_args()

    fleet_counts = load_fleet_counts(args.fleet)

    if not args.log.exists():
        print(f"No such file: {args.log}", file=sys.stderr)
        sys.exit(1)

    runs = []
    with open(args.log, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: {args.log}:{line_no} is not valid JSON ({e}), skipping",
                      file=sys.stderr)

    warnings = []
    per_run_rows = []
    tier_totals = {}  # tier -> {successes, failures, incomplete}
    grand_totals = {"successes": 0, "failures": 0, "incomplete": 0}

    print("=" * 100)
    print("PER-RUN BREAKDOWN")
    print("=" * 100)

    for idx, run in enumerate(runs, 1):
        instruction = run.get("instruction", "")
        tier, n_colors, has_pattern_word = classify_tier(instruction)

        dispatches = score_run(run, fleet_counts, warnings)
        successes = sum(1 for d in dispatches if d.score == 1)
        failures = sum(1 for d in dispatches if d.score == 0)
        incomplete = sum(1 for d in dispatches if d.score is None)
        denom = successes + failures
        success_rate = (successes / denom) if denom else None

        tier_bucket = tier_totals.setdefault(
            tier, {"successes": 0, "failures": 0, "incomplete": 0}
        )
        tier_bucket["successes"] += successes
        tier_bucket["failures"] += failures
        tier_bucket["incomplete"] += incomplete
        grand_totals["successes"] += successes
        grand_totals["failures"] += failures
        grand_totals["incomplete"] += incomplete

        reported_completed = run.get("subtasks_completed")
        reported_dispatched = run.get("subtasks_dispatched")

        print(f"\nRun {idx}: {instruction!r}")
        print(f"  tier={tier}  (distinct colors mentioned={n_colors}, "
              f"pattern word hit={has_pattern_word})")
        print(f"  run status={run.get('status')}  replans_used={run.get('replans_used')}  "
              f"plans={run.get('plan_ids')}")
        print(f"  scored: successes={successes} failures={failures} "
              f"incomplete={incomplete}  "
              f"subtask_success_rate={'n/a' if success_rate is None else f'{success_rate:.3f}'}")
        print(f"  cross-check: this run's own reported subtasks_dispatched="
              f"{reported_dispatched}, subtasks_completed={reported_completed} "
              f"vs. this script's successes={successes} "
              f"(gap = ambiguity this schema can't resolve, see incomplete count/reasons)")

        if args.verbose:
            for d in dispatches:
                label = {1: "SUCCESS", 0: "FAILURE", None: "INCOMPLETE"}[d.score]
                print(f"    [{d.plan_id}:{d.task_id}] {label:10s} ({d.robot}) -- {d.reason}")

        per_run_rows.append({
            "run_index": idx,
            "instruction": instruction,
            "tier": tier,
            "n_colors": n_colors,
            "pattern_word_hit": has_pattern_word,
            "status": run.get("status"),
            "replans_used": run.get("replans_used"),
            "plan_ids": ";".join(run.get("plan_ids", [])),
            "successes": successes,
            "failures": failures,
            "incomplete": incomplete,
            "subtask_success_rate": success_rate,
            "reported_subtasks_dispatched": reported_dispatched,
            "reported_subtasks_completed": reported_completed,
            "llm_calls": run.get("llm_calls"),
            "total_latency_s": run.get("total_latency_s"),
        })

    if warnings:
        print("\n" + "=" * 100)
        print("WARNINGS (schema ambiguity encountered while scoring -- read before trusting the numbers)")
        print("=" * 100)
        for w in warnings:
            print(w)

    print("\n" + "=" * 100)
    print("AGGREGATE BY COMPLEXITY TIER")
    print("=" * 100)
    header = f"{'tier':<22}{'successes':>10}{'failures':>10}{'incomplete':>12}{'success_rate':>14}"
    print(header)
    print("-" * len(header))
    for tier, t in sorted(tier_totals.items()):
        denom = t["successes"] + t["failures"]
        rate = f"{t['successes'] / denom:.3f}" if denom else "n/a"
        print(f"{tier:<22}{t['successes']:>10}{t['failures']:>10}{t['incomplete']:>12}{rate:>14}")

    print("-" * len(header))
    g = grand_totals
    gdenom = g["successes"] + g["failures"]
    grate = f"{g['successes'] / gdenom:.3f}" if gdenom else "n/a"
    print(f"{'TOTAL':<22}{g['successes']:>10}{g['failures']:>10}{g['incomplete']:>12}{grate:>14}")

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()) if per_run_rows else [])
        writer.writeheader()
        writer.writerows(per_run_rows)
    print(f"\nPer-run rows written to {args.csv}")


if __name__ == "__main__":
    main()
