import csv
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

# Import core components from your main pipeline
from layer1_pipeline import (
    load_skills,
    SKILLS_FILE,
    build_skill_prompt_block,
    build_prompt,
    call_llm,
    build_dag,
    validate_plan,
    MAX_RETRIES,
    GROQ_API_KEY,
    _next_plan_id
)
from groq import Groq

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

# How many times to run EACH instruction.
# Keep the default low so the suite stays within token and cost budget.
RUNS_PER_INSTRUCTION = int(os.environ.get("LAYER1_EVAL_RUNS_PER_INSTRUCTION", "1"))

# Optional cap on how many instructions to evaluate in one run.
# Set to 0 to evaluate the full list.
MAX_TESTS = int(os.environ.get("LAYER1_EVAL_MAX_TESTS", "0"))

# Stable checkpoint file that can be resumed if evaluation stops early.
CHECKPOINT_CSV = "layer1_evaluation_checkpoint.csv"

# ----------------------------------------------------------------------
# TEST INSTRUCTIONS
#
# Each instruction now carries an "expected_status" which is what a
# CORRECT Layer 1 should return before/instead of DAG construction:
#
#   "ok"                   -> instruction is clear and feasible; a valid
#                             DAG should be produced (possibly after a
#                             cycle-repair retry, e.g. the paradox cases)
#   "clarification_needed" -> instruction has an unresolved referent /
#                             missing detail; Layer 1 should ask for
#                             clarification instead of guessing
#   "infeasible"            -> instruction is well-specified but cannot
#                             be executed (no matching skill, violates
#                             physical/temporal ordering, out-of-domain)
#
# Heterogeneous-robot cases intentionally excluded (single robot only).
# ----------------------------------------------------------------------

TEST_INSTRUCTIONS = [
    # ---------------- Normal / Valid (bigger bucket) ----------------
    {
        "category": "Normal / Valid",
        "instruction": "Pick up the red block and place it on the shelf.",
        "expected_status": "ok",
    },
    {
        "category": "Normal / Valid",
        "instruction": "Pick up the blue block and place it in the bin.",
        "expected_status": "ok",
    },
    {
        "category": "Normal / Valid",
        "instruction": "Move the green cube from the table to the shelf.",
        "expected_status": "ok",
    },
    {
        "category": "Normal / Valid (Multi-branch DAG)",
        "instruction": "Pick up the red block and place it on the shelf, "
                       "and separately, pick up the blue block and place it in the bin.",
        "expected_status": "ok",
    },
    {
        "category": "Normal / Valid (Multi-branch DAG)",
        "instruction": "Stack the red block on the blue block, and at the same time "
                       "move the green block onto the shelf.",
        "expected_status": "ok",
    },

    # ---------------- Paraphrase robustness (same task, 3 wordings) ----------------
    {
        "category": "Paraphrase Robustness",
        "instruction": "Grab the red block and set it down on the shelf.",
        "expected_status": "ok",
    },
    {
        "category": "Paraphrase Robustness",
        "instruction": "The red block needs to end up on the shelf, please handle that.",
        "expected_status": "ok",
    },
    {
        "category": "Paraphrase Robustness",
        "instruction": "Could you take the red block over to the shelf and leave it there?",
        "expected_status": "ok",
    },

    # ---------------- Vague / Ambiguous ----------------
    {
        "category": "Vague / Ambiguous",
        "instruction": "Put the thing over there.",
        "expected_status": "clarification_needed",
    },
    {
        "category": "Vague / Ambiguous",
        "instruction": "Move it to the other spot.",
        "expected_status": "clarification_needed",
    },
    {
        "category": "Vague / Ambiguous",
        "instruction": "Take care of the block.",
        "expected_status": "clarification_needed",
    },

    # ---------------- Partially Ambiguous (one clear part, one not) ----------------
    {
        "category": "Partially Ambiguous",
        "instruction": "Pick up the red block and put it wherever makes sense.",
        "expected_status": "clarification_needed",
    },
    {
        "category": "Partially Ambiguous",
        "instruction": "Place the block on the shelf.",  # which block?
        "expected_status": "clarification_needed",
    },

    # ---------------- Impossible / Out-of-Registry ----------------
    {
        "category": "Impossible / Out-of-Registry",
        "instruction": "Wash the red block with soap and water.",
        "expected_status": "infeasible",
    },
    {
        "category": "Impossible / Out-of-Registry",
        "instruction": "Paint the blue block bright orange.",
        "expected_status": "infeasible",
    },

    # ---------------- Physically Impossible (Missing Preconditions) ----------------
    {
        "category": "Physically Impossible (Missing Preconditions)",
        "instruction": "Place the red block on the shelf without picking it up first.",
        "expected_status": "infeasible",
    },
    {
        "category": "Physically Impossible (Missing Preconditions)",
        "instruction": "Release the block from the gripper before ever grasping it.",
        "expected_status": "infeasible",
    },

    # ---------------- Contradictory / Time Travel ----------------
    {
        "category": "Contradictory / Time Travel",
        "instruction": "Place the red block on the shelf, but before you do that, "
                       "make sure you have already placed it on the shelf.",
        "expected_status": "infeasible",
    },
    {
        "category": "Contradictory / Time Travel",
        "instruction": "Finish placing the block on the shelf before you pick it up.",
        "expected_status": "infeasible",
    },

    # ---------------- Out-of-Domain (Non-Robotic) ----------------
    {
        "category": "Out-of-Domain (Non-Robotic)",
        "instruction": "Write a Python script to calculate the Fibonacci sequence.",
        "expected_status": "infeasible",
    },
    {
        "category": "Out-of-Domain (Non-Robotic)",
        "instruction": "Send an email to my supervisor summarizing today's progress.",
        "expected_status": "infeasible",
    },

    # ---------------- Invalid DAG (should self-repair via retry, still "ok") ----------------
    {
        "category": "Invalid DAG (Circular Dependency)",
        "instruction": "Create a plan with two tasks: T1 is 'Pick up the red block' and "
                       "T2 is 'Place the red block'. You MUST make T1 depend on T2, and "
                       "T2 depend on T1.",
        "expected_status": "ok",
    },
    {
        "category": "Invalid DAG (Physical Paradox)",
        "instruction": "To open the box, you must first pick up the key. But to pick up "
                       "the key, you must first open the box. Plan this sequence.",
        "expected_status": "ok",
    },
]


def evaluate_status(plan_json, expected_status, dag_valid, skill_valid, final_success):
    """
    Compares what Layer 1 actually returned against what it SHOULD have
    returned, given the new status field.

    Backward-compatible: if the pipeline hasn't been updated yet and never
    emits a "status" key, any parsed plan is treated as an implicit "ok".
    """
    actual_status = plan_json.get("status", "ok") if isinstance(plan_json, dict) else "ok"

    if expected_status == "ok":
        correct = (actual_status == "ok") and dag_valid and skill_valid and final_success
    else:
        correct = (actual_status == expected_status)

    return actual_status, correct


def _load_checkpoint_results(filepath):
    """Load previously completed runs so evaluation can resume safely."""
    rows = []
    completed_runs = set()

    if not os.path.exists(filepath):
        return rows, completed_runs

    with open(filepath, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
            completed_runs.add((row.get("Category", ""), row.get("Instruction", ""), row.get("Run", "")))

    return rows, completed_runs


def _append_checkpoint_row(filepath, row, write_header):
    """Persist each run immediately so partial progress is not lost."""
    fieldnames = [
        "Category", "Instruction", "Run", "Expected Status", "Actual Status",
        "JSON Validity", "DAG Validity", "Skill Validity",
        "Number of Attempts", "Correct Behavior", "Errors",
    ]

    with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_single_pass(client, skill_block, robot_config, instruction):
    """
    Runs ONE attempt-sequence (with internal JSON/cycle retries) for a
    single instruction and returns the raw metrics for that pass.
    """
    plan_id = _next_plan_id()
    user_prompt = build_prompt(instruction, skill_block, plan_id)

    attempts = 0
    json_valid = False
    dag_valid = False
    skill_valid = False
    final_success = False
    errors = []
    error_suffix = ""
    last_plan = {}

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        raw = call_llm(client, user_prompt + error_suffix)

        try:
            plan = json.loads(raw)
            json_valid = True
            last_plan = plan
        except json.JSONDecodeError as e:
            json_valid = False
            errors.append(f"JSON Parse Error: {e}")
            error_suffix = f"\n\nYour previous response was not valid JSON. Error: {e}. Output ONLY raw JSON."
            continue

        # If the model rejected the instruction, there's no DAG to build/validate.
        status = plan.get("status", "ok")
        if status != "ok":
            dag_valid = True      # N/A - not counted against it
            skill_valid = True    # N/A - not counted against it
            final_success = True  # a rejection IS the correct terminal state; correctness
                                    # against expectation is judged separately in evaluate_status
            errors = []
            break

        dag_payload = plan.get("plan", plan)  # supports both wrapped and unwrapped schemas
        G = build_dag(dag_payload)
        validation_errors = validate_plan(dag_payload, G, robot_config)

        if validation_errors:
            errors = validation_errors
            skill_valid = not any("INVALID_SKILL" in e for e in errors)
            dag_valid = not any("CYCLE_DETECTED" in e or "INVALID_DEPENDENCY" in e for e in errors)
            error_str = "\n".join(f"  - {e}" for e in errors)
            error_suffix = f"\n\nYour previous plan had these errors. Fix ALL of them:\n{error_str}"
        else:
            json_valid = True
            dag_valid = True
            skill_valid = True
            final_success = True
            break

    return {
        "plan": last_plan,
        "attempts": attempts,
        "json_valid": json_valid,
        "dag_valid": dag_valid,
        "skill_valid": skill_valid,
        "final_success": final_success,
        "errors": errors,
    }


def run_evaluation():
    print("=" * 70)
    print("  LAYER 1 AUTOMATED EVALUATION SUITE (multi-run, rate-based)")
    print("=" * 70)

    robot_config = load_skills(SKILLS_FILE)
    skill_block = build_skill_prompt_block(robot_config)
    client = Groq(api_key=GROQ_API_KEY)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_run_csv = f"layer1_evaluation_runs_{timestamp}.csv"
    summary_csv = f"layer1_evaluation_summary_{timestamp}.csv"

    per_run_results, completed_runs = _load_checkpoint_results(CHECKPOINT_CSV)
    # category -> list of per-run "correct" booleans, plus attempt counts
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "attempts": []})

    for row in per_run_results:
        category = row["Category"]
        category_stats[category]["correct"] += int(row["Correct Behavior"] == "True")
        category_stats[category]["total"] += 1
        category_stats[category]["attempts"].append(int(row["Number of Attempts"]))

    instructions_to_run = TEST_INSTRUCTIONS[:MAX_TESTS] if MAX_TESTS > 0 else TEST_INSTRUCTIONS
    total_calls = len(instructions_to_run) * RUNS_PER_INSTRUCTION
    call_counter = 0

    checkpoint_exists = os.path.exists(CHECKPOINT_CSV) and os.path.getsize(CHECKPOINT_CSV) > 0

    for test in instructions_to_run:
        instruction = test["instruction"]
        category = test["category"]
        expected_status = test["expected_status"]

        for run_idx in range(1, RUNS_PER_INSTRUCTION + 1):
            run_key = (category, instruction, str(run_idx))
            if run_key in completed_runs:
                print(f"\n[SKIP] {category} (run {run_idx}/{RUNS_PER_INSTRUCTION}) already completed")
                continue

            call_counter += 1
            print(f"\n[{call_counter}/{total_calls}] {category} (run {run_idx}/{RUNS_PER_INSTRUCTION})")
            print(f"  Instruction: \"{instruction}\"")

            pass_result = run_single_pass(client, skill_block, robot_config, instruction)

            actual_status, correct = evaluate_status(
                pass_result["plan"],
                expected_status,
                pass_result["dag_valid"],
                pass_result["skill_valid"],
                pass_result["final_success"],
            )

            row = {
                "Category": category,
                "Instruction": instruction,
                "Run": run_idx,
                "Expected Status": expected_status,
                "Actual Status": actual_status,
                "JSON Validity": pass_result["json_valid"],
                "DAG Validity": pass_result["dag_valid"],
                "Skill Validity": pass_result["skill_valid"],
                "Number of Attempts": pass_result["attempts"],
                "Correct Behavior": correct,
                "Errors": " | ".join(pass_result["errors"]) if pass_result["errors"] else "None",
            }
            per_run_results.append(row)
            _append_checkpoint_row(CHECKPOINT_CSV, row, not checkpoint_exists)
            checkpoint_exists = True
            completed_runs.add(run_key)

            category_stats[category]["correct"] += int(correct)
            category_stats[category]["total"] += 1
            category_stats[category]["attempts"].append(pass_result["attempts"])

            print(f"  -> Expected: {expected_status} | Actual: {actual_status} | "
                  f"Correct: {correct} | Attempts: {pass_result['attempts']}")

    # ---------------- Write per-run CSV ----------------
    print("\n" + "=" * 70)
    print(f"  Saving per-run results to {per_run_csv} ...")
    with open(per_run_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Category", "Instruction", "Run", "Expected Status", "Actual Status",
                      "JSON Validity", "DAG Validity", "Skill Validity",
                      "Number of Attempts", "Correct Behavior", "Errors"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_run_results:
            writer.writerow(row)

    # ---------------- Write category summary CSV ----------------
    print(f"  Saving category summary to {summary_csv} ...")
    with open(summary_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Category", "Runs", "Correct", "Success Rate (%)", "Avg Attempts"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for category, stats in category_stats.items():
            rate = 100.0 * stats["correct"] / stats["total"] if stats["total"] else 0.0
            avg_attempts = sum(stats["attempts"]) / len(stats["attempts"]) if stats["attempts"] else 0.0
            writer.writerow({
                "Category": category,
                "Runs": stats["total"],
                "Correct": stats["correct"],
                "Success Rate (%)": round(rate, 1),
                "Avg Attempts": round(avg_attempts, 2),
            })
            print(f"    {category}: {stats['correct']}/{stats['total']} "
                  f"({rate:.1f}%) | avg attempts {avg_attempts:.2f}")

    overall_correct = sum(s["correct"] for s in category_stats.values())
    overall_total = sum(s["total"] for s in category_stats.values())
    overall_rate = 100.0 * overall_correct / overall_total if overall_total else 0.0
    print(f"\n  OVERALL: {overall_correct}/{overall_total} ({overall_rate:.1f}%)")
    print("  -> Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
