"""
run_dart_style_evaluation.py
------------------------------
Runs your own ground-truth-annotated instruction bank through your Layer 1
pipeline, computes SR / IPA / DSR / SGSR / RTR (DART-LLM's metric
definitions, scored on YOUR data), and prints a results table in the same
shape as DART-LLM's Table V so your numbers can sit next to theirs.

Usage:
    python run_dart_style_evaluation.py
"""

import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, ".")  # adjust if layer1_pipeline.py lives elsewhere

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
    get_client,
    _next_plan_id,
)

from dart_metrics_lib import (
    normalize_predicted_plan,
    compute_ipa,
    compute_dsr,
    compute_sr,
    compute_sgsr,
    aggregate_metrics,
)
from l1_ground_truth_bank import GROUND_TRUTH_INSTRUCTIONS

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RUNS_PER_INSTRUCTION = 5  # repeat runs needed for a meaningful RTR (variance)


def run_single_pass(client, skill_block, robot_config, instruction):
    """
    One full attempt-sequence (including internal JSON/cycle retries) for a
    single instruction. Times the whole pass for RTR, and returns enough
    info to score IPA/DSR/SR/SGSR against ground truth.
    """
    plan_id = _next_plan_id()
    user_prompt = build_prompt(instruction, skill_block, plan_id)

    error_suffix = ""
    last_plan = {}
    structurally_valid = False

    start_time = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        raw = call_llm(client, user_prompt + error_suffix)

        try:
            plan = json.loads(raw)
            last_plan = plan
        except json.JSONDecodeError as e:
            error_suffix = (
                f"\n\nYour previous response was not valid JSON. "
                f"Error: {e}. Output ONLY raw JSON."
            )
            continue

        status = plan.get("status", "ok")
        if status != "ok":
            # A well-formed rejection is still structurally valid JSON,
            # it's just the wrong call for an instruction we know is "ok" --
            # SGSR reflects well-formedness, SR/IPA/DSR will naturally
            # come out as 0 since there are no subtasks to match.
            structurally_valid = True
            break

        G = build_dag(plan)
        validation_errors = validate_plan(plan, G, robot_config)

        if validation_errors:
            error_str = "\n".join(f"  - {e}" for e in validation_errors)
            error_suffix = (
                f"\n\nYour previous plan had these errors. Fix ALL of them:\n{error_str}"
            )
        else:
            structurally_valid = True
            break

    elapsed = time.perf_counter() - start_time
    return {
        "plan": last_plan,
        "structurally_valid": structurally_valid,
        "elapsed_seconds": elapsed,
    }


def run_evaluation():
    print("=" * 70)
    print("  DART-STYLE METRIC EVALUATION (SR / IPA / DSR / SGSR / RTR)")
    print("  Scored on your own instruction bank, not DART-LLM's dataset")
    print("=" * 70)

    robot_config = load_skills(SKILLS_FILE)
    skill_block = build_skill_prompt_block(robot_config)
    client = get_client()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_run_csv = f"dart_style_runs_{timestamp}.csv"
    summary_csv = f"dart_style_summary_{timestamp}.csv"

    per_run_rows = []
    per_instruction_results = []  # feeds aggregate_metrics()
    category_results = defaultdict(list)  # category -> list of per-instruction dicts

    total_calls = len(GROUND_TRUTH_INSTRUCTIONS) * RUNS_PER_INSTRUCTION
    call_counter = 0

    for item in GROUND_TRUTH_INSTRUCTIONS:
        instruction = item["instruction"]
        category = item["category"]
        gt = item["ground_truth"]

        response_times = []
        ipa_runs, dsr_runs, sgsr_runs = [], [], []

        for run_idx in range(1, RUNS_PER_INSTRUCTION + 1):
            call_counter += 1
            print(f"\n[{call_counter}/{total_calls}] {item['id']} ({category}) run {run_idx}/{RUNS_PER_INSTRUCTION}")
            print(f"  \"{instruction}\"")

            result = run_single_pass(client, skill_block, robot_config, instruction)
            pred_normalized = normalize_predicted_plan(result["plan"])

            ipa = compute_ipa(pred_normalized, gt)
            dsr = compute_dsr(pred_normalized, gt)
            sr = compute_sr(pred_normalized, gt, ipa=ipa, dsr=dsr)
            sgsr = compute_sgsr(result["structurally_valid"])

            response_times.append(result["elapsed_seconds"])
            ipa_runs.append(ipa)
            dsr_runs.append(dsr)
            sgsr_runs.append(sgsr)

            print(f"  -> IPA={ipa:.2f} DSR={dsr:.2f} SR={sr:.0f} SGSR={sgsr:.0f} "
                  f"time={result['elapsed_seconds']:.2f}s")

            per_run_rows.append({
                "Instruction ID": item["id"],
                "Category": category,
                "Run": run_idx,
                "IPA": round(ipa, 3),
                "DSR": round(dsr, 3),
                "SR": sr,
                "SGSR": sgsr,
                "Response Time (s)": round(result["elapsed_seconds"], 3),
            })

        # Average this instruction's runs into one instruction-level result
        # (matches DART-LLM reporting one number per instruction, averaged
        # into per-tier numbers -- see aggregate_metrics()).
        sr_hits = sum(
            1 for i in range(RUNS_PER_INSTRUCTION)
            if ipa_runs[i] == 1.0 and dsr_runs[i] == 1.0
        )
        instr_result = {
            "sr": sr_hits / RUNS_PER_INSTRUCTION,
            "ipa": sum(ipa_runs) / RUNS_PER_INSTRUCTION,
            "dsr": sum(dsr_runs) / RUNS_PER_INSTRUCTION,
            "sgsr": sum(sgsr_runs) / RUNS_PER_INSTRUCTION,
            "response_times": response_times,
        }
        per_instruction_results.append(instr_result)
        category_results[category].append(instr_result)

    # ---------------- Per-run CSV ----------------
    print("\n" + "=" * 70)
    print(f"Saving per-run results to {per_run_csv} ...")
    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Instruction ID", "Category", "Run", "IPA", "DSR", "SR", "SGSR", "Response Time (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_run_rows)

    # ---------------- Category + overall summary CSV ----------------
    print(f"Saving summary to {summary_csv} ...")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Category", "N Instructions", "SR", "IPA", "DSR", "SGSR", "RTR"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for category, results in category_results.items():
            m = aggregate_metrics(results)
            writer.writerow({
                "Category": category,
                "N Instructions": len(results),
                **m,
            })
            print(f"  {category:25s} n={len(results):2d}  "
                  f"SR={m['SR']:.2f} IPA={m['IPA']:.2f} DSR={m['DSR']:.2f} "
                  f"SGSR={m['SGSR']:.2f} RTR={m['RTR']:.2f}")

        overall = aggregate_metrics(per_instruction_results)
        writer.writerow({
            "Category": "OVERALL (all instructions)",
            "N Instructions": len(per_instruction_results),
            **overall,
        })
        print(f"\n  {'OVERALL':25s} n={len(per_instruction_results):2d}  "
              f"SR={overall['SR']:.2f} IPA={overall['IPA']:.2f} DSR={overall['DSR']:.2f} "
              f"SGSR={overall['SGSR']:.2f} RTR={overall['RTR']:.2f}")

    print("\n" + "=" * 70)
    print("  Compare the OVERALL row above against DART-LLM's L1 row:")
    print("  DART-LLM L1 (best case, e.g. GPT-4o): SR=1.00 IPA=1.00 DSR=1.00 SGSR=1.00 RTR=0.55")
    print("  (Your single-robot arm-manipulation domain vs. their construction-robot")
    print("   domain -- report this as 'comparable metric, different instruction set',")
    print("   not as literally the same benchmark.)")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
