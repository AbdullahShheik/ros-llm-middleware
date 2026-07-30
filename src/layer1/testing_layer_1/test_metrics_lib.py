from dart_metrics_lib import compute_ipa, compute_dsr, compute_sr, compute_rtr

gt = {
    "subtasks": [
        {"id": "T1", "skill": "PICK_UP", "params": {"object": "red_block"}, "depends_on": []},
        {"id": "T2", "skill": "PLACE_ON", "params": {"object": "red_block", "target": "shelf"}, "depends_on": ["T1"]},
    ]
}

def check(name, pred, expected_ipa, expected_dsr, expected_sr):
    ipa = compute_ipa(pred, gt)
    dsr = compute_dsr(pred, gt)
    sr = compute_sr(pred, gt, ipa=ipa, dsr=dsr)
    status = "PASS" if (abs(ipa - expected_ipa) < 1e-6 and abs(dsr - expected_dsr) < 1e-6 and sr == expected_sr) else "FAIL"
    print(f"[{status}] {name}: IPA={ipa:.2f} (exp {expected_ipa}) DSR={dsr:.2f} (exp {expected_dsr}) SR={sr} (exp {expected_sr})")

# Case 1: perfect match
perfect = {
    "subtasks": [
        {"id": "A1", "skill": "PICK_UP", "params": {"object": "red_block"}, "depends_on": []},
        {"id": "A2", "skill": "PLACE_ON", "params": {"object": "red_block", "target": "shelf"}, "depends_on": ["A1"]},
    ]
}
check("Perfect match (different IDs, same structure)", perfect, 1.0, 1.0, 1.0)

# Case 2: missing subtask entirely
missing = {
    "subtasks": [
        {"id": "A1", "skill": "PICK_UP", "params": {"object": "red_block"}, "depends_on": []},
    ]
}
check("Missing PLACE_ON subtask", missing, 0.5, 0.0, 0.0)

# Case 3: correct subtasks, wrong dependency order (reversed edge)
reversed_dep = {
    "subtasks": [
        {"id": "A1", "skill": "PICK_UP", "params": {"object": "red_block"}, "depends_on": ["A2"]},
        {"id": "A2", "skill": "PLACE_ON", "params": {"object": "red_block", "target": "shelf"}, "depends_on": []},
    ]
}
check("Correct skills, reversed dependency", reversed_dep, 1.0, 0.0, 0.0)

# Case 4: extra hallucinated subtask
extra = {
    "subtasks": [
        {"id": "A1", "skill": "PICK_UP", "params": {"object": "red_block"}, "depends_on": []},
        {"id": "A2", "skill": "PLACE_ON", "params": {"object": "red_block", "target": "shelf"}, "depends_on": ["A1"]},
        {"id": "A3", "skill": "WAVE", "params": {}, "depends_on": ["A2"]},
    ]
}
check("Extra hallucinated subtask", extra, 1.0, 1.0, 0.0)  # SR fails due to extra subtask

# Case 5: completely empty prediction (e.g. model rejected an "ok" instruction)
empty = {"subtasks": []}
check("Empty prediction", empty, 0.0, 0.0, 0.0)

# RTR sanity
print(f"\nRTR (tight times [1.0,1.05,0.98,1.02]): {compute_rtr([1.0,1.05,0.98,1.02]):.3f} (expect close to 1.0)")
print(f"RTR (volatile times [0.5,3.0,1.0,4.5]): {compute_rtr([0.5,3.0,1.0,4.5]):.3f} (expect well below 1.0)")
print(f"RTR (single run [1.2]): {compute_rtr([1.2]):.3f} (expect 1.0, undefined variance)")

# ---------------------------------------------------------------
# Real-schema tests: normalize_predicted_plan() with your actual
# layer1_pipeline.py output shape (required_skills list, args dict)
# ---------------------------------------------------------------
from dart_metrics_lib import normalize_predicted_plan

print("\n--- Real schema (required_skills / args) ---")

# Ground truth using your ACTUAL skill names (matches l1_ground_truth_bank.py),
# not the PICK_UP/PLACE_ON placeholders used in the earlier synthetic tests above.
gt_real = {
    "subtasks": [
        {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
        {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "shelf"}, "depends_on": ["T1"]},
    ]
}

def check_real(name, raw_plan, expected_ipa, expected_dsr, expected_sr):
    pred_norm = normalize_predicted_plan(raw_plan)
    ipa = compute_ipa(pred_norm, gt_real)
    dsr = compute_dsr(pred_norm, gt_real)
    sr = compute_sr(pred_norm, gt_real, ipa=ipa, dsr=dsr)
    status = "PASS" if (abs(ipa - expected_ipa) < 1e-6 and abs(dsr - expected_dsr) < 1e-6 and sr == expected_sr) else "FAIL"
    print(f"[{status}] {name}: IPA={ipa:.2f} (exp {expected_ipa}) DSR={dsr:.2f} (exp {expected_dsr}) SR={sr} (exp {expected_sr})")

raw_plan_correct = {
    "plan_id": "plan_001",
    "original_instruction": "Pick up the red block and place it on the shelf.",
    "subtasks": [
        {"id": "T1", "description": "pick up red block", "required_skills": ["pick"],
         "args": {"object_name": "red_block", "location": "table"},
         "dependencies": [], "parallelizable": False, "priority": 0},
        {"id": "T2", "description": "place on shelf", "required_skills": ["place"],
         "args": {"object_name": "red_block", "target_location": "shelf"},
         "dependencies": ["T1"], "parallelizable": False, "priority": 1},
    ],
}
check_real("Real schema, correct plan", raw_plan_correct, 1.0, 1.0, 1.0)

# Predicted subtask lists an extra redundant skill alongside the correct one
raw_plan_extra_skill = {
    "subtasks": [
        {"id": "T1", "required_skills": ["pick", "inspect"],
         "args": {"object_name": "red_block", "location": "table"}, "dependencies": []},
        {"id": "T2", "required_skills": ["place"],
         "args": {"object_name": "red_block", "target_location": "shelf"}, "dependencies": ["T1"]},
    ],
}
check_real("Real schema, redundant extra skill listed (should still match)", raw_plan_extra_skill, 1.0, 1.0, 1.0)

# Predicted subtask MISSING the correct skill from its required_skills list
raw_plan_wrong_skill = {
    "subtasks": [
        {"id": "T1", "required_skills": ["inspect"],  # wrong -- should be "pick"
         "args": {"object_name": "red_block", "location": "table"}, "dependencies": []},
        {"id": "T2", "required_skills": ["place"],
         "args": {"object_name": "red_block", "target_location": "shelf"}, "dependencies": ["T1"]},
    ],
}
check_real("Real schema, wrong skill on T1 (should NOT match)", raw_plan_wrong_skill, 0.5, 0.0, 0.0)


