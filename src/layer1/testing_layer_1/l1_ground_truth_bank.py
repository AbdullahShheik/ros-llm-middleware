"""
l1_ground_truth_bank.py
------------------------
Your own instruction set, each annotated with the CORRECT decomposition
(skill sequence + dependency structure), so SR/IPA/DSR can be scored
against a known-correct answer -- exactly the role DART-LLM's dataset
`output` field plays for their 102 instructions.

Skill names and param keys below match your actual robot_skills.json
exactly (pick: object_name, location | place: object_name, target_location |
release: no inputs | push: object_name, direction, distance_cm |
inspect: object_name | rotate: object_name, angle_deg, axis | home: no inputs).

Note: your registry has no dedicated "stack" skill -- "stack X on Y" is
expressed as `place` with a target_location describing the stacking spot
(e.g. "on_blue_block"), since `place` is generic ("move to target location
then release held object").

Only "ok" (clear + feasible) instructions belong here, since SR/IPA/DSR/SGSR
are about decomposition correctness -- not meaningful for instructions that
should be rejected. Keep rejection-accuracy testing (clarification_needed /
infeasible) as the separate metric from your earlier evaluate.py; don't mix
it into these DART-style numbers.

Ground truth shape per instruction:
  "subtasks": [
      {"id": "T1", "skill": "<SKILL_NAME>", "params": {...}, "depends_on": []},
      ...
  ]
"""

GROUND_TRUTH_INSTRUCTIONS = [
    # ---------------- Simple linear (2-step: pick + place) ----------------
    {
        "id": "gt_001",
        "category": "Simple Linear",
        "instruction": "Pick up the red block and place it on the shelf.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "shelf"}, "depends_on": ["T1"]},
            ]
        },
    },
    {
        "id": "gt_002",
        "category": "Simple Linear",
        "instruction": "Pick up the blue block and place it in the bin.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "blue_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "blue_block", "target_location": "bin"}, "depends_on": ["T1"]},
            ]
        },
    },
    {
        "id": "gt_003",
        "category": "Simple Linear",
        "instruction": "Move the green cube from the table to the shelf.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "green_cube", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "green_cube", "target_location": "shelf"}, "depends_on": ["T1"]},
            ]
        },
    },

    # ---------------- Paraphrase robustness (same ground truth as gt_001, different wording) ----------------
    {
        "id": "gt_004",
        "category": "Paraphrase Robustness",
        "instruction": "Grab the red block and set it down on the shelf.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "shelf"}, "depends_on": ["T1"]},
            ]
        },
    },
    {
        "id": "gt_005",
        "category": "Paraphrase Robustness",
        "instruction": "The red block needs to end up on the shelf, please handle that.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "shelf"}, "depends_on": ["T1"]},
            ]
        },
    },
    {
        "id": "gt_006",
        "category": "Paraphrase Robustness",
        "instruction": "Could you take the red block over to the shelf and leave it there?",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "shelf"}, "depends_on": ["T1"]},
            ]
        },
    },

    # ---------------- Multi-branch (parallel, independent sub-goals) ----------------
    {
        "id": "gt_007",
        "category": "Multi-branch",
        "instruction": "Pick up the red block and place it on the shelf, "
                       "and separately, pick up the blue block and place it in the bin.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "shelf"}, "depends_on": ["T1"]},
                {"id": "T3", "skill": "pick", "params": {"object_name": "blue_block", "location": "table"}, "depends_on": []},
                {"id": "T4", "skill": "place", "params": {"object_name": "blue_block", "target_location": "bin"}, "depends_on": ["T3"]},
            ]
        },
    },
    {
        "id": "gt_008",
        "category": "Multi-branch",
        "instruction": "Stack the red block on the blue block, and at the same time "
                       "move the green block onto the shelf.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "on_blue_block"}, "depends_on": ["T1"]},
                {"id": "T3", "skill": "pick", "params": {"object_name": "green_block", "location": "table"}, "depends_on": []},
                {"id": "T4", "skill": "place", "params": {"object_name": "green_block", "target_location": "shelf"}, "depends_on": ["T3"]},
            ]
        },
    },

    # ---------------- Longer chain (3-4 step) ----------------
    {
        "id": "gt_009",
        "category": "Longer Chain",
        "instruction": "Pick up the red block, place it on the blue block, then pick up the "
                       "green block and place it on top of the red block.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "on_blue_block"}, "depends_on": ["T1"]},
                {"id": "T3", "skill": "pick", "params": {"object_name": "green_block", "location": "table"}, "depends_on": ["T2"]},
                {"id": "T4", "skill": "place", "params": {"object_name": "green_block", "target_location": "on_red_block"}, "depends_on": ["T3"]},
            ]
        },
    },
    {
        "id": "gt_010",
        "category": "Longer Chain",
        "instruction": "Move the red block to the shelf, then move the blue block to where "
                       "the red block used to be.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "shelf"}, "depends_on": ["T1"]},
                {"id": "T3", "skill": "pick", "params": {"object_name": "blue_block", "location": "table"}, "depends_on": ["T2"]},
                {"id": "T4", "skill": "place", "params": {"object_name": "blue_block", "target_location": "red_block_original_position"}, "depends_on": ["T3"]},
            ]
        },
    },

    # ---------------- Single-step (baseline floor case) ----------------
    {
        "id": "gt_011",
        "category": "Single Step",
        "instruction": "Pick up the red block.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
            ]
        },
    },
    {
        "id": "gt_012",
        "category": "Single Step",
        "instruction": "Release the block you're holding.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "release", "params": {}, "depends_on": []},
            ]
        },
    },

    # ---------------- Broader skill coverage (push, inspect, rotate, home) ----------------
    {
        "id": "gt_013",
        "category": "Other Skills",
        "instruction": "Push the blue block 10cm to the left.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "push", "params": {"object_name": "blue_block", "direction": "left", "distance_cm": 10}, "depends_on": []},
            ]
        },
    },
    {
        "id": "gt_014",
        "category": "Other Skills",
        "instruction": "Inspect the red block to check its condition.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "inspect", "params": {"object_name": "red_block"}, "depends_on": []},
            ]
        },
    },
    {
        "id": "gt_015",
        "category": "Other Skills",
        "instruction": "Pick up the green block, rotate it 90 degrees, then place it on the shelf.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "green_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "rotate", "params": {"object_name": "green_block", "angle_deg": 90, "axis": "z"}, "depends_on": ["T1"]},
                {"id": "T3", "skill": "place", "params": {"object_name": "green_block", "target_location": "shelf"}, "depends_on": ["T2"]},
            ]
        },
    },
    {
        "id": "gt_016",
        "category": "Other Skills",
        "instruction": "Place the red block on the shelf, then return the arm to its home position.",
        "ground_truth": {
            "subtasks": [
                {"id": "T1", "skill": "pick", "params": {"object_name": "red_block", "location": "table"}, "depends_on": []},
                {"id": "T2", "skill": "place", "params": {"object_name": "red_block", "target_location": "shelf"}, "depends_on": ["T1"]},
                {"id": "T3", "skill": "home", "params": {}, "depends_on": ["T2"]},
            ]
        },
    },
]
