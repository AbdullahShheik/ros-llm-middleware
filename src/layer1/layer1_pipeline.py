"""
Layer 1: NLI -> Subtask Decomposition -> DAG -> Validation
Following the DART-LLM paper architecture.

Single robotic arm, no environment locked yet.
Uses Groq API with Llama 3.3 70B.

Usage:
  # Interactive mode (prompts you for instruction):
  python layer1_pipeline.py

  # Single instruction via command line:
  python layer1_pipeline.py "Pick up the red block and place it on the shelf"

  # ROS2 node mode (publishes to /layer1/taskplan):
  python layer1_pipeline.py --ros
  latest 2.0
"""

import json
import os
import sys
import argparse
import itertools
import time
from datetime import datetime, timezone
import networkx as nx
from groq import Groq
# NOTE: world_model.build_environment is deliberately NOT imported here.
# It pulls in yaml/numpy/PIL and is only ever needed in --ros mode, so
# run_ros_node() imports it locally instead -- that keeps standalone CLI
# and the offline evaluation harnesses runnable without those deps (and
# without a sourced ROS workspace).

# The RAG package lives beside this file; make it importable regardless of
# the working directory the pipeline is launched from (ROS2 launch, the demo
# scripts, and the evaluation harnesses all use different ones).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RAG.few_shot_bank import bank_summary, load_examples
from RAG.few_shot_retriever import CosineFewShotRetriever, format_examples_for_prompt
from RAG.robot_bank import bank_summary as robot_bank_summary, load_robot_bank, FLEET_FILE
from RAG.robot_retriever import CosineRobotTypeRetriever, union_skills

def _load_env_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_load_env_file(os.path.join(_PROJECT_ROOT, ".env"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY")
MODEL        = "llama-3.3-70b-versatile"
MAX_RETRIES  = 3
# How many times a single original instruction may be replanned after a
# subtask failure before Layer1Node gives up and aborts (see
# Layer1Node._handle_feedback). Bounded to stop a failure that reliably
# recurs (e.g. a truly unreachable target) from replanning forever.
MAX_REPLAN_ATTEMPTS = 1
SKILLS_FILE  = os.path.join(os.path.dirname(__file__), "robot_skills.json")
# How many few-shot examples get retrieved into each prompt. The bank itself
# can grow freely -- only these k reach the prompt.
FEW_SHOT_K   = 4
# Live environment context (ROS2 mode only) -- see build_environment.py.
# Computed from _PROJECT_ROOT rather than hardcoded per-machine, so this
# works regardless of whose checkout or OS it's running on.
MAP_YAML_PATH = os.path.join(_PROJECT_ROOT, "src", "world", "maps", "panda_world_map.yaml")
SDF_PATH      = os.path.join(_PROJECT_ROOT, "src", "world", "worlds", "panda_world.sdf")
# One JSON-lines record per user instruction (see Layer1Node._finalize_run):
# success/failure, subtask completion counts, LLM call count + latency, and
# the full plan(s) produced, for evaluation metrics (success rate, subtasks
# completed, LLM query efficiency, plan quality, latency) without having to
# scrape them out of console logs by hand.
RUN_LOG_PATH  = os.path.join(os.path.dirname(__file__), "eval_runs.jsonl")

# get_client() returns the LLM client. Currently only Groq is supported.


def get_client():
    """Return a Groq client for LLM calls."""
    return Groq(api_key=GROQ_API_KEY)


_plan_counter = itertools.count(1)

def _next_plan_id() -> str:
    return f"plan_{next(_plan_counter):03d}"



# STEP 1 — Robot skills (retrieved, not hardcoded)
# R + S components in P = (I, E, R, S, F)
#
# Which robot TYPES are relevant to an instruction is selected per-instruction
# from the active fleet (robot_fleet.json + robots_registry.json +
# master_skills.json, see RAG/robot_bank.py and RAG/robot_retriever.py).
# Retrieval only ever picks whole robot types, never individual skills: once
# a type is selected, its COMPLETE registered skill set -- each already
# resolved to its full description/inputs/preconditions/effects from
# master_skills.json -- goes into the prompt. The union of the selected
# types' skills is what reaches the prompt and the validator.

_robot_retriever = None


def get_robot_retriever() -> CosineRobotTypeRetriever:
    """Build the retriever once and reuse it -- vectorizing the robot bank
    per call would redo the same work on every instruction."""
    global _robot_retriever
    if _robot_retriever is None:
        bank = load_robot_bank()
        _robot_retriever = CosineRobotTypeRetriever(bank)
        print(f"[Layer 1] Robot bank loaded: {robot_bank_summary(bank)}")
    return _robot_retriever


def retrieve_robot_config(instruction: str) -> dict:
    """Returns the {"skills": [...]} robot_config for this instruction --
    the union of the selected robot types' own complete skill sets. Shape
    matches what build_skill_prompt_block and validate_plan already expect."""
    selected = get_robot_retriever().retrieve(instruction)
    print("[Layer 1] Retrieved robot types: " + ", ".join(
        f"{entry['robot_type']}({score:.2f})" for entry, score in selected
    ))
    return union_skills(selected)


# Legacy full-catalog loading. Not used by decompose_instruction below
# (which retrieves per-instruction instead) -- kept only because other
# scripts (evaluate.py, evaluate_final.py, multi_robot_test.py,
# testing_layer_1/run_dart_style_evaluation.py) still import these names.
# NOTE: robot_skills.json no longer exists -- those scripts will fail at
# the load_skills(SKILLS_FILE) call until they're moved onto the RAG bank too.
SKILLS_FILE = os.path.join(os.path.dirname(__file__), "robot_skills.json")


def load_skills(filepath: str) -> dict:
    """Load robot skill definitions from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def build_skill_prompt_block(robot_config: dict) -> str:
    """
    Convert the skill list into a readable string injected into the LLM prompt.

    Each skill's "robot_types" (which selected robot type(s) actually offer
    it -- see union_skills) is shown here because Layer 1 now assigns the
    "robot" field itself (see validate_plan Check 4): the model needs to see
    which type owns a skill to fill that field correctly, not guess it.
    """
    lines = ["Available Skills:"]
    for skill in robot_config["skills"]:
        inputs_str = ", ".join(skill["inputs"]) if skill["inputs"] else "none"
        pre_str    = "; ".join(skill["preconditions"]) if skill["preconditions"] else "none"
        eff_str    = "; ".join(skill["effects"])       if skill["effects"]       else "none"
        robots_str = ", ".join(skill["robot_types"])
        lines.append(
            f"\n  name         : {skill['name']}"
            f"\n  description  : {skill['description']}"
            f"\n  robot_types  : [{robots_str}]"
            f"\n  inputs       : [{inputs_str}]"
            f"\n  preconditions: {pre_str}"
            f"\n  effects      : {eff_str}"
        )
    return "\n".join(lines)


# STEP 2 — Few-shot examples (retrieved, not hardcoded)
# F component in P = (I, E, R, S, F)
#
# Examples live in few_shot_examples.json and are selected per-instruction by
# cosine similarity (see RAG/few_shot_retriever.py). Only the top-k reach the
# prompt, so the bank can grow without inflating every request. Retrieval
# guarantees at least one rejection example and one combined arm+mobile
# example are always present.

_retriever = None


def get_retriever() -> CosineFewShotRetriever:
    """Build the retriever once and reuse it -- fitting TF-IDF per call would
    redo the same work on every instruction."""
    global _retriever
    if _retriever is None:
        examples = load_examples()
        _retriever = CosineFewShotRetriever(examples)
        print(f"[Layer 1] Few-shot bank loaded: {bank_summary(examples)}")
    return _retriever


def retrieve_few_shot_examples(instruction: str, k: int = FEW_SHOT_K):
    """Returns the (example, score) pairs selected for this instruction."""
    return get_retriever().retrieve(instruction, k=k)


# STEP 3 — Prompt builder
# Follows DART-LLM's P = (I, E, R, S, F)


SYSTEM_PROMPT = """You are a task decomposition planner for a multi-robot system consisting of a robotic arm and a mobile robot (TurtleBot).
Your job is to take a high-level natural language instruction and break it down
into atomic subtasks that the appropriate robot can execute.

Rules you must follow:
1. BEFORE decomposing, classify the instruction into exactly one status:
   "ok", "clarification_needed", or "infeasible".
2. Use "ok" only when the instruction is clear, physically executable, and can be
   decomposed using only the provided skills (arm or mobile robot).
3. Use "clarification_needed" when an unresolved referent such as "it", "the thing",
   or "over there" cannot be grounded in the environment or skill registry. Return
   only status and reason; do not produce a plan.
4. Use "infeasible" when the instruction is unambiguous but cannot be executed
   by either the arm or the mobile robot: outside the robot domain, missing required skills, physically impossible,
   temporally impossible, or self-contradictory. Return only status and reason;
   do not produce a plan.
5. For status "ok", required_skills must only contain skill names from the provided
   skill list. No other values allowed.
6. Every subtask's "robot" field must be one of that subtask's skill's own
   robot_types (shown per-skill in === ROBOT SKILLS ===). If a skill lists more
   than one robot_type, pick whichever one is doing the rest of that subtask's
   chain in this plan. Never write a robot type that isn't in the skill's own
   robot_types list.
7. Every subtask must have a unique id in the format T1, T2, T3 ...
8. dependencies must only reference ids of other subtasks in the same plan.
9. If a subtask has no prerequisites, set dependencies to an empty list [].
10. Think step by step about the physical actions needed and their order before writing JSON.
11. Output ONLY the raw JSON. No explanation, no markdown, no code fences.
12. If the instruction gives an explicit target position for a "place" skill (e.g.
   coordinates like "0.3, 0.4" or "(0.3, -0.2)"), copy that value into
   target_location verbatim, exactly as written. Do NOT paraphrase it into a
   generic placeholder like "target_position" -- the executor parses this string
   directly and a placeholder cannot be resolved to a real pose.
13. When an instruction acts on multiple distinct objects (e.g. "place the red
   block on the shelf and place the green block on the shelf"), each object's
   subtask chain is independent by default. Do NOT add a dependency from one
   object's chain to another's unless the instruction says one must wait for
   the other, or the physical action requires it (e.g. stacking object B onto
   object A). Multiple robots may be available to work on independent chains
   at the same time -- an unnecessary dependency prevents that.
14. Each object in === ENVIRONMENT === is tagged reachable_by_arm: true or
   false. Before an arm subtask chain (pick -> place) acts on an object
   tagged false, prepend a transport chain: navigate_to(the object's zone)
   -> attach(object_name) -> navigate_to(handoff_point) -> detach, and make
   the pick subtask depend on that chain's detach subtask. If tagged true,
   go straight to pick -> place with no transport steps.
15. Never compute coordinates for a "place" skill, exactly as in rule 12 --
   use one of these instead:
   - landmark: one of top_left, top_right, bottom_left, bottom_right, center
     -- a named point in the arm's workspace. Use this for the first object
     in an arrangement, or any object placed at a described workspace
     position with no stated relation to another object.
   - relative_to + direction + distance: relative_to is another tracked
     object's name; direction is one of left, right (beside it), front,
     behind, or above, below (stacked on top of / underneath it); distance
     is an integer count of cube-widths (cube-heights for above/below),
     default 1. Use this for "beside X", "in front of X", "on top of X",
     etc., and for every object after the first in a multi-object
     arrangement (a shape, a line, a corner layout, a stack) -- decompose
     the arrangement into a chain, each object relative_to an earlier one,
     reasoning about directions the way you would describe the arrangement
     in words. Never populate both landmark and relative_to on one subtask.
   - left/right move only along one axis, front/behind only along another,
     above/below only along the third -- so two objects placed relative_to
     the SAME anchor using directions from the SAME axis (e.g. one "left"
     of it and another "right" of it) always end up collinear with it,
     which is a LINE, never a shape with a bend or a corner in it (a
     triangle, an L, etc). For a shape that isn't a straight line, at least
     one object needs an offset from a DIFFERENT axis than the others.
   - Prefer anchoring every object relative_to the SAME shared anchor
     (using a different axis per object, as needed to avoid collinearity)
     over chaining an object relative_to another object that is itself
     relative_to something -- only chain through a second object when the
     shape genuinely calls for a position defined off that specific object
     (e.g. "stack this one on top of the second one"), not merely to avoid
     collinearity. Chaining through an intermediate object forces that
     object to be placed before the next one even can be, which idles
     whichever robot delivered the next object for no reason when placing
     it relative to the shared anchor instead would have let it proceed
     immediately. Example -- a line of 3: A at a landmark, B relative_to A
     direction=left, C relative_to A direction=right. A triangle of 3
     instead: A at a landmark, B relative_to A direction=left, C
     relative_to A (not B) direction=behind -- a different axis than B
     used, so C is still off the line A-B, but B and C can now each be
     placed as soon as they arrive, independently of one another.
16. Rule 13's independence-by-default does NOT apply once a "place" subtask
   uses relative_to referencing another object -- that object's real
   resting position must be known first, so add a dependencies entry on
   its own place subtask (skip this only if relative_to names a fixed
   environment location that nothing ever moves).
17. If the instruction text begins with "Original goal:" and includes
   "Progress so far" and "This step failed", you are continuing a
   partially-executed plan, not starting a fresh one. Do not repeat
   subtasks already listed as completed, and take the stated failure
   reason into account when planning the remaining work. The "Progress
   so far" list is context, not something you can depend on: every id in
   it belongs to the OLD plan and does not exist in the one you are
   writing now. Never put one of those ids in a new subtask's
   dependencies -- rule 8 still applies (dependencies must only reference
   ids of other subtasks in THIS plan). If a new subtask's only
   prerequisite is something already listed as completed, that
   prerequisite is already satisfied -- give it an empty dependencies
   list, don't reference the old id.
18. A relative_to reference must itself be at a known, reachable
   position before it means anything -- you cannot place an object
   relative to another one that is still sitting wherever it happened
   to start. If a place subtask's relative_to names an object tagged
   reachable_by_arm: false, and nothing else in the plan already moves
   that object to a reachable position (its own pick -> place chain,
   with a transport chain first per rule 14), add that chain for it
   too, even though the instruction never explicitly asked you to move
   it -- give its own place subtask a landmark (or a named location if
   the instruction gives one) as its target. This applies on top of
   rule 16's dependency requirement, not instead of it.
19. There are two independent robotic arms (robotic_arm, robotic_arm_2),
   each with identical pick/place capability and its own gripper -- not
   one arm with two slots. A single object's own pick and place subtasks
   must use the SAME arm (the object is physically held between those two
   steps). Across DIFFERENT objects' independent chains (rule 13), prefer
   spreading them across both arms rather than assigning every chain to
   the same one, so the two arms actually work concurrently instead of
   one sitting idle. This does not override rule 16: a chain that depends
   on another object's place subtask completing first still waits for
   it regardless of which arm either one uses."""


def build_prompt(instruction: str,
                 skill_block: str,
                 plan_id: str,
                 robot_types: list[str],
                 environment: str = None,
                 few_shot_k: int = FEW_SHOT_K) -> str:
    """
    Assembles the full prompt following DART-LLM's P = (I, E, R, S, F).
    Environment is a placeholder for now — will be replaced by
    a live object map topic subscription later.

    The F component is retrieved per-instruction rather than fixed: the
    few_shot_k examples most similar to this instruction, plus the coverage
    guarantees enforced by the retriever.

    robot_types is the R component's active set for this instruction (from
    retrieve_robot_config) -- the schema's "robot" placeholder is built from
    it instead of a hardcoded literal, so it always matches whatever robot
    types were actually retrieved, however many the fleet grows to.
    """
    env_section = environment if environment else (
        "Environment: Not yet specified. "
        "Use generic location names such as object_position, target_location, home_position."
    )

    retrieved = retrieve_few_shot_examples(instruction, k=few_shot_k)
    few_shot_block = format_examples_for_prompt(retrieved)
    print("[Layer 1] Retrieved few-shot examples: " + ", ".join(
        f"{ex['id']}({score:.2f})" for ex, score in retrieved
    ))

    robot_types_placeholder = " | ".join(robot_types)

    schema = f"""For status "ok", output this flat structure:
{{
  "status": "ok",
  "plan_id": "{plan_id}",
  "original_instruction": "<copy the instruction exactly>",
  "subtasks": [
    {{
      "id": "<T1, T2, T3 ...>",
      "description": "<what this subtask does in plain English>",
      "required_skills": ["<exactly one skill name from the skill list>"],
      "robot": "<{robot_types_placeholder}>",
      "args": {{ "<input_name>": "<value>" }},
      "dependencies": ["<ids of subtasks that must complete before this one, or empty list>"],
      "parallelizable": "<true if this can run in parallel with other independent subtasks>",
      "priority": "<integer, lower = higher priority>"
    }}
  ]
}}

For status "clarification_needed", output this structure and no plan:
{{
  "status": "clarification_needed",
  "reason": "<short reason naming what is ambiguous>"
}}

For status "infeasible", output this structure and no plan:
{{
  "status": "infeasible",
  "reason": "<short reason naming why it cannot be executed>"
}}"""

    return f"""
=== ROBOT SKILLS ===
{skill_block}

=== ENVIRONMENT ===
{env_section}

=== OUTPUT SCHEMA ===
First classify the instruction, then output exactly one of these JSON structures.
Use this plan_id for status "ok": {plan_id}
{schema}

=== FEW-SHOT EXAMPLES ===
{few_shot_block}

=== YOUR TASK ===
Decompose the following instruction:
Instruction: "{instruction}"

Output ONLY the raw JSON. Nothing else.
"""

# STEP 4 — LLM call
def call_llm(client: Groq, user_prompt: str) -> str:
    """Send prompt to Groq and return raw response text."""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content


# STEP 5 — DAG builder
def build_dag(plan: dict) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from subtask dependency lists.
    Node  = subtask id
    Edge A→B = A must complete before B starts
    """
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
    """Build execution waves directly from the whole DAG, respecting
    per-robot-type fleet capacity (robot_fleet.json) at every step.

    An earlier version of this split each nx.topological_generations()
    wave independently and concatenated the results. That is NOT enough:
    it fully drains one whole generation's sub-waves (e.g. every chain's
    *first* navigate_to) before touching the next generation (every
    chain's attach) at all -- observed live as a capacity-1 mobile robot
    reaching one cube's zone, then leaving for the next chain's zone
    without ever attaching, because the attach for the first cube doesn't
    become eligible until every chain's navigate_to has already run.

    Instead, at every wave this looks at ALL currently-ready tasks
    (dependencies already satisfied by an earlier wave), and when more
    tasks of one robot type are ready than that type's fleet capacity
    allows, prefers whichever task is FURTHEST INTO an already-started
    chain (highest `priority` -- the LLM already numbers this 1, 2, 3...
    per chain, per rule 9) over one just starting a new chain, so a
    constrained robot finishes what it started before beginning something
    else. Different robot types draw from independent capacity pools, so
    e.g. the arm placing one object and the mobile robot starting the next
    chain can still share a wave.

    Safe by construction: a task only ever enters `ready` once every
    predecessor is in `completed`, and `completed` only grows by whole
    waves -- matching Layer1Node's existing wave-gated dispatch (advance
    only once every task in the current wave reports "complete"), so this
    changes WHICH waves get built, not the gating semantics."""
    completed = set()
    remaining = set(G.nodes)
    waves = []

    while remaining:
        ready = [
            n for n in remaining
            if all(dep in completed for dep in G.predecessors(n))
        ]
        if not ready:
            # Shouldn't happen for a validated acyclic plan -- avoid an
            # infinite loop if it somehow does rather than hanging.
            waves.append(sorted(remaining))
            break

        by_robot: dict[str, list] = {}
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


# STEP 6 — Validator
# Four checks following DART-LLM's validation approach

def validate_plan(plan: dict, G: nx.DiGraph, robot_config: dict) -> list[str]:
    """
    Validates the plan. Returns list of error strings.
    Empty list = valid plan.

    Check 1: No cycles in DAG
    Check 2: No dangling dependency references
    Check 3: All required_skills exist in skill registry
    Check 4: Every subtask has a valid 'robot' field matching its skill

    Check 4 has no hardcoded skill->robot table: robot_config["robot_types"]
    (the valid values) and each skill's own "robot_types" (who may run it)
    both come from union_skills, straight off the robot types retrieved for
    this instruction -- so it stays correct as robot_fleet.json/
    robots_registry.json gain new types, with nothing here to update by hand.
    """
    errors = []
    valid_skills      = {s["name"] for s in robot_config["skills"]}
    skill_robot_types = {s["name"]: s["robot_types"] for s in robot_config["skills"]}
    valid_robot_types = robot_config["robot_types"]
    task_ids          = {t["id"] for t in plan["subtasks"]}

    # Check 1 — Cycle detection
    if not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
            errors.append(
                f"CYCLE_DETECTED: dependency graph has a cycle: {cycle}. "
                f"No task may depend on its own successor."
            )
        except nx.NetworkXNoCycle:
            errors.append("CYCLE_DETECTED: cycle found but could not be identified.")

    # Check 2 — Dangling references
    for task in plan["subtasks"]:
        for dep in task["dependencies"]:
            if dep not in task_ids:
                errors.append(
                    f"INVALID_DEPENDENCY: task '{task['id']}' depends on '{dep}' "
                    f"which does not exist. Valid ids: {sorted(task_ids)}"
                )

    # Check 3 — Skill validity
    for task in plan["subtasks"]:
        for skill in task["required_skills"]:
            if skill not in valid_skills:
                errors.append(
                    f"INVALID_SKILL: task '{task['id']}' lists skill '{skill}' "
                    f"which is not in the skill registry. "
                    f"Valid skills: {sorted(valid_skills)}"
                )

    # Check 4 — Robot field validity and skill-robot consistency
    for task in plan["subtasks"]:
        robot = task.get("robot")
        if robot not in valid_robot_types:
            errors.append(
                f"MISSING_ROBOT: task '{task['id']}' has no valid 'robot' field "
                f"(got '{robot}'). Must be one of: {valid_robot_types}."
            )
            continue

        skill = task["required_skills"][0] if task["required_skills"] else None
        owners = skill_robot_types.get(skill)
        if owners and robot not in owners:
            errors.append(
                f"WRONG_ROBOT: task '{task['id']}' uses skill '{skill}' "
                f"which is only available on {owners}, but got robot='{robot}'."
            )

    return errors

# STEP 7 — Full pipeline with retry loop
# Mirrors DART-LLM Algorithm 2 re-prompting on failure

def decompose_instruction(
    instruction: str,
    client: Groq,
    environment: str = None,
    stats: dict = None,
) -> tuple[dict, nx.DiGraph]:
    """
    Full Layer 1 pipeline:
    NLI → prompt → LLM → parse JSON → build DAG → validate → retry if needed.
    Returns (plan_dict, dag) on success.
    Raises RuntimeError if all retries are exhausted.

    Robot skills (the R+S components of P = (I, E, R, S, F)) are retrieved
    per-instruction from the active fleet rather than passed in statically --
    see retrieve_robot_config.

    stats, if given, gets 'llm_calls' incremented once per actual call_llm
    invocation (including ones that fail JSON parsing/validation and retry)
    -- an optional out-parameter rather than a return-signature change, so
    every existing caller (CLI mode included) is unaffected.
    """
    robot_config = retrieve_robot_config(instruction)
    skill_block  = build_skill_prompt_block(robot_config)
    plan_id      = _next_plan_id()
    user_prompt  = build_prompt(
        instruction, skill_block, plan_id, robot_config["robot_types"], environment
    )
    error_suffix = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n[Layer 1] Attempt {attempt}/{MAX_RETRIES} — calling LLM...")

        raw = call_llm(client, user_prompt + error_suffix)
        if stats is not None:
            stats["llm_calls"] = stats.get("llm_calls", 0) + 1

        # Parse JSON
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [ERROR] JSON parse failed: {e}")
            error_suffix = (
                f"\n\nYour previous response was not valid JSON. "
                f"Error: {e}. Output ONLY a raw JSON object."
            )
            continue

        status = str(plan.get("status", "ok")).strip().lower()
        if status in {"clarification_needed", "infeasible"}:
            reason = plan.get("reason", "No reason provided.")
            print(f"  [STOP] {status}: {reason}")
            raise RuntimeError(f"{status}: {reason}")

        if status != "ok":
            print(f"  [ERROR] Invalid status: {status}")
            error_suffix = (
                "\n\nYour previous response had an invalid status. "
                "Set status to exactly one of: ok, clarification_needed, infeasible."
            )
            continue

        # Always use the plan_id we generated, not whatever the LLM wrote
        plan["plan_id"] = plan_id

        # Build DAG
        G = build_dag(plan)

        # Validate
        errors = validate_plan(plan, G, robot_config)

        if not errors:
            # Layer 2 (robot assignment) doesn't exist yet, and every skill in
            # robot_skills.json is arm-only for now, so default every subtask
            # to the arm so the dispatcher's required 'robot' field is always
            # present. Revisit once a wheeled skill is added to the registry.
            for subtask in plan["subtasks"]:
                subtask.setdefault("robot", "arm")

            print(f"  [OK] Valid plan — {len(plan['subtasks'])} subtasks, "
                  f"{G.number_of_edges()} dependency edges.")
            return plan, G

        # Feed errors back to LLM and retry
        error_str = "\n".join(f"  - {e}" for e in errors)
        print(f"  [INVALID] {len(errors)} error(s):\n{error_str}")
        error_suffix = (
            f"\n\nYour previous plan had these errors. Fix ALL of them:\n{error_str}"
        )

    raise RuntimeError(
        f"Could not produce a valid plan after {MAX_RETRIES} attempts."
    )


# STEP 8 — Pretty-print DAG to terminal

def print_dag(plan: dict, G: nx.DiGraph):
    """Print a human-readable execution plan summary."""
    print("\n" + "═" * 60)
    print("  EXECUTION PLAN")
    print("═" * 60)
    print(f"  Plan ID    : {plan['plan_id']}")
    print(f"  Instruction: {plan['original_instruction']}")
    print(f"  Subtasks   : {len(plan['subtasks'])}")
    print(f"  DAG edges  : {G.number_of_edges()}")
    print("─" * 60)

    task_map = {t["id"]: t for t in plan["subtasks"]}
    try:
        ordered_ids = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        ordered_ids = [t["id"] for t in plan["subtasks"]]

    for tid in ordered_ids:
        task = task_map[tid]
        dep_str = f"after {task['dependencies']}" if task["dependencies"] \
                  else "no dependencies (starts immediately)"
        print(f"\n  [{tid}] {task['description']}")
        print(f"         skill : {task['required_skills']}")
        print(f"         args  : {task['args']}")
        print(f"         order : {dep_str}")

    print("\n" + "─" * 60)
    print("  EXECUTION WAVES (tasks in same wave run in parallel):")
    print("─" * 60)
    for i, wave in enumerate(nx.topological_generations(G), start=1):
        wave_list = sorted(wave)
        print(f"  Wave {i}: {wave_list}")
        for tid in wave_list:
            desc = task_map[tid]["description"]
            print(f"    {tid} → {desc}")
    print("═" * 60)


# STEP 9 — ROS2 publisher node (optional mode)
# Publishes validated subtasks to /layer1/taskplan as std_msgs/String


def run_ros_node():
    """
    ROS2 node mode.
    Subscribes to /layer1/instruction (std_msgs/String)
    Subscribes to /execution_feedback    (std_msgs/String)
    Publishes to  /layer1/taskplan    (std_msgs/String)

    Publishes one topological generation at a time. Tasks in the same
    generation are sent back-to-back so they can run in parallel. The next
    dependent generation is published only after feedback arrives for every
    task in the current generation.

    To test from terminal:
      ros2 topic pub /layer1/instruction std_msgs/String "data: 'Pick up the red block'"
      ros2 topic echo /layer1/taskplan
      ros2 topic pub /layer1/feedback std_msgs/String "data: '{\"task_id\":\"T1\",\"status\":\"done\"}'"
    """
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError:
        print("[ERROR] rclpy not found. Run inside a ROS2 environment.")
        sys.exit(1)

    # Imported lazily (not at module level) for the same reason as rclpy
    # above: world_model.build_environment pulls in PIL/numpy/yaml, and
    # reaching it at all requires a sourced ROS workspace -- none of which
    # should be required just to run standalone/CLI mode.
    try:
        from world_model.build_environment import build_environment_prompt
    except ImportError as e:
        print(f"[ERROR] world_model not importable ({e}). "
              "Build the workspace and source install/setup.bash.")
        sys.exit(1)

    class Layer1Node(Node):
        def __init__(self):
            super().__init__("layer1_node")
            self.client       = get_client()
            self.latest_object_map = {}        
            self.active_plan = None
            self.active_graph = None
            self.task_map = {}
            self.execution_waves = []
            self.current_wave_index = 0
            self.pending_feedback = set()
            self.completed_tasks = set()
            # How many times the active *original* instruction has been
            # replanned after a subtask failure. Reset only in
            # instruction_callback (a genuinely new instruction) -- NOT in
            # clear_active_plan(), since that also runs at the end of a
            # successful replanned continuation and the cap must survive
            # across it, or a failure could trigger unbounded replanning.
            self.replan_attempts = 0
            # Latest parsed payload from /object_map -- {"red_cube": {"x":.., "y":.., "z":..}, ...}.
            # Empty until perception publishes at least once (see object_map_callback).
            self.latest_object_map = {}
            # Evaluation run record for the active *original* instruction --
            # see _finalize_run. None whenever no instruction is in flight.
            self.current_run = None

            self.publisher_ = self.create_publisher(
                String, "/layer1/taskplan", 10
            )
            self.subscription = self.create_subscription(
                String, "/layer1/instruction", self.instruction_callback, 10
            )
            self.feedback_subscription = self.create_subscription(
                String, "/execution_feedback", self.feedback_callback, 10
            )
            self.object_map_subscription = self.create_subscription(
                String, "/object_map", self.object_map_callback, 10
            )
            self.get_logger().info(
                "Layer 1 node ready. "
                "Listening on /layer1/instruction, "
                "listening on /execution_feedback, "
                "listening on /object_map, "
                "publishing to /layer1/taskplan."
            )
            self.get_logger().info(f"Map context: {MAP_YAML_PATH}")
            self.get_logger().info(f"SDF world:   {SDF_PATH}")

        def object_map_callback(self, msg: String):
            try:
                self.latest_object_map = json.loads(msg.data)
            except json.JSONDecodeError as e:
                self.get_logger().warn(f"Failed to parse /object_map message: {e}")

        def instruction_callback(self, msg: String):
            instruction = msg.data.strip()
            if not instruction:
                return

            if self.active_plan is not None:
                self.get_logger().warn(
                    "Ignoring new instruction because plan "
                    f"{self.active_plan['plan_id']} is still awaiting feedback."
                )
                return

            self.replan_attempts = 0
            self.get_logger().info(f"Received instruction: {instruction}")
            self.current_run = {
                "original_instruction": instruction,
                "model": MODEL,
                "start_wall": datetime.now(timezone.utc).isoformat(),
                "start_monotonic": time.monotonic(),
                "llm_calls": 0,
                "decompose_calls": [],
                "replans_used": 0,
                "subtasks_completed_before_replan": 0,
                "plans": [],
            }
            if not self._decompose_and_dispatch(instruction):
                # Initial decomposition itself failed (clarification_needed,
                # infeasible, or all MAX_RETRIES exhausted) -- no plan was
                # ever dispatched, so none of this node's other three
                # termination points (publish_current_wave's success
                # branch, _handle_feedback's abort, _trigger_replan's
                # abort) will ever run for this run; finalize it here.
                self._finalize_run("failed")

        def _decompose_and_dispatch(self, instruction: str) -> bool:
            """Shared by a fresh instruction and a bounded replan
            continuation (see _handle_feedback) -- both are just "call the
            LLM with this instruction text and this moment's environment,
            then dispatch whatever plan comes back." Returns whether a plan
            was successfully dispatched, so a replan caller can fall back
            to aborting instead of leaving the old (failed) plan dangling."""
            if not self.latest_object_map:
                self.get_logger().warn(
                    "No /object_map data received yet -- environment context will "
                    "have no live object or zone positions. "
                    "Is the perception node running?"
                )
            # Created before the try (not inside it) and folded into
            # self.current_run in BOTH the success and except paths below --
            # decompose_instruction can raise (clarification_needed,
            # infeasible, retries exhausted) after already making one or
            # more real LLM calls, and those must still count toward the
            # llm_calls metric, not just the calls from an eventual success.
            stats = {}
            call_start = time.monotonic()
            try:
                environment = build_environment_prompt(
                    map_yaml_path=MAP_YAML_PATH,
                    sdf_path=SDF_PATH,
                    object_map=self.latest_object_map,
                )
                plan, G = decompose_instruction(
                    instruction, self.client, environment=environment, stats=stats
                )
                if self.current_run is not None:
                    self.current_run["llm_calls"] += stats.get("llm_calls", 0)
                    self.current_run["decompose_calls"].append({
                        "plan_id": plan["plan_id"],
                        "latency_s": round(time.monotonic() - call_start, 3),
                        "llm_calls": stats.get("llm_calls", 0),
                        "subtasks": len(plan["subtasks"]),
                    })
                    self.current_run["plans"].append(plan)
                print_dag(plan, G)
                self.start_plan_dispatch(plan, G)
                return True
            except Exception as e:
                self.get_logger().error(f"Decomposition failed: {e}")
                if self.current_run is not None:
                    self.current_run["llm_calls"] += stats.get("llm_calls", 0)
                    self.current_run["decompose_calls"].append({
                        "plan_id": None,
                        "latency_s": round(time.monotonic() - call_start, 3),
                        "llm_calls": stats.get("llm_calls", 0),
                        "subtasks": 0,
                    })
                    self.current_run.setdefault("decompose_errors", []).append(str(e))
                return False

        def start_plan_dispatch(self, plan: dict, G: nx.DiGraph):
            self.active_plan = plan
            self.active_graph = G
            self.task_map = {task["id"]: task for task in plan["subtasks"]}

            # Fleet capacity isn't accounted for by the LLM's own DAG (rule
            # 13 gives independent objects independent, parallel-eligible
            # chains by default) -- e.g. two different cubes' `attach`
            # subtasks can land in the same wave even though there's only
            # one mobile robot. build_resource_constrained_waves accounts
            # for it directly rather than trusting the plan to get it
            # right; this is what actually prevents two concurrent
            # /attach_cube calls from racing against attach_detach_node.py's
            # single `attached_cube` slot -- and, just as importantly,
            # keeps a capacity-1 robot finishing one chain before starting
            # another instead of wandering between every chain's first step.
            try:
                with open(FLEET_FILE, "r", encoding="utf-8") as f:
                    fleet_counts = json.load(f)["fleet"]
            except (OSError, json.JSONDecodeError, KeyError) as e:
                self.get_logger().warn(
                    f"Could not load fleet capacity from {FLEET_FILE} ({e}); "
                    "defaulting every robot type to capacity 1."
                )
                fleet_counts = {}

            self.execution_waves = build_resource_constrained_waves(
                G, self.task_map, fleet_counts
            )
            self.current_wave_index = 0
            self.pending_feedback = set()
            self.completed_tasks = set()

            self.get_logger().info(
                f"Dispatching plan {plan['plan_id']} as "
                f"{len(self.execution_waves)} execution wave(s)."
            )
            self.publish_current_wave()

        def publish_current_wave(self):
            if self.active_plan is None:
                return

            if self.current_wave_index >= len(self.execution_waves):
                self.get_logger().info(
                    f"Plan {self.active_plan['plan_id']} complete; "
                    "all subtasks received feedback."
                )
                self._finalize_run("success")
                self.clear_active_plan()
                return

            wave = self.execution_waves[self.current_wave_index]
            self.pending_feedback = set(wave)
            wave_number = self.current_wave_index + 1

            self.get_logger().info(
                f"Publishing wave {wave_number}/"
                f"{len(self.execution_waves)}: {wave}"
            )

            for task_id in wave:
                subtask = dict(self.task_map[task_id])
                subtask["plan_id"] = self.active_plan["plan_id"]

                out_msg = String()
                out_msg.data = json.dumps(subtask)
                self.publisher_.publish(out_msg)
                self.get_logger().info(
                    f"Published subtask {task_id} from plan "
                    f"{self.active_plan['plan_id']} to /layer1/taskplan"
                )

            if not wave:
                self.current_wave_index += 1
                self.publish_current_wave()

        def feedback_callback(self, msg: String):
            try:
                self._handle_feedback(msg)
            except Exception as e:
                # As in instruction_callback: an uncaught exception here
                # would propagate out of rclpy.spin() and kill the node,
                # permanently wedging every instruction after this one.
                self.get_logger().error(f"Error handling feedback: {e}")

        def _handle_feedback(self, msg: String):
            feedback = msg.data.strip()
            if not feedback:
                return

            self.get_logger().info(f"Received feedback: {feedback}")

            if self.active_plan is None:
                self.get_logger().warn(
                    "Feedback received, but no plan is currently active."
                )
                return

            plan_id, task_id, status, stage = self.parse_feedback(feedback)

            if plan_id and plan_id != self.active_plan["plan_id"]:
                self.get_logger().warn(
                    f"Ignoring feedback for plan {plan_id}; active plan is "
                    f"{self.active_plan['plan_id']}."
                )
                return

            if task_id is None:
                if len(self.pending_feedback) == 1:
                    task_id = next(iter(self.pending_feedback))
                else:
                    self.get_logger().warn(
                        "Could not match feedback to a task. Include one of "
                        "'task_id', 'subtask_id', or 'id' in feedback JSON."
                    )
                    return

            if status in {"failed", "failure", "error", "rejected", "infeasible"}:
                if self.replan_attempts < MAX_REPLAN_ATTEMPTS:
                    self.replan_attempts += 1
                    self.get_logger().warn(
                        f"Feedback reported {status} for task {task_id} in plan "
                        f"{self.active_plan['plan_id']}; attempting replan "
                        f"{self.replan_attempts}/{MAX_REPLAN_ATTEMPTS} instead of "
                        "aborting."
                    )
                    self._trigger_replan(feedback, task_id)
                    return

                self.get_logger().error(
                    f"Feedback reported {status} for task {task_id}; "
                    f"aborting plan {self.active_plan['plan_id']} "
                    f"(replan attempts exhausted: {self.replan_attempts}/"
                    f"{MAX_REPLAN_ATTEMPTS})."
                )
                self._finalize_run("failed")
                self.clear_active_plan()
                return

            # The actuator publishes multiple intermediate progress
            # messages per task (e.g. "arm_motion", "gripper") before its
            # final "complete" stage. Only "complete" means the task is
            # actually done — treating any success feedback as done would
            # advance to the next wave/instruction while the actuator is
            # still mid-sequence on this one.
            if stage != "complete":
                self.get_logger().info(
                    f"Progress for task {task_id}: stage={stage} status={status}"
                )
                return

            if task_id not in self.pending_feedback:
                if task_id in self.completed_tasks:
                    self.get_logger().info(
                        f"Ignoring duplicate feedback for task {task_id}."
                    )
                else:
                    self.get_logger().warn(
                        f"Feedback for task {task_id} is not expected in "
                        f"current wave; waiting for "
                        f"{sorted(self.pending_feedback)}."
                    )
                return

            self.pending_feedback.remove(task_id)
            self.completed_tasks.add(task_id)
            self.get_logger().info(
                f"Feedback accepted for task {task_id}. Remaining in wave: "
                f"{sorted(self.pending_feedback)}"
            )

            if not self.pending_feedback:
                self.current_wave_index += 1
                self.publish_current_wave()

        def _trigger_replan(self, raw_feedback: str, failed_task_id: str):
            """Build a continuation instruction (original goal + what's
            done + what failed + why) and re-run decomposition for the
            remaining work, in place of aborting outright. Reuses
            decompose_instruction/start_plan_dispatch as-is -- this is a
            new trigger path, not new plan-execution machinery.

            Must read everything it needs from self.active_plan/task_map/
            completed_tasks BEFORE calling _decompose_and_dispatch, since
            that calls start_plan_dispatch which overwrites all of them
            with the new (continuation) plan's state."""
            original_instruction = self.active_plan.get(
                "original_instruction", "(original instruction unavailable)"
            )
            failed_task = self.task_map.get(failed_task_id, {})

            try:
                detail = json.loads(raw_feedback).get("detail", raw_feedback)
            except (json.JSONDecodeError, AttributeError):
                detail = raw_feedback

            completed_lines = [
                f"  - {tid}: {self.task_map[tid].get('description', tid)}"
                for tid in sorted(self.completed_tasks)
                if tid in self.task_map
            ]
            completed_block = "\n".join(completed_lines) if completed_lines else "  (none)"

            # Snapshot before _decompose_and_dispatch -> start_plan_dispatch
            # resets self.completed_tasks for the continuation plan; without
            # this, subtasks completed under the ORIGINAL (failed) plan
            # would be silently dropped from the final subtasks_completed
            # count in _finalize_run.
            if self.current_run is not None:
                self.current_run["subtasks_completed_before_replan"] += len(
                    self.completed_tasks
                )
                self.current_run["replans_used"] += 1

            replan_instruction = (
                f"Original goal: \"{original_instruction}\"\n\n"
                f"Progress so far: The following subtasks already completed "
                f"successfully:\n{completed_block}\n\n"
                f"This step failed and blocked the plan:\n"
                f"  - {failed_task_id}: {failed_task.get('description', failed_task_id)} "
                f"(required_skills={failed_task.get('required_skills')}, "
                f"robot={failed_task.get('robot')}, args={failed_task.get('args')})\n"
                f"  Failure reason: {detail}\n\n"
                f"Produce a plan to complete the REMAINING goal from the current "
                f"state. Do not repeat subtasks that already completed "
                f"successfully. Take the failure reason into account."
            )

            plan_id_for_log = self.active_plan.get("plan_id", "?")
            if not self._decompose_and_dispatch(replan_instruction):
                # The old (failed) plan is still sitting in self.active_plan
                # at this point -- _decompose_and_dispatch only overwrites it
                # on success (inside start_plan_dispatch). Left as-is, it
                # would permanently block every future instruction (see the
                # active_plan-not-None guard in instruction_callback), so an
                # unsuccessful replan must still fall through to an abort.
                self.get_logger().error(
                    f"Replan attempt for plan {plan_id_for_log} itself failed to "
                    "produce a plan; aborting."
                )
                self._finalize_run("failed")
                self.clear_active_plan()

        def _finalize_run(self, status: str):
            """Append one JSON-lines record for the just-finished original
            instruction to RUN_LOG_PATH, for the evaluation metrics
            described at RUN_LOG_PATH's definition. Called from this
            node's four termination points -- instruction_callback (initial
            decompose failed outright), publish_current_wave (all waves
            completed), _handle_feedback (replan cap exhausted), and
            _trigger_replan (the replan attempt itself failed to produce a
            plan) -- always BEFORE clear_active_plan()/start_plan_dispatch()
            reset the state this reads (self.completed_tasks in
            particular)."""
            if self.current_run is None:
                return
            run = self.current_run
            total_dispatched = sum(c["subtasks"] for c in run["decompose_calls"])
            total_completed = run["subtasks_completed_before_replan"] + len(
                self.completed_tasks
            )
            record = {
                "instruction": run["original_instruction"],
                "model": run["model"],
                "start_wall": run["start_wall"],
                "status": status,
                "total_latency_s": round(
                    time.monotonic() - run["start_monotonic"], 3
                ),
                "llm_calls": run["llm_calls"],
                "replans_used": run["replans_used"],
                "subtasks_dispatched": total_dispatched,
                "subtasks_completed": total_completed,
                "plan_ids": [c["plan_id"] for c in run["decompose_calls"]],
                "decompose_calls": run["decompose_calls"],
                "plans": run["plans"],
            }
            if "decompose_errors" in run:
                record["decompose_errors"] = run["decompose_errors"]
            try:
                with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                self.get_logger().info(f"Eval run recorded -> {RUN_LOG_PATH}")
            except OSError as e:
                self.get_logger().warn(f"Could not write eval run log: {e}")
            self.current_run = None

        def parse_feedback(self, feedback: str):
            plan_id = None
            task_id = None
            status = None
            # Plain string feedback (no JSON, no "stage") is treated as an
            # immediate terminal signal for backwards compatibility, e.g.
            # the manual `ros2 topic pub ... "data: 'T1'"` test in this
            # function's docstring.
            stage = "complete"

            try:
                parsed = json.loads(feedback)
            except json.JSONDecodeError:
                parsed = feedback

            if isinstance(parsed, dict):
                raw_plan_id = parsed.get("plan_id")
                raw_task_id = (
                    parsed.get("task_id")
                    or parsed.get("subtask_id")
                    or parsed.get("id")
                )
                if raw_plan_id is not None:
                    plan_id = str(raw_plan_id)
                if raw_task_id is not None:
                    task_id = str(raw_task_id)
                raw_status = parsed.get("status")
                if raw_status is not None:
                    status = str(raw_status).strip().lower()
                raw_stage = parsed.get("stage")
                if raw_stage is not None:
                    stage = str(raw_stage).strip().lower()
            elif isinstance(parsed, str):
                candidate = parsed.strip()
                if candidate in self.task_map:
                    task_id = candidate

            return plan_id, task_id, status, stage

        def clear_active_plan(self):
            self.active_plan = None
            self.active_graph = None
            self.task_map = {}
            self.execution_waves = []
            self.current_wave_index = 0
            self.pending_feedback = set()
            self.completed_tasks = set()

    rclpy.init()
    node = Layer1Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# MAIN — supports three modes

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: NLI to DAG decomposition"
    )
    parser.add_argument(
        "instruction", nargs="?", default=None,
        help="Natural language instruction (optional). "
             "If omitted, runs in interactive mode."
    )
    parser.add_argument(
        "--ros", action="store_true",
        help="Run as a ROS2 node (subscribes/publishes on ROS2 topics)."
    )
    args = parser.parse_args()

    # ── ROS2 node mode ────────────────────────────
    if args.ros:
        run_ros_node()
        return

    # ── Standalone modes ──────────────────────────
    client = get_client()

    if args.instruction:
        # Single instruction passed as CLI argument
        instructions = [args.instruction]
    else:
        # Interactive mode — keep asking until user types 'exit'
        print("\nLayer 1 — Interactive Mode")
        print("Type your instruction and press Enter. Type 'exit' to quit.\n")
        instructions = []
        while True:
            try:
                user_input = input("Instruction: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("exit", "quit", ""):
                break
            instructions.append(user_input)
            # Process immediately in interactive mode
            print(f"\n{'━'*60}")
            try:
                plan, G = decompose_instruction(user_input, client)
                print_dag(plan, G)
                out_file = f"{plan['plan_id']}.json"
                with open(out_file, "w") as f:
                    json.dump(plan, f, indent=2)
                print(f"\n  [Saved] {out_file}\n")
            except RuntimeError as e:
                print(f"  [FAILED] {e}\n")
        return

    # Process CLI-provided instructions
    for instruction in instructions:
        print(f"\n{'━'*60}")
        print(f"INSTRUCTION: {instruction}")
        print('━'*60)
        try:
            plan, G = decompose_instruction(instruction, client)
            print_dag(plan, G)
            out_file = f"{plan['plan_id']}.json"
            with open(out_file, "w") as f:
                json.dump(plan, f, indent=2)
            print(f"\n  [Saved] {out_file}")
        except RuntimeError as e:
            print(f"\n  [FAILED] {e}")


if __name__ == "__main__":
    main()
