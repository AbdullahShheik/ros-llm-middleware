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
  latest
"""

import json
import os
import sys
import argparse
import itertools
import networkx as nx
from groq import Groq

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
SKILLS_FILE  = os.path.join(os.path.dirname(__file__), "robot_skills.json")

# get_client() returns the LLM client. Currently only Groq is supported.


def get_client():
    """Return a Groq client for LLM calls."""
    return Groq(api_key=GROQ_API_KEY)


_plan_counter = itertools.count(1)

def _next_plan_id() -> str:
    return f"plan_{next(_plan_counter):03d}"



# STEP 1 — Load robot skill registry
def load_skills(filepath: str) -> dict:
    """Load robot skill definitions from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def build_skill_prompt_block(robot_config: dict) -> str:
    """
    Convert the skill list into a readable string injected into the LLM prompt.
    NOTE: We do NOT include robot_id or robot_type here.
    Layer 1 only decomposes tasks — robot assignment is Layer 2's job.
    """
    lines = ["Available Skills for the Robotic Arm:"]
    for skill in robot_config["skills"]:
        inputs_str = ", ".join(skill["inputs"]) if skill["inputs"] else "none"
        pre_str    = "; ".join(skill["preconditions"]) if skill["preconditions"] else "none"
        eff_str    = "; ".join(skill["effects"])       if skill["effects"]       else "none"
        lines.append(
            f"\n  name         : {skill['name']}"
            f"\n  description  : {skill['description']}"
            f"\n  inputs       : [{inputs_str}]"
            f"\n  preconditions: {pre_str}"
            f"\n  effects      : {eff_str}"
        )
    return "\n".join(lines)


# STEP 2 — Few-shot examples
# F component in P = (I, E, R, S, F)

FEW_SHOT_EXAMPLES = """
========== EXAMPLE 1 — Simple sequential (L1) ==========
Instruction: "Pick up the red block and place it on the shelf."
Output:
{
  "status": "ok",
  "plan_id": "plan_001",
  "original_instruction": "Pick up the red block and place it on the shelf.",
  "subtasks": [
    {
      "id": "T1",
      "description": "Pick up the red block",
      "required_skills": ["pick"],
      "args": { "object_name": "red_block", "location": "red_block" },
      "dependencies": [],
      "parallelizable": false,
      "priority": 1
    },
    {
      "id": "T2",
      "description": "Place the red block on the shelf",
      "required_skills": ["place"],
      "args": { "object_name": "red_block", "target_location": "shelf" },
      "dependencies": ["T1"],
      "parallelizable": false,
      "priority": 2
    },
    {
      "id": "T3",
      "description": "Return arm to home position",
      "required_skills": ["home"],
      "args": {},
      "dependencies": ["T2"],
      "parallelizable": false,
      "priority": 3
    }
  ]
}

========== EXAMPLE 1B - Clarification needed ==========
Instruction: "Put the thing over there."
Output:
{
  "status": "clarification_needed",
  "reason": "Ambiguous object and target location: 'the thing' and 'over there'."
}

========== EXAMPLE 1C - Infeasible ==========
Instruction: "Write a Python script to calculate the Fibonacci sequence."
Output:
{
  "status": "infeasible",
  "reason": "The instruction is outside the robotic arm manipulation domain."
}

========== EXAMPLE 1D - Explicit coordinate target ==========
Instruction: "Pick up the green block and place it at 0.3, -0.4."
Output:
{
  "status": "ok",
  "plan_id": "plan_00X",
  "original_instruction": "Pick up the green block and place it at 0.3, -0.4.",
  "subtasks": [
    {
      "id": "T1",
      "description": "Pick up the green block",
      "required_skills": ["pick"],
      "args": { "object_name": "green_block", "location": "green_block" },
      "dependencies": [],
      "parallelizable": false,
      "priority": 1
    },
    {
      "id": "T2",
      "description": "Place the green block at 0.3, -0.4",
      "required_skills": ["place"],
      "args": { "object_name": "green_block", "target_location": "0.3, -0.4" },
      "dependencies": ["T1"],
      "parallelizable": false,
      "priority": 2
    },
    {
      "id": "T3",
      "description": "Return arm to home position",
      "required_skills": ["home"],
      "args": {},
      "dependencies": ["T2"],
      "parallelizable": false,
      "priority": 3
    }
  ]
}
Note how target_location above is "0.3, -0.4" -- copied verbatim from the
instruction, not reworded to something like "target_position" or "the given
coordinates".

========== EXAMPLE 2 - Inspect then act (L2) ==========
Instruction: "Inspect the blue cylinder and then push it 5cm to the right."
Output:
{
  "status": "ok",
  "plan_id": "plan_002",
  "original_instruction": "Inspect the blue cylinder and then push it 5cm to the right.",
  "subtasks": [
    {
      "id": "T1",
      "description": "Inspect the blue cylinder",
      "required_skills": ["inspect"],
      "args": { "object_name": "blue_cylinder" },
      "dependencies": [],
      "parallelizable": false,
      "priority": 1
    },
    {
      "id": "T2",
      "description": "Push the blue cylinder 5cm to the right",
      "required_skills": ["push"],
      "args": { "object_name": "blue_cylinder", "direction": "right", "distance_cm": 5 },
      "dependencies": ["T1"],
      "parallelizable": false,
      "priority": 2
    },
    {
      "id": "T3",
      "description": "Return arm to home position",
      "required_skills": ["home"],
      "args": {},
      "dependencies": ["T2"],
      "parallelizable": false,
      "priority": 3
    }
  ]
}

========== EXAMPLE 3 — Multi-object sequence (L3) ==========
Instruction: "Pick up the green cube and the yellow cone and place both on the tray."
Output:
{
  "status": "ok",
  "plan_id": "plan_003",
  "original_instruction": "Pick up the green cube and the yellow cone and place both on the tray.",
  "subtasks": [
    {
      "id": "T1",
      "description": "Pick up the green cube",
      "required_skills": ["pick"],
      "args": { "object_name": "green_cube", "location": "green_cube" },
      "dependencies": [],
      "parallelizable": false,
      "priority": 1
    },
    {
      "id": "T2",
      "description": "Place the green cube on the tray",
      "required_skills": ["place"],
      "args": { "object_name": "green_cube", "target_location": "tray" },
      "dependencies": ["T1"],
      "parallelizable": false,
      "priority": 2
    },
    {
      "id": "T3",
      "description": "Pick up the yellow cone",
      "required_skills": ["pick"],
      "args": { "object_name": "yellow_cone", "location": "yellow_cone" },
      "dependencies": ["T2"],
      "parallelizable": false,
      "priority": 3
    },
    {
      "id": "T4",
      "description": "Place the yellow cone on the tray",
      "required_skills": ["place"],
      "args": { "object_name": "yellow_cone", "target_location": "tray" },
      "dependencies": ["T3"],
      "parallelizable": false,
      "priority": 4
    },
    {
      "id": "T5",
      "description": "Return arm to home position",
      "required_skills": ["home"],
      "args": {},
      "dependencies": ["T4"],
      "parallelizable": false,
      "priority": 5
    }
  ]
}
"""

# STEP 3 — Prompt builder
# Follows DART-LLM's P = (I, E, R, S, F)


SYSTEM_PROMPT = """You are a task decomposition planner for a robotic arm system.
Your job is to take a high-level natural language instruction and break it down
into atomic subtasks the arm can execute.

Rules you must follow:
1. BEFORE decomposing, classify the instruction into exactly one status:
   "ok", "clarification_needed", or "infeasible".
2. Use "ok" only when the instruction is clear, physically executable, and can be
   decomposed using only the provided robotic arm skills.
3. Use "clarification_needed" when an unresolved referent such as "it", "the thing",
   or "over there" cannot be grounded in the environment or skill registry. Return
   only status and reason; do not produce a plan.
4. Use "infeasible" when the instruction is unambiguous but cannot be executed:
   outside the robot domain, missing required skills, physically impossible,
   temporally impossible, or self-contradictory. Return only status and reason;
   do not produce a plan.
5. For status "ok", required_skills must only contain skill names from the provided
   skill list. No other values allowed.
6. Do NOT assign tasks to robots. That is handled by a separate layer.
7. Every subtask must have a unique id in the format T1, T2, T3 ...
8. dependencies must only reference ids of other subtasks in the same plan.
9. If a subtask has no prerequisites, set dependencies to an empty list [].
10. Think step by step about the physical actions needed and their order before writing JSON.
11. Output ONLY the raw JSON. No explanation, no markdown, no code fences.
12. If the instruction gives an explicit target position for a "place" skill (e.g.
   coordinates like "0.3, 0.4" or "(0.3, -0.2)"), copy that value into
   target_location verbatim, exactly as written. Do NOT paraphrase it into a
   generic placeholder like "target_position" -- the executor parses this string
   directly and a placeholder cannot be resolved to a real pose."""


def build_prompt(instruction: str,
                 skill_block: str,
                 plan_id: str,
                 environment: str = None) -> str:
    """
    Assembles the full prompt following DART-LLM's P = (I, E, R, S, F).
    Environment is a placeholder for now — will be replaced by
    a live object map topic subscription later.
    """
    env_section = environment if environment else (
        "Environment: Not yet specified. "
        "Use generic location names such as object_position, target_location, home_position."
    )

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
{FEW_SHOT_EXAMPLES}

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


# STEP 6 — Validator
# Three checks following DART-LLM's validation approach

def validate_plan(plan: dict, G: nx.DiGraph, robot_config: dict) -> list[str]:
    """
    Validates the plan. Returns list of error strings.
    Empty list = valid plan.

    Check 1: No cycles in DAG
    Check 2: No dangling dependency references
    Check 3: All required_skills exist in skill registry
    """
    errors = []
    valid_skills = {s["name"] for s in robot_config["skills"]}
    task_ids     = {t["id"]   for t in plan["subtasks"]}

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

    return errors


# STEP 7 — Full pipeline with retry loop
# Mirrors DART-LLM Algorithm 2 re-prompting on failure

def decompose_instruction(
    instruction: str,
    client: Groq,
    robot_config: dict,
    environment: str = None
) -> tuple[dict, nx.DiGraph]:
    """
    Full Layer 1 pipeline:
    NLI → prompt → LLM → parse JSON → build DAG → validate → retry if needed.
    Returns (plan_dict, dag) on success.
    Raises RuntimeError if all retries are exhausted.
    """
    skill_block  = build_skill_prompt_block(robot_config)
    plan_id      = _next_plan_id()
    user_prompt  = build_prompt(instruction, skill_block, plan_id, environment)
    error_suffix = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n[Layer 1] Attempt {attempt}/{MAX_RETRIES} — calling LLM...")

        raw = call_llm(client, user_prompt + error_suffix)

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


def run_ros_node(robot_config: dict):
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

    class Layer1Node(Node):
        def __init__(self):
            super().__init__("layer1_node")
            self.robot_config = robot_config
            self.client       = get_client()
            self.active_plan = None
            self.active_graph = None
            self.task_map = {}
            self.execution_waves = []
            self.current_wave_index = 0
            self.pending_feedback = set()
            self.completed_tasks = set()

            self.publisher_ = self.create_publisher(
                String, "/layer1/taskplan", 10
            )
            self.subscription = self.create_subscription(
                String, "/layer1/instruction", self.instruction_callback, 10
            )
            self.feedback_subscription = self.create_subscription(
                String, "/execution_feedback", self.feedback_callback, 10
            )
            self.get_logger().info(
                "Layer 1 node ready. "
                "Listening on /layer1/instruction, "
                "listening on /execution_feedback, "
                "publishing to /layer1/taskplan."
            )

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

            self.get_logger().info(f"Received instruction: {instruction}")
            try:
                plan, G = decompose_instruction(
                    instruction, self.client, self.robot_config
                )
                print_dag(plan, G)
                self.start_plan_dispatch(plan, G)
            except Exception as e:
                # Catch everything, not just RuntimeError: decompose_instruction
                # only retries on malformed LLM JSON -- a network blip, rate
                # limit, or any other Groq API error propagates straight out
                # of the client call uncaught. Left as RuntimeError-only, an
                # unhandled exception here escapes this callback and takes
                # down rclpy.spin(), killing the whole node -- meaning a
                # single transient API failure on the *second* instruction
                # (or any instruction after the first) permanently ends the
                # session, since active_plan was never set and nothing
                # restarts the node. Log and stay alive for the next attempt.
                self.get_logger().error(f"Decomposition failed: {e}")

        def start_plan_dispatch(self, plan: dict, G: nx.DiGraph):
            self.active_plan = plan
            self.active_graph = G
            self.task_map = {task["id"]: task for task in plan["subtasks"]}
            self.execution_waves = [
                sorted(wave) for wave in nx.topological_generations(G)
            ]
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
                self.get_logger().error(
                    f"Feedback reported {status} for task {task_id}; "
                    f"aborting plan {self.active_plan['plan_id']}."
                )
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

    # Load skills
    robot_config = load_skills(SKILLS_FILE)
    print(f"[Layer 1] Loaded {len(robot_config['skills'])} skills.")

    # ── ROS2 node mode ────────────────────────────
    if args.ros:
        run_ros_node(robot_config)
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
                plan, G = decompose_instruction(user_input, client, robot_config)
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
            plan, G = decompose_instruction(instruction, client, robot_config)
            print_dag(plan, G)
            out_file = f"{plan['plan_id']}.json"
            with open(out_file, "w") as f:
                json.dump(plan, f, indent=2)
            print(f"\n  [Saved] {out_file}")
        except RuntimeError as e:
            print(f"\n  [FAILED] {e}")


if __name__ == "__main__":
    main()
