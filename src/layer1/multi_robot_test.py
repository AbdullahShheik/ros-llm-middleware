"""
multi_robot_test.py
---------------------
Demo/test script: with the fleet now configured for 2 robotic arms (see
robot_fleet.json), checks whether Layer 1 correctly recognizes that placing
two DIFFERENT blocks on the shelf are two logically independent tasks --
rather than inventing an artificial dependency between them that would waste
the second arm.

IMPORTANT CONTEXT for presenting this:
Layer 1 does not read the robot COUNT anywhere in its decomposition logic --
by the pipeline's original design, "Layer 1 only decomposes tasks -- robot
assignment is Layer 2's job" (see build_skill_prompt_block's docstring in
layer1_pipeline.py). Whether picking the red block and picking the green
block can run in the same execution "wave" is purely a question of whether
the LLM marks them as logically independent (no shared data/ordering
requirement) -- that answer should be the same whether the fleet has 1 arm
or 100. What CHANGES with 2 arms is whether that independent wave can
actually be executed concurrently in practice (a Layer 2 scheduling
concern, out of scope here). This script checks the one thing Layer 1 IS
responsible for: does the DAG it produces avoid a spurious dependency
between the red-block chain and the green-block chain?

Known risk factor, flagged in advance rather than silently patched around:
the few-shot bank's plan_014 ("pick up the green cube and the yellow cone
and place both on the tray") demonstrates a very similarly-worded
two-object instruction with an UNNECESSARY sequential dependency between
the two objects (picking the second object is made to wait on placing the
first). Since few-shot retrieval is based on textual similarity, plan_014
is likely to be retrieved for this test instruction and could bias the
model toward copying that same unnecessary sequencing.

This test hits the real Groq LLM (uses GROQ_API_KEY from .env) -- it is not
a mocked/offline check.

Usage:
    python multi_robot_test.py
"""

import json
import os

import networkx as nx

from layer1_pipeline import (
    GROQ_API_KEY,
    SKILLS_FILE,
    decompose_instruction,
    get_client,
    load_skills,
    print_dag,
)

INSTRUCTION = "place the red block on the shelf and place the green block on the shelf"
FLEET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_fleet.json")


def print_fleet_context():
    with open(FLEET_FILE, "r", encoding="utf-8") as f:
        fleet = json.load(f)["fleet"]

    print("=" * 70)
    for robot_type, count in fleet.items():
        print(f"  {robot_type}: {count}")
   
    print()


def classify_task(task: dict) -> str:
    """Which block-chain a subtask belongs to, based on its description/args."""
    text = (task.get("description", "") + " " + json.dumps(task.get("args", {}))).lower()
    if "red" in text:
        return "red"
    if "green" in text:
        return "green"
    return "other"


def check_independence(plan: dict, G: nx.DiGraph):
    task_map = {t["id"]: t for t in plan["subtasks"]}
    chains = {tid: classify_task(t) for tid, t in task_map.items()}

    print("=" * 70)
    print("INDEPENDENCE CHECK")
    print("=" * 70)
    for tid, chain in chains.items():
        print(f"  {tid} ({chain:5s}): {task_map[tid]['description']}")
    print()

    cross_edges = [
        (a, b) for a, b in G.edges()
        if chains.get(a) in ("red", "green") and chains.get(b) in ("red", "green")
        and chains[a] != chains[b]
    ]

    waves = [sorted(w) for w in nx.topological_generations(G)]
    red_first_wave = next(
        (i for i, w in enumerate(waves) for tid in w if chains.get(tid) == "red"), None
    )
    green_first_wave = next(
        (i for i, w in enumerate(waves) for tid in w if chains.get(tid) == "green"), None
    )

    if cross_edges:
        print(f"  RESULT: SEQUENTIAL -- found {len(cross_edges)} dependency edge(s) "
              f"crossing between the red-block chain and the green-block chain: {cross_edges}")
        print("  The plan forces one block to fully finish before the other starts,")
        print("  even though nothing in the instruction requires that ordering.")
        print("  With 2 arms available, this artificial dependency wastes the second arm.")
    elif red_first_wave is not None and green_first_wave is not None and red_first_wave == green_first_wave:
        print(f"  RESULT: PARALLEL -- the red and green chains both start in the same "
              f"execution wave (wave {red_first_wave + 1}), with no dependency between them.")
        print("  With 2 arms available, a Layer 2 scheduler could run both chains at once.")
    else:
        print("  RESULT: INDEPENDENT BUT STAGGERED -- no dependency edge crosses between")
        print(f"  the two chains (so they COULD run concurrently), but they start in")
        print(f"  different waves (red: wave {red_first_wave}, green: wave {green_first_wave})")
        print("  without a logical reason requiring that order.")

    return cross_edges, waves


def main():
    print_fleet_context()

    if GROQ_API_KEY in (None, "YOUR_API_KEY"):
        print("[ERROR] GROQ_API_KEY is not configured -- this test calls the real LLM.")
        return

    robot_config = load_skills(SKILLS_FILE)
    client = get_client()

    print(f'INSTRUCTION: "{INSTRUCTION}"\n')

    try:
        plan, G = decompose_instruction(INSTRUCTION, client, robot_config)
    except RuntimeError as e:
        print(f"[FAILED] {e}")
        return

    print_dag(plan, G)
    print()
    check_independence(plan, G)


if __name__ == "__main__":
    main()
