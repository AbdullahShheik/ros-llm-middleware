"""
dart_metrics_lib.py
--------------------
Implements the same metric DEFINITIONS DART-LLM's paper uses (SR, IPA, DSR,
SGSR, RTR), but scored against your own ground-truth-annotated instruction
bank instead of their dataset. This lets you report a comparable row in the
same table shape, on your own domain, without needing their skill registry.

Metric definitions (matching the paper):
  SR   (Success Rate)                 - whole instruction decomposed correctly
  IPA  (Instruction Parsing Accuracy) - fraction of subtasks with correct
                                         skill/function + parameters
  DSR  (Dependency Satisfaction Rate) - fraction of dependency edges correctly
                                         reproduced / correctly ordered
  SGSR (Semantic Grounding Success Rate) - fraction of outputs that are
                                         structurally well-formed and resolve
                                         (valid JSON, valid skills, valid refs)
  RTR  (Response Time Reliability)    - consistency of response latency
                                         across repeated runs

NOTE on SGSR: DART-LLM define this via their own Breakdown Function Parser,
which you don't have. Here it's implemented as "passed your own structural
validator" (valid JSON + validate_plan clean). This is a documented
substitution, not a literal reproduction of their check -- say so in your
write-up.
"""

import statistics
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    nx = None


# ----------------------------------------------------------------------
# Ground truth / prediction shape (both normalized into this form
# internally via _extract_skills(), which accepts either a "skill" string
# (ground truth bank's format) or a "skills" list (predicted schema's
# required_skills)):
#
# {
#   "subtasks": [
#       {"id": "T1", "skill": "pick", "params": {"object_name": "red_block"},
#        "depends_on": []},
#       {"id": "T2", "skill": "place", "params": {"object_name": "red_block",
#        "target_location": "shelf"}, "depends_on": ["T1"]},
#   ]
# }
#
# Ground truth (l1_ground_truth_bank.py) is already in this shape.
# Predicted plans go through normalize_predicted_plan() below, which maps
# your actual layer1_pipeline.py schema (required_skills / args /
# dependencies) into this normalized form.
# ----------------------------------------------------------------------


def normalize_predicted_plan(raw_plan):
    """
    Adapted to your actual layer1_pipeline.py output schema:
      {
        "plan_id": "...",
        "original_instruction": "...",
        "subtasks": [
            {"id": "T1", "description": "...", "required_skills": ["pick"],
             "args": {"object_name": "red_block", "location": "table"},
             "dependencies": [], "parallelizable": true, "priority": 0},
            ...
        ]
      }

    Note "required_skills" is a LIST (a subtask could in principle name more
    than one skill), unlike the ground truth bank's single "skill" string
    per subtask. This is preserved as a list here -- see _extract_skills()
    in the matching logic, which checks whether the ground-truth skill is
    CONTAINED IN the predicted skills list, rather than requiring exact
    string equality. That tolerates a predicted subtask listing an extra
    skill while still crediting it as correct on the core one; it does NOT
    tolerate omitting the correct skill.

    If status is not "ok" (rejection), or subtasks is missing, returns an
    empty normalized plan -- IPA/DSR/SR will correctly come out as 0 against
    any non-empty ground truth.
    """
    if not isinstance(raw_plan, dict):
        return {"subtasks": []}

    raw_subtasks = raw_plan.get("subtasks", [])
    normalized = []
    for t in raw_subtasks:
        normalized.append({
            "id": t.get("id"),
            "skills": t.get("required_skills", []),
            "params": t.get("args", {}) or t.get("params", {}),
            "depends_on": t.get("dependencies", []) or t.get("depends_on", []),
        })
    return {"subtasks": normalized}


def _extract_skills(subtask):
    """
    Returns a subtask's skill(s) as a list, regardless of whether it's
    stored as "skills" (list, predicted schema) or "skill" (single string,
    ground truth bank's schema).
    """
    if subtask.get("skills"):
        return list(subtask["skills"])
    if subtask.get("skill"):
        return [subtask["skill"]]
    return []


def _build_graph(normalized_plan):
    """Builds a networkx DiGraph from a normalized plan (skills list on each node)."""
    G = nx.DiGraph()
    for t in normalized_plan["subtasks"]:
        G.add_node(t["id"], skills=_extract_skills(t), params=t.get("params", {}))
    for t in normalized_plan["subtasks"]:
        for dep in t["depends_on"]:
            if dep in G:
                G.add_edge(dep, t["id"])
    return G


def _topological_waves(G):
    """Groups nodes into parallel-safe waves (topological generations)."""
    try:
        return [list(wave) for wave in nx.topological_generations(G)]
    except Exception:
        return [[n] for n in G.nodes]


def _match_nodes(pred_G, gt_G):
    """
    Globally match predicted nodes to ground-truth nodes by skill compatibility
    and parameter overlap.

    This is deliberately NOT restricted by topological wave/position -- IPA
    should credit "was the right skill identified" independent of ordering; DSR
    separately penalizes wrong ordering/dependencies. Tying matching to wave
    position conflates the two (a reversed-dependency plan with the exact right
    skills would wrongly score IPA=0 instead of IPA=1/DSR=0).

    The matching is solved as a weighted bipartite matching problem with
    NetworkX. Candidate edges require every ground-truth skill to be present in
    the predicted subtask's skills. Edge weight primarily maximizes parameter
    overlap; a small deterministic tie-breaker favors matching identical ids
    and then similarly ordered nodes without changing the overlap objective.

    Returns: dict pred_node_id -> matched gt_node_id (only for matched pairs)
    """
    pred_nodes_sorted = sorted(pred_G.nodes)
    gt_nodes_sorted = sorted(gt_G.nodes)
    if not pred_nodes_sorted or not gt_nodes_sorted:
        return {}

    candidate_edges = []
    max_tie_bonus = 0

    for p_index, p_node in enumerate(pred_nodes_sorted):
        p_skills = pred_G.nodes[p_node]["skills"]
        p_params = pred_G.nodes[p_node].get("params", {}) or {}

        for g_index, g_node in enumerate(gt_nodes_sorted):
            g_skills = gt_G.nodes[g_node]["skills"]  # ground truth: usually one skill
            if not all(gs in p_skills for gs in g_skills):
                continue

            g_params = gt_G.nodes[g_node].get("params", {}) or {}
            overlap = sum(
                1 for k, v in g_params.items()
                if k in p_params and str(p_params[k]).lower() == str(v).lower()
            )

            id_bonus = len(pred_nodes_sorted) + len(gt_nodes_sorted) if p_node == g_node else 0
            order_bonus = (
                len(pred_nodes_sorted)
                + len(gt_nodes_sorted)
                - abs(p_index - g_index)
            )
            tie_bonus = id_bonus + order_bonus
            max_tie_bonus = max(max_tie_bonus, tie_bonus)
            candidate_edges.append((p_node, g_node, overlap, tie_bonus))

    if not candidate_edges:
        return {}

    # One parameter match must outweigh every possible deterministic tie-break
    # accumulated across the whole matching.
    max_matches = min(len(pred_nodes_sorted), len(gt_nodes_sorted))
    param_unit = (max_matches * max_tie_bonus) + 1

    matching_graph = nx.Graph()
    for p_node in pred_nodes_sorted:
        matching_graph.add_node(("pred", p_node), bipartite=0)
    for g_node in gt_nodes_sorted:
        matching_graph.add_node(("gt", g_node), bipartite=1)

    for p_node, g_node, overlap, tie_bonus in candidate_edges:
        matching_graph.add_edge(
            ("pred", p_node),
            ("gt", g_node),
            weight=(overlap * param_unit) + tie_bonus,
        )

    optimal_pairs = nx.algorithms.matching.max_weight_matching(
        matching_graph,
        maxcardinality=True,
        weight="weight",
    )

    matched = {}
    for left, right in optimal_pairs:
        if left[0] == "pred":
            pred_node, gt_node = left[1], right[1]
        else:
            pred_node, gt_node = right[1], left[1]
        matched[pred_node] = gt_node
    return matched


def compute_ipa(pred_normalized, gt_normalized):
    """
    IPA = (# ground-truth subtasks correctly matched to a predicted subtask
           with the same skill) / (# ground-truth subtasks)
    """
    if not gt_normalized["subtasks"]:
        return 1.0 if not pred_normalized["subtasks"] else 0.0

    pred_G = _build_graph(pred_normalized)
    gt_G = _build_graph(gt_normalized)
    matched = _match_nodes(pred_G, gt_G)

    return len(matched) / len(gt_normalized["subtasks"])


def compute_dsr(pred_normalized, gt_normalized):
    """
    DSR = (# ground-truth dependency edges correctly reproduced by the
           matched predicted nodes) / (# ground-truth dependency edges)

    An edge (u -> v) in ground truth counts as "correctly reproduced" if
    both u and v were matched to predicted nodes, and there's an edge (or
    a directed path -- i.e. the ordering constraint is respected even if
    not a literal 1-hop edge) from matched(u) to matched(v) in the
    predicted graph.

    If ground truth has zero edges (single-subtask instructions), DSR is
    defined as 1.0 when the (sole) subtask was matched, else 0.0.
    """
    pred_G = _build_graph(pred_normalized)
    gt_G = _build_graph(gt_normalized)
    matched = _match_nodes(pred_G, gt_G)  # pred_id -> gt_id
    gt_to_pred = {v: k for k, v in matched.items()}  # gt_id -> pred_id

    gt_edges = list(gt_G.edges())
    if not gt_edges:
        return 1.0 if len(matched) == len(gt_normalized["subtasks"]) else 0.0

    try:
        pred_reachability = nx.transitive_closure_dag(pred_G)
    except Exception:
        pred_reachability = nx.transitive_closure(pred_G)

    correct = 0
    for u, v in gt_edges:
        pu, pv = gt_to_pred.get(u), gt_to_pred.get(v)
        if pu is None or pv is None:
            continue
        if pred_reachability.has_edge(pu, pv):
            correct += 1

    return correct / len(gt_edges)


def compute_sr(pred_normalized, gt_normalized, ipa=None, dsr=None):
    """
    SR = 1.0 only if EVERY ground-truth subtask was correctly matched
         (IPA == 1.0), EVERY dependency was correctly reproduced (DSR == 1.0),
         AND the predicted plan has no extra/hallucinated subtasks beyond
         ground truth. Otherwise 0.0.

    This is intentionally strict (matches DART-LLM's "whole instruction
    decomposed correctly" framing) -- SR is not an average, it's pass/fail.
    """
    if ipa is None:
        ipa = compute_ipa(pred_normalized, gt_normalized)
    if dsr is None:
        dsr = compute_dsr(pred_normalized, gt_normalized)

    no_extra = len(pred_normalized["subtasks"]) == len(gt_normalized["subtasks"])
    return 1.0 if (ipa == 1.0 and dsr == 1.0 and no_extra) else 0.0


def compute_sgsr(structurally_valid: bool) -> float:
    """
    SGSR substitute: whether the output was structurally well-formed --
    valid JSON, valid skill names, resolvable dependencies, no cycles --
    per YOUR OWN validate_plan(), not DART-LLM's Breakdown Function Parser.
    Pass this in per-run as a bool; it gets averaged at the aggregate level.
    """
    return 1.0 if structurally_valid else 0.0


def compute_rtr(response_times):
    """
    RTR = 1 - (stdev / mean), clipped to [0, 1].

    This is the standard coefficient-of-variation-based reliability score:
    tight, consistent response times -> RTR near 1; wildly varying response
    times -> RTR near 0. Needs at least 2 response times to compute a
    meaningful stdev; with fewer, returns 1.0 (undefined variance).

    NOTE: verify this formula against the exact wording in the DART-LLM
    paper if your professor wants a literal reproduction -- this is the
    standard interpretation of "response time reliability" but the paper's
    exact equation wasn't independently confirmed here.
    """
    if len(response_times) < 2:
        return 1.0
    mean = statistics.mean(response_times)
    if mean == 0:
        return 0.0
    stdev = statistics.stdev(response_times)
    rtr = 1 - (stdev / mean)
    return max(0.0, min(1.0, rtr))


def aggregate_metrics(per_instruction_results):
    """
    per_instruction_results: list of dicts, each with keys:
        sr, ipa, dsr, sgsr, response_times (list of floats, one per run)

    Returns a single summary dict with the 5 metrics averaged the way
    DART-LLM reports them (mean across all instructions in the tier).
    """
    if not per_instruction_results:
        return {"SR": 0, "IPA": 0, "DSR": 0, "SGSR": 0, "RTR": 0}

    sr_vals = [r["sr"] for r in per_instruction_results]
    ipa_vals = [r["ipa"] for r in per_instruction_results]
    dsr_vals = [r["dsr"] for r in per_instruction_results]
    sgsr_vals = [r["sgsr"] for r in per_instruction_results]

    all_response_times = []
    for r in per_instruction_results:
        all_response_times.extend(r["response_times"])

    return {
        "SR": round(statistics.mean(sr_vals), 3),
        "IPA": round(statistics.mean(ipa_vals), 3),
        "DSR": round(statistics.mean(dsr_vals), 3),
        "SGSR": round(statistics.mean(sgsr_vals), 3),
        "RTR": round(compute_rtr(all_response_times), 3),
    }
