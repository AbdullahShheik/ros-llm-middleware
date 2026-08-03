"""
few_shot_retriever.py
-----------------------
Retrieves the top-k most relevant few-shot examples for a given instruction,
instead of injecting the whole example bank into every prompt.

Retrieval method: TF-IDF vectors + COSINE SIMILARITY over the examples'
instruction text, fit once at start-up. This is the only backend -- the
neural-embedding alternative that used to sit alongside it has been removed
so there is exactly one retrieval path to reason about and evaluate.

On top of pure top-k ranking, retrieve() applies a deterministic coverage
safety net before returning:

  1. Rank the entire bank by cosine similarity to the query (this is still
     the primary relevance signal -- the guarantees only adjust the tail).
  2. Check whether the natural top-k already contains at least one REJECTION
     example (status != "ok") and at least one COMBINED-domain example
     (skill_domains == {"arm", "mobile"}).
  3. If either is missing, swap in the best-scoring example that satisfies
     it, displacing the weakest current pick -- so the least relevance
     possible is given up.

This does not try to guess the query's true status or domain (that is what
the LLM is being asked to decide, so filtering on a guess would be circular).
It only insures the retrieved SET against having zero coverage of a category
that matters: without a rejection example in the prompt, the model tends to
force a plan for instructions it should be refusing.
"""

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from RAG.few_shot_bank import COMBINED_DOMAIN


class CosineFewShotRetriever:
    def __init__(self, examples: list[dict]):
        """examples: list of dicts from few_shot_bank.load_examples()."""
        if not examples:
            raise ValueError("Few-shot bank is empty -- nothing to retrieve from.")

        self.examples = examples
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),   # unigrams + bigrams: "pick up", "red block"
            stop_words="english",
        )
        self.example_matrix = self.vectorizer.fit_transform(
            [ex["instruction"] for ex in examples]
        )

    def rank_all(self, query_instruction: str) -> list[tuple[dict, float]]:
        """Full bank ranked by cosine similarity, highest first."""
        query_vec = self.vectorizer.transform([query_instruction])
        scores = cosine_similarity(query_vec, self.example_matrix)[0]
        ranked = [(self.examples[i], float(scores[i]))
                  for i in np.argsort(scores)[::-1]]
        return ranked

    def retrieve(self, query_instruction: str, k: int = 4,
                 min_score: float = 0.0,
                 guarantee_rejection: bool = True,
                 guarantee_combined_domain: bool = True) -> list[tuple[dict, float]]:
        """
        Returns up to k (example, score) tuples, highest score first.

        min_score: examples scoring below this are dropped from the natural
                   top-k, so a query that matches nothing gets few relevant
                   examples rather than being padded with noise. The coverage
                   guarantees deliberately IGNORE min_score -- a safety net
                   that switches itself off exactly when the query is unlike
                   anything in the bank would be no safety net at all.
        """
        full_ranking = self.rank_all(query_instruction)

        selected = [(ex, s) for ex, s in full_ranking[:k] if s >= min_score]
        selected_ids = {ex["id"] for ex, _ in selected}
        # Ids inserted by a guarantee. Held back from displacement so the
        # second guarantee cannot evict what the first one just inserted --
        # a guaranteed example is often the weakest-scoring slot, which is
        # exactly the slot the swap targets.
        guaranteed_ids = set()

        def swap_in_best(predicate) -> bool:
            """Replace the weakest displaceable pick with the best-scoring
            example satisfying predicate, if the bank has one not already
            selected."""
            for ex, score in full_ranking:
                if ex["id"] in selected_ids or not predicate(ex):
                    continue
                if len(selected) >= k:
                    displaceable = sorted(
                        (pair for pair in selected if pair[0]["id"] not in guaranteed_ids),
                        key=lambda pair: pair[1],
                    )
                    if not displaceable:
                        return False  # every slot is already a guarantee
                    removed = displaceable[0]
                    selected.remove(removed)
                    selected_ids.discard(removed[0]["id"])
                selected.append((ex, score))
                selected_ids.add(ex["id"])
                guaranteed_ids.add(ex["id"])
                return True
            return False  # bank contains no example satisfying predicate

        is_rejection = lambda ex: ex["status"] != "ok"
        is_combined = lambda ex: ex["skill_domains"] == COMBINED_DOMAIN

        if guarantee_rejection and not any(is_rejection(ex) for ex, _ in selected):
            swap_in_best(is_rejection)

        if guarantee_combined_domain and not any(is_combined(ex) for ex, _ in selected):
            swap_in_best(is_combined)

        selected.sort(key=lambda pair: -pair[1])  # highest similarity first
        return selected


def format_examples_for_prompt(retrieved: list[tuple[dict, float]],
                               indent: int = 2) -> str:
    """
    Renders retrieved examples into a prompt-ready text block.

    The bank stores each example's output as {"status", "subtasks"} only, but
    the prompt's OUTPUT SCHEMA also requires plan_id and original_instruction
    on "ok" plans. Those two fields are synthesised here from the example's own
    id and instruction, so every rendered example is a valid instance of the
    schema the model is being asked to follow -- examples that contradict the
    schema are worse than no examples.
    """
    blocks = []
    for i, (ex, score) in enumerate(retrieved, 1):
        output = dict(ex["output"])
        if output.get("status") == "ok":
            output = {
                "status": "ok",
                "plan_id": ex["id"],
                "original_instruction": ex["instruction"],
                "subtasks": output.get("subtasks", []),
            }
        blocks.append(
            f"========== EXAMPLE {i} - {output.get('status', 'ok')} ==========\n"
            f"Instruction: \"{ex['instruction']}\"\n"
            f"Output:\n{json.dumps(output, indent=indent)}\n"
        )
    return "\n".join(blocks)
