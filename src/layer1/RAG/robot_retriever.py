"""
robot_retriever.py
-------------------
Retrieves which currently-active robot TYPES are relevant to an instruction,
by cosine similarity between the instruction and each type's skill/capability
text (see robot_bank.py). Vectorization is a local sentence-embedding model
(sentence-transformers), NOT TF-IDF: the robot-type corpus is tiny (2-3
documents right now) and grows by whole robot types, not by volume, so there
is no benefit to term-frequency statistics here -- and dense embeddings catch
paraphrases ("bring the bottle over" ~ "navigate to a location") that share
few exact words with a skill's own description, which matters more for this
corpus than for the much larger, more example-dense few-shot bank.

Selection is at the robot-TYPE granularity, not individual skills: once a
type is selected, ALL of its own skills go into the prompt as a unit -- never
skills belonging to a type that wasn't selected, and never the full master
catalog. Whole-type selection is deliberate: picking individual skills by
similarity risks silently leaving out a skill the instruction actually needs
just because its wording didn't overlap with the instruction text, which
would make an otherwise-valid plan fail validation for "using a skill that
wasn't offered". Selecting whole types trades coarser trimming for that
safety.

A combined instruction (e.g. "go to the shelf and pick up the box") should
surface BOTH the arm and the mobile-base types, since cosine similarity
naturally gives partial credit to each half of the instruction's meaning --
not a hardcoded rule, just how embedding similarity plays out across two
relevant documents instead of one.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class CosineRobotTypeRetriever:
    def __init__(self, bank: list, model_name: str = DEFAULT_MODEL):
        if not bank:
            raise ValueError("Robot bank is empty -- nothing to retrieve from.")
        self.bank = bank
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [entry["text"] for entry in bank],
            convert_to_numpy=True, normalize_embeddings=True,
        )

    def retrieve(self, instruction: str, k: int = None,
                 min_score: float = 0.0) -> list:
        """
        Returns (entry, score) tuples for every active robot type scoring at
        or above min_score, ranked highest first, capped at k entries if k
        is given. Always returns at least one entry (the single best-scoring
        type) even if every score is below min_score or exactly zero -- a
        prompt with zero available skills can never produce a plan, so there
        is no instruction for which returning nothing is the right answer.
        """
        query_emb = self.model.encode(
            [instruction], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = cosine_similarity(query_emb, self.embeddings)[0]
        ranked = sorted(range(len(self.bank)), key=lambda i: -scores[i])

        selected = [(self.bank[i], float(scores[i])) for i in ranked
                    if scores[i] >= min_score]
        if not selected:
            top = ranked[0]
            selected = [(self.bank[top], float(scores[top]))]

        if k is not None:
            selected = selected[:k]
        return selected


def union_skills(selected: list) -> dict:
    """Flattens selected (entry, score) pairs into one deduplicated skill
    list -- the union of only the SELECTED types' own skills -- in the exact
    shape build_skill_prompt_block/validate_plan expect: {"skills": [...]}.
    """
    seen = {}
    for entry, _ in selected:
        for skill in entry["skills"]:
            seen.setdefault(skill["name"], skill)
    return {"skills": list(seen.values())}
