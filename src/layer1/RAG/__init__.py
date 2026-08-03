"""Retrieval-augmented prompting (RAG) for Layer 1.

Two independent retrieval paths:
  few_shot_bank / few_shot_retriever -- which worked examples go in the prompt
                                          (TF-IDF + cosine similarity)
  robot_bank / robot_retriever        -- which robot types' skills go in the prompt
                                          (sentence embeddings + cosine similarity)
"""

from RAG.few_shot_bank import COMBINED_DOMAIN, load_examples
from RAG.few_shot_bank import bank_summary as few_shot_bank_summary
from RAG.few_shot_retriever import CosineFewShotRetriever, format_examples_for_prompt
from RAG.robot_bank import load_robot_bank
from RAG.robot_bank import bank_summary as robot_bank_summary
from RAG.robot_retriever import CosineRobotTypeRetriever, union_skills

__all__ = [
    "COMBINED_DOMAIN",
    "CosineFewShotRetriever",
    "CosineRobotTypeRetriever",
    "few_shot_bank_summary",
    "format_examples_for_prompt",
    "load_examples",
    "load_robot_bank",
    "robot_bank_summary",
    "union_skills",
]
