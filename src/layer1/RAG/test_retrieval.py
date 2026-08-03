"""
test_retrieval.py
------------------
Manual smoke test for few-shot retrieval. Prints, for each query, the naive
cosine top-k next to the guaranteed top-k, so the effect of the coverage
safety net is visible.

Run from anywhere:
    python src/layer1/RAG/test_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.few_shot_bank import bank_summary, load_examples
from RAG.few_shot_retriever import CosineFewShotRetriever, format_examples_for_prompt

EXAMPLES = load_examples()
retriever = CosineFewShotRetriever(EXAMPLES)

TEST_QUERIES = [
    "Pick up the blue block and set it on the shelf.",     # looks purely arm-only
    "Grab the toolbox from the shelf and bring it to the bench.",  # needs arm+mobile
    "Take care of the block situation.",                   # vague -> wants a rejection example
    "Drive to the kitchen.",                               # mobile-only
    "Sing me a song about robots.",                        # out of domain, matches nothing
]


def show(label, results):
    print(f"  {label}:")
    for ex, score in results:
        domains = ",".join(sorted(ex["skill_domains"])) or "-"
        print(f"    [{score:.3f}] {ex['id']}  status={ex['status']:<20} "
              f"domains={domains:<12} {ex['instruction']}")


print(f"Bank: {bank_summary(EXAMPLES)}\n")

for query in TEST_QUERIES:
    print(f'Query: "{query}"')
    show("NAIVE top-4", retriever.retrieve(
        query, k=4, guarantee_rejection=False, guarantee_combined_domain=False))
    show("GUARANTEED top-4", retriever.retrieve(query, k=4))
    print()

print("--- Formatted prompt block for one query ---")
print(format_examples_for_prompt(retriever.retrieve(TEST_QUERIES[0], k=2)))
