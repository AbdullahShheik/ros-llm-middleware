"""
test_robot_retrieval.py
-------------------------
Manual smoke test for robot-type retrieval. Prints, for each query, which
active robot types get selected and the resulting unioned skill set, so
arm-only / mobile-only / combined behavior is all visible.

Run from anywhere:
    python src/layer1/RAG/test_robot_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.robot_bank import bank_summary, load_robot_bank
from RAG.robot_retriever import CosineRobotTypeRetriever, union_skills

BANK = load_robot_bank()
retriever = CosineRobotTypeRetriever(BANK)

TEST_QUERIES = [
    "Pick up the red block and place it on the shelf.",           # arm-only
    "Inspect the blue cylinder and push it 5cm to the right.",    # arm-only
    "Navigate to the kitchen and wait there.",                    # mobile-only
    "Go to the storage room, pick up the toolbox, and bring it to the workbench.",  # combined
    "Drive to the shelf, grab the blue box, and drop it at the loading dock.",      # combined
]

print(f"Active fleet: {bank_summary(BANK)}\n")

for query in TEST_QUERIES:
    print(f'Query: "{query}"')
    selected = retriever.retrieve(query, k=3)
    for entry, score in selected:
        print(f"  [{score:.3f}] {entry['robot_type']} (x{entry['count']}, "
              f"{len(entry['skills'])} skills)")
    config = union_skills(selected)
    skill_names = sorted(s["name"] for s in config["skills"])
    print(f"  -> unioned skill block: {len(skill_names)} skills: {skill_names}\n")
