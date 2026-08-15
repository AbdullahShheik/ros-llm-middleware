"""
llm_skill_matcher.py
----------------------
given a URDF summary (from urdf_parser.py), asks an LLM to select which
skills from master_skills.json this robot type can perform. The LLM is
constrained to SELECT from the existing skill_id list, never to invent new
skill definitions -- this keeps matching a closed-set classification
(checkable) rather than open-ended generation (not checkable).

Mandatory safety net: every returned skill_id is validated against
master_skills.json before being trusted. This isn't a human-approval gate
(you chose auto-go-live) -- it's the same automatic code-side check
already used elsewhere in your pipeline (INVALID_SKILL validation, the
evidence-checking discussed for rejection judgments). An LLM selecting
from a fixed list can still typo or hallucinate an id that doesn't exist;
this is what catches that before it corrupts the registry.
"""

import json

from layer1.layer1_pipeline import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
)


def build_master_skill_prompt_block(master_skills):
    """Compact listing: skill_id, description, and capability tags for every
    master skill -- this is the full list the LLM chooses from."""
    lines = ["Master skill list (choose ONLY from these skill_ids):"]
    for s in master_skills:
        caps = ", ".join(s["requires_capabilities"])
        lines.append(f"  - {s['skill_id']}: {s['description']} [requires: {caps}]")
    return "\n".join(lines)


MATCHING_SYSTEM_PROMPT = """You are matching a robot's physical capabilities, \
described from its URDF, against a FIXED master list of possible skills.

Rules:
1. You may ONLY select skill_ids that appear in the provided master skill list.
   Do NOT invent, rename, or slightly modify any skill_id.
2. Select a skill only if the robot's physical structure plausibly supports it \
   (e.g. don't select gripper-based skills for a robot with no gripper-like \
   joints/links; don't select navigation/mapping skills for a robot with no \
   wheels or mobility joints).
3. For skills requiring capabilities not directly visible in URDF structure \
   (e.g. power_monitoring), use reasonable judgment based on the type of \
   platform described, and note your uncertainty in the reasoning field.
4. Output ONLY raw JSON, no markdown, no explanation outside the JSON, in \
   exactly this shape:
{
  "detected_capabilities": ["<capability>", ...],
  "matched_skill_ids": ["<skill_id>", ...],
  "reasoning": "<brief explanation of why each capability/skill was or was not matched>"
}
"""


def call_groq_matcher(urdf_summary_text, master_skills, model=GROQ_MODEL):
    """
    LLM-call implementation using Groq. Returns the raw text response (to be
    parsed by the caller) -- kept separate from parsing so
    match_skills_for_robot() can be tested with a mock instead of this.
    """
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    skill_block = build_master_skill_prompt_block(master_skills)
    user_prompt = f"{skill_block}\n\n{urdf_summary_text}\n\nReturn the JSON now."

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": MATCHING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def call_claude_matcher(urdf_summary_text, master_skills, model=CLAUDE_MODEL):
    """
    LLM-call implementation using Claude. Returns the raw text response (to
    be parsed by the caller) -- kept separate from parsing so
    match_skills_for_robot() can be tested with a mock instead of this.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    skill_block = build_master_skill_prompt_block(master_skills)
    user_prompt = f"{skill_block}\n\n{urdf_summary_text}\n\nReturn the JSON now."

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=MATCHING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")


def call_default_matcher(urdf_summary_text, master_skills, model=None):
    """Dispatches to call_claude_matcher or call_groq_matcher per LLM_PROVIDER."""
    if LLM_PROVIDER == "groq":
        return call_groq_matcher(urdf_summary_text, master_skills, model or GROQ_MODEL)
    return call_claude_matcher(urdf_summary_text, master_skills, model or CLAUDE_MODEL)


def match_skills_for_robot(urdf_summary_text, master_skills, call_llm_fn=None, model=None):
    """
    call_llm_fn: callable(urdf_summary_text, master_skills, model) -> raw text.
                 Defaults to call_default_matcher, which uses whichever LLM
                 provider LLM_PROVIDER selects (real API call). Tests should
                 pass a mock here instead of hitting the real API.

    Returns: {
        "raw_matched_skill_ids": [...],   # exactly what the LLM returned
        "valid_skill_ids": [...],         # after dropping anything not in master
        "invalid_skill_ids": [...],       # hallucinated/typo'd ids, if any
        "detected_capabilities": [...],
        "reasoning": "...",
    }
    """
    if call_llm_fn is None:
        call_llm_fn = call_default_matcher

    raw_response = call_llm_fn(urdf_summary_text, master_skills, model)

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM matcher did not return valid JSON: {e}\nRaw response: {raw_response}")

    raw_matched = parsed.get("matched_skill_ids", [])
    detected_capabilities = parsed.get("detected_capabilities", [])
    reasoning = parsed.get("reasoning", "")

    master_ids = {s["skill_id"] for s in master_skills}
    valid_ids = [sid for sid in raw_matched if sid in master_ids]
    invalid_ids = [sid for sid in raw_matched if sid not in master_ids]

    return {
        "raw_matched_skill_ids": raw_matched,
        "valid_skill_ids": valid_ids,
        "invalid_skill_ids": invalid_ids,
        "detected_capabilities": detected_capabilities,
        "reasoning": reasoning,
    }
