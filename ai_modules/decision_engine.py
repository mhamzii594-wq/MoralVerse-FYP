"""
Decision engine for branching stories.

Given a story JSON and an option ("A" or "B"), this module adjusts the
following scenes to reflect the selected path. For now (30% scope), the
engine annotates the next scene(s) with a brief consequence note while
keeping the structure SRDS-compatible.
"""

from __future__ import annotations

from typing import Any, Dict


def apply_decision(story: Dict[str, Any], option: str) -> Dict[str, Any]:
    """
    Apply a user decision ("A" or "B") to the story.

    Strategy (simple but extendable):
    - Find the first scene that contains a non-null "decision" that hasn't been applied yet.
    - Append a short consequence message to the text of the immediately
      following scene, indicating the chosen path.
    - Mark the decision as applied and record the chosen option.
    """
    if not story or "scenes" not in story:
        return story

    scenes = story.get("scenes", [])
    if not isinstance(scenes, list):
        return story

    # Track which decisions have been made
    applied_decisions = story.get("applied_decisions", [])

    # Find the first decision point that hasn't been applied yet
    decision_index = None
    for idx, scene in enumerate(scenes):
        if isinstance(scene, dict) and scene.get("decision") and idx not in applied_decisions:
            decision_index = idx
            break

    if decision_index is None:
        # No unapplied decision point found, return story as-is
        return story

    follow_index = decision_index + 1
    if follow_index >= len(scenes):
        # Decision is at the last scene, mark it as applied
        applied_decisions.append(decision_index)
        story["applied_decisions"] = applied_decisions
        story["last_choice"] = option
        return story

    follow_scene = scenes[follow_index]
    if not isinstance(follow_scene, dict):
        return story

    # Append a short deterministic consequence note to the next scene's text.
    # This ensures the choice has a visible effect even if LLM regeneration fails.
    decision_obj = scenes[decision_index].get("decision", {})
    if isinstance(decision_obj, dict):
        chosen_label = decision_obj.get(f"option_{option.upper()}", "") or decision_obj.get(option.lower(), "")
    else:
        chosen_label = str(decision_obj) if decision_obj else ""

    if chosen_label:
        consequence_note = f" [Choice: {chosen_label}]"
        existing_text = follow_scene.get("text", "")
        if consequence_note not in existing_text:
            follow_scene["text"] = existing_text + consequence_note
            scenes[follow_index] = follow_scene

    # Mark this decision as applied
    applied_decisions.append(decision_index)
    story["applied_decisions"] = applied_decisions
    story["last_choice"] = option

    return story


