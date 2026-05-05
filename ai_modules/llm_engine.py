import os
import logging
import time
from typing import Any, Dict, List
from pathlib import Path
from django.conf import settings
from . import image_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _gemini_client():
    from google import genai
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def _groq_client():
    from groq import Groq
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def _extract_google_text(response) -> str:
    try:
        return response.text
    except Exception:
        return ""

def _clean_json(content: str) -> str:
    """Strip markdown code fences that some LLMs wrap around JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
    return content.strip()

def _get_provider(override: str | None = None) -> str:
    if override and override.lower() in {"openai", "gemini", "groq", "stub"}:
        return override.lower()
    return (os.getenv("LLM_PROVIDER") or "gemini").lower()


# ---------------------------------------------------------------------------
# Shared LLM caller  (avoids repeating provider logic in every function)
# ---------------------------------------------------------------------------

def _call_llm(provider: str, system: str, user: str) -> str:
    """Call the given provider and return raw text content."""
    if provider == "openai":
        client = _openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    elif provider == "gemini":
        client = _gemini_client()
        model = os.getenv("GEMINI_STORY_MODEL") or "gemini-1.5-flash"
        resp = client.models.generate_content(
            model=model,
            contents=system + "\n\n" + user,
        )
        return _extract_google_text(resp)
    else:  # groq
        client = _groq_client()
        resp = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Character anchor — locked description injected into EVERY image prompt
# ---------------------------------------------------------------------------

def _build_character_anchor(child_name: str, child_age: int, avatar_desc: str) -> str:
    """
    Build a short, locked character token for SD prompts.
    Using the EXACT SAME string in every prompt is what gives SD
    character consistency across scenes — same hair, same outfit, same face.
    """
    name = child_name.strip() or "the child"
    age = int(child_age) if child_age else 8
    desc = (avatar_desc or "").strip().rstrip(".")

    if not desc or desc.lower() in {"a friendly child", "friendly child", ""}:
        return f"{name} ghibli-style {age}-year-old child large round expressive eyes colorful simple outfit"

    # Remove sentence-level filler prefixes that add no visual info
    for prefix in (
        "the child is ", "this child is ", "a child with ", "they are wearing ",
        "they have ", "child has ", "this is a child with ",
    ):
        if desc.lower().startswith(prefix):
            desc = desc[len(prefix):]
            break

    return f"{name} ghibli-style {desc}"


# ---------------------------------------------------------------------------
# Pass 2 — generate image prompts from scene texts  (focused, no drift)
# ---------------------------------------------------------------------------

def _generate_image_prompts_for_scenes(
    scenes: list,
    character_anchor: str,
    child_name: str,
    provider: str,
) -> list:
    """
    Given the story's scene texts, generate one tightly-coupled image prompt
    per scene via a dedicated LLM call.  Separating this from story generation
    (pass 1) means the LLM focuses entirely on visual translation, not
    storytelling — which eliminates the text/image drift.
    """
    import json as _json

    if not scenes:
        return []

    scene_lines = []
    for s in scenes:
        if isinstance(s, dict):
            scene_lines.append(f"Scene {s.get('id', '?')}: \"{s.get('text', '')}\"")
    if not scene_lines:
        return []

    system = (
        "You convert children's story scene texts into highly specific image prompts.\n"
        "Rules:\n"
        "  1. Each prompt MUST describe the EXACT specific action happening in the scene.\n"
        "     - BAD: 'a child in a forest' (too vague, missing what the child is DOING)\n"
        "     - GOOD: 'a child kneeling beside a glowing mushroom in a dark pine forest at night'\n"
        "  2. Include: WHO is doing WHAT, WHERE, with WHAT objects, in WHAT lighting/time of day.\n"
        "  3. Put the character + action FIRST. Style tags go LAST.\n"
        "  4. Keep each prompt between 30-50 words.\n"
        "  5. Every key noun from the scene text MUST appear in the prompt.\n"
        "     - If the scene mentions 'a golden key', the prompt MUST include 'golden key'.\n"
        "     - If the scene mentions 'a dark cave', the prompt MUST include 'dark cave'.\n"
        "  6. Include the character's body pose (standing, kneeling, running, reaching, sitting).\n"
        "  7. Include specific environmental details from the text (not generic ones).\n"
        "  8. Never write a close-up or face portrait — always show full body in environment.\n\n"
        f"LOCKED CHARACTER (use in EVERY prompt): {character_anchor}\n\n"
        f"EXAMPLE — scene: '{child_name} found a sparkling gem hidden under an old oak tree'\n"
        f"  PROMPT: \"{child_name} kneeling under a large oak tree with twisted roots, "
        f"picking up a sparkling blue gem from the dirt, sunlight filtering through leaves, "
        f"{character_anchor}, ghibli anime style, masterpiece, detailed background\"\n\n"
        f"EXAMPLE — scene: '{child_name} returned the wallet to the teacher in the school office'\n"
        f"  PROMPT: \"{child_name} standing in a school office doorway handing a brown leather wallet "
        f"to a smiling teacher at a wooden desk, afternoon light through windows, "
        f"{character_anchor}, ghibli anime style, masterpiece, detailed background\"\n\n"
        "Return ONLY JSON: {\"image_prompts\": [\"...\", \"...\", \"...\", \"...\"]}"
    )

    user_msg = "Generate image prompts for these scenes:\n" + "\n".join(scene_lines)

    try:
        content = _call_llm(provider, system, user_msg)
        if content:
            data = _json.loads(_clean_json(content))
            prompts = data.get("image_prompts", [])
            if isinstance(prompts, list) and len(prompts) == len(scenes):
                return [str(p) for p in prompts]
            logger.warning("Pass-2 returned %d prompts for %d scenes", len(prompts), len(scenes))
    except Exception as e:
        logger.warning("Image prompt generation (pass 2) failed: %s", e)

    # Fallback: build prompts directly from scene text + anchor
    fallbacks = []
    for s in scenes:
        if isinstance(s, dict):
            text = s.get("text", "")
            first = text.split(".")[0].strip()
            fallbacks.append(
                f"{first}, {character_anchor}, ghibli anime style, masterpiece, wide angle, small full body figure"
            )
    return fallbacks


# ---------------------------------------------------------------------------
# Safety-net enrichment  (corrects prompts where action words are absent)
# ---------------------------------------------------------------------------

def _enrich_image_prompt(scene_text: str, image_prompt: str) -> str:
    """
    Ensure the scene's main action and key objects appear in the image prompt.
    If key elements are missing, prepend scene context to force the model to
    attend to the correct action. Action-first ordering is handled later by
    image_engine._reorder_prompt_for_sd().
    """
    if not image_prompt or not scene_text:
        return image_prompt or ""

    # Extract key words from the scene text (skip short filler words)
    scene_words = [w.lower().strip(".,!?\"'") for w in scene_text.split() if len(w) > 3]
    # Skip common filler words that don't define the visual scene
    skip_words = {"once", "upon", "time", "then", "that", "this", "with", "from",
                  "were", "been", "have", "their", "them", "they", "very",
                  "into", "also", "about", "would", "could", "should", "learned",
                  "decided", "because", "always", "every", "never", "some"}
    key_words = [w for w in scene_words if w not in skip_words][:8]

    matches = sum(1 for w in key_words if w in image_prompt.lower())

    # If less than 40% of key words match, the prompt is drifting from the text
    if len(key_words) > 0 and (matches / len(key_words)) < 0.4:
        # Inject the first two sentences for maximum context
        sentences = scene_text.split(".")
        context = ". ".join(s.strip() for s in sentences[:2] if s.strip())
        image_prompt = f"{context}, {image_prompt}"

    return image_prompt


# ---------------------------------------------------------------------------
# Stub story  (used when LLM call fails)
# ---------------------------------------------------------------------------

def _make_prompt(action_env: str, character_anchor: str) -> str:
    """Assemble a complete, anchor-injected image prompt."""
    parts = [action_env]
    if character_anchor:
        parts.append(character_anchor)
    parts.append("ghibli anime style, masterpiece, wide angle, small full body figure")
    return ", ".join(parts)


def _generate_stub_story(input_data: Dict[str, Any], character_anchor: str = "") -> Dict[str, Any]:
    child_name = input_data.get("child_name", "the child")
    moral_theme = input_data.get("moral_theme", "kindness")
    avatar_used = bool(input_data.get("avatar_path"))
    theme_lower = moral_theme.lower()

    p = lambda ae: _make_prompt(ae, character_anchor)  # noqa: E731

    if "honest" in theme_lower:
        scene1_text = f"Once upon a time, {child_name} was playing at the school playground."
        scene2_text = f"{child_name} found a lost wallet on the ground. It had someone's name on it."
        scene3_text = f"{child_name} decided to return the wallet to its owner right away."
        scene4_text = f"The owner was overjoyed! {child_name} learned that honesty is always the best choice."
        choice_a, choice_b = "Return the wallet", "Keep it for candy"
        scene1_img = p(f"{child_name} playing and running on a school playground with swings and a slide, red brick school building behind")
        scene2_img = p(f"{child_name} bending down picking up a brown leather wallet from a school playground path, other children playing behind")
        scene3_img = p(f"{child_name} walking into a school office doorway holding a wallet out to a teacher, school corridor with lockers and sunlit windows")
        scene4_img = p(f"{child_name} and a smiling teacher on school steps with the returned wallet, golden afternoon light, potted flowers around")

    elif "courage" in theme_lower or "brave" in theme_lower:
        scene1_text = f"Once upon a time, {child_name} went on a hiking adventure in the mountains."
        scene2_text = f"{child_name} saw a dark cave entrance and heard a strange rumbling sound from inside."
        scene3_text = f"Taking a deep breath, {child_name} stepped into the cave and discovered a glowing crystal."
        scene4_text = f"The cave was full of wonder! {child_name} learned that courage means facing your fears."
        choice_a, choice_b = "Explore the cave", "Run back home"
        scene1_img = p(f"{child_name} hiking up a winding mountain trail with pine trees and wildflowers, rocky peaks in the distance under blue sky")
        scene2_img = p(f"{child_name} standing at the dark entrance of a cave in a rocky mountainside, mossy rocks and ferns around the cave mouth, faint glow from inside")
        scene3_img = p(f"{child_name} inside a cave reaching toward a large glowing blue-green crystal embedded in the cave wall, stalactites on ceiling")
        scene4_img = p(f"{child_name} emerging from the cave holding a glowing crystal on a mountain path at sunset, orange-pink sky and valley below")

    elif "respect" in theme_lower:
        scene1_text = f"One afternoon, {child_name} was walking home through the neighborhood."
        scene2_text = f"{child_name} met an elderly neighbor struggling to carry heavy grocery bags."
        scene3_text = f"{child_name} helped the neighbor carry the bags all the way to their door."
        scene4_text = f"The neighbor smiled warmly. {child_name} learned that showing respect makes the world kinder."
        choice_a, choice_b = "Help carry the bags", "Keep walking"
        scene1_img = p(f"{child_name} walking along a quiet suburban street lined with blooming trees and colorful houses, afternoon sunlight")
        scene2_img = p(f"{child_name} stopping on a sidewalk watching an elderly woman struggling with two heavy grocery bags, a grocery store visible behind")
        scene3_img = p(f"{child_name} and an elderly neighbor walking together carrying grocery bags toward a cozy house with a garden gate, tree-lined path")
        scene4_img = p(f"an elderly neighbor smiling and waving goodbye to {child_name} from the front door of a cozy house, garden full of flowers, warm afternoon light")

    else:  # Default: Kindness
        scene1_text = f"One sunny morning, {child_name} was exploring the meadow near the village."
        scene2_text = f"{child_name} found a small injured bird with a broken wing lying in the tall grass."
        scene3_text = f"{child_name} gently picked up the bird and carefully bandaged its wing."
        scene4_text = f"The bird healed and flew away happily. {child_name} learned that kindness makes everything better."
        choice_a, choice_b = "Help the bird", "Leave it alone"
        scene1_img = p(f"{child_name} walking through a wide sunlit meadow with tall wildflowers and golden grass, a village with red-roofed cottages on a hill in the background")
        scene2_img = p(f"{child_name} kneeling in tall grass gently holding a tiny injured bird with a drooping wing, wildflowers and insects around, soft dappled light")
        scene3_img = p(f"{child_name} sitting cross-legged in a meadow carefully wrapping a tiny bird's wing with a strip of cloth, warm sunlight, butterflies nearby")
        scene4_img = p(f"{child_name} standing in an open meadow looking up joyfully as a small bird flies upward into a bright blue sky, colorful wildflowers around")

    return {
        "title": f"{child_name} and the Lesson of {moral_theme.capitalize()}",
        "avatar_used": avatar_used,
        "character_anchor": character_anchor,
        "scenes": [
            {"id": 1, "text": scene1_text, "image_prompt": scene1_img, "decision": None},
            {"id": 2, "text": scene2_text, "image_prompt": scene2_img, "decision": {"A": choice_a, "B": choice_b}},
            {"id": 3, "text": scene3_text, "image_prompt": scene3_img, "decision": None},
            {"id": 4, "text": scene4_text, "image_prompt": scene4_img, "decision": None},
        ],
    }


# ---------------------------------------------------------------------------
# Image attachment
# ---------------------------------------------------------------------------

def _attach_images(
    story: Dict[str, Any],
    image_provider: str | None = None,
    avatar_path: str | None = None,
    input_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scenes = story.get("scenes")
    if not isinstance(scenes, list):
        return story

    child_name = (input_data or {}).get("child_name", "")

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        raw_prompt = scene.get("image_prompt", "")
        scene_text = scene.get("text", "")
        if not raw_prompt:
            continue
        try:
            prompt = _enrich_image_prompt(scene_text, raw_prompt)
            # Use generate_scene_image for character consistency via avatar img2img
            scene["image_path"] = image_engine.generate_scene_image(
                prompt,
                avatar_path=avatar_path,
                image_provider=image_provider,
            )
        except Exception:
            continue
    return story


# ---------------------------------------------------------------------------
# Pass 1 — story text generation
# ---------------------------------------------------------------------------

def _generate_story_with_llm(
    input_data: Dict[str, Any],
    provider: str,
    character_anchor: str,
) -> Dict[str, Any]:
    import json as _json

    child_name = input_data.get("child_name", "the child")
    child_age = input_data.get("child_age", 7)
    moral_theme = input_data.get("moral_theme", "kindness")
    preferred_language = (input_data.get("preferred_language") or "en").lower()
    language_label = "Urdu" if preferred_language == "ur" else "English"

    # PASS 1 — story text and decisions only (image_prompt is deliberately blank here)
    system_p1 = (
        "You are a children's interactive storytelling engine. "
        "Return ONLY valid JSON, no extra text.\n"
        "JSON structure:\n"
        "  {\"title\": string, \"avatar_used\": bool, \"scenes\": [\n"
        "    {\"id\": int, \"text\": string, \"image_prompt\": \"\", \"decision\": null or {\"A\": string, \"B\": string}}\n"
        "  ]}\n"
        "Rules:\n"
        "  - Exactly 4 scenes.\n"
        "  - Exactly 2 scenes with a non-null decision.\n"
        f"  - All scene `text` and decision texts in {language_label}.\n"
        "  - Each scene `text` must describe ONE clear, specific action or event — what the character does or encounters.\n"
        "  - image_prompt must be an empty string \"\" — it is generated separately.\n"
        f"  - Make the story age-appropriate for a {child_age}-year-old and rich in detail."
    )

    user_p1 = f"Topic: {input_data.get('prompt')}\nChild: {child_name}, Age: {child_age}, Theme: {moral_theme}"

    content_p1 = _call_llm(provider, system_p1, user_p1)
    data = _json.loads(_clean_json(content_p1))
    data["avatar_used"] = bool(input_data.get("avatar_path"))
    data["character_anchor"] = character_anchor

    # PASS 2 — image prompts derived directly from each scene text
    scenes = data.get("scenes", [])
    image_prompts = _generate_image_prompts_for_scenes(scenes, character_anchor, child_name, provider)
    for i, scene in enumerate(scenes):
        if isinstance(scene, dict) and i < len(image_prompts):
            scene["image_prompt"] = image_prompts[i]

    return data


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_story(input_data: Dict[str, Any]) -> Dict[str, Any]:
    provider = _get_provider(input_data.get("llm_provider"))
    image_provider = input_data.get("image_provider")
    avatar_path = input_data.get("avatar_path")

    child_name = input_data.get("child_name", "the child")
    child_age = input_data.get("child_age", 7)
    avatar_desc = input_data.get("avatar_description", "a friendly child")
    character_anchor = _build_character_anchor(child_name, child_age, avatar_desc)

    try:
        story = _generate_story_with_llm(input_data, provider, character_anchor)
        return _attach_images(story, image_provider, avatar_path, input_data)
    except Exception as e:
        import traceback
        logger.error("LLM failure: %s\n%s", e, traceback.format_exc())
        stub = _generate_stub_story(input_data, character_anchor)
        return _attach_images(stub, image_provider, avatar_path, input_data)


def regenerate_story_after_decision(
    existing_story: Dict[str, Any],
    option: str,
    decision_scene_id: int,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    import json as _json

    if not existing_story or "scenes" not in existing_story:
        return existing_story

    provider = _get_provider(input_data.get("llm_provider"))
    child_name = input_data.get("child_name", "the child")
    child_age = input_data.get("child_age", 7)
    avatar_desc = input_data.get("avatar_description", "a friendly child")
    language_label = "Urdu" if input_data.get("preferred_language") == "ur" else "English"

    # Reuse existing anchor so character looks the same as earlier scenes
    character_anchor = (
        existing_story.get("character_anchor")
        or _build_character_anchor(child_name, child_age, avatar_desc)
    )

    system_p1 = (
        "You are a children's interactive storytelling engine. "
        "The child has made a decision. Continue the story from that choice.\n"
        "Return ONLY valid JSON with a 'scenes' list (same structure as before).\n"
        f"Write all scene `text` in {language_label}. "
        "Set image_prompt to \"\" in every scene — it will be generated separately.\n"
        "Focus on the specific consequences of the choice made."
    )

    user_p1 = (
        f"The child ({child_name}) chose Option {option} at Scene {decision_scene_id}.\n"
        f"Existing Story: {_json.dumps(existing_story)}"
    )

    try:
        content = _call_llm(provider, system_p1, user_p1)
        data = _json.loads(_clean_json(content))
        data["avatar_used"] = bool(input_data.get("avatar_path"))
        data["character_anchor"] = character_anchor

        # Pass 2 — image prompts for the continued scenes
        scenes = data.get("scenes", [])
        image_prompts = _generate_image_prompts_for_scenes(scenes, character_anchor, child_name, provider)
        for i, scene in enumerate(scenes):
            if isinstance(scene, dict) and i < len(image_prompts):
                scene["image_prompt"] = image_prompts[i]

        return _attach_images(data, input_data.get("image_provider"), input_data.get("avatar_path"), input_data)

    except Exception as e:
        logger.error("Regeneration failure: %s", e)
        return existing_story
