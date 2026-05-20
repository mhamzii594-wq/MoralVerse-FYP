import os
import logging
import time
import random
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
    if override and override.lower() in {"openai", "gemini", "groq", "modelslab", "fireworks", "stub"}:
        return override.lower()
    env_provider = os.getenv("LLM_PROVIDER", "").lower()
    if env_provider in {"openai", "gemini", "groq", "modelslab", "fireworks"}:
        return env_provider
    if os.getenv("FIREWORKS_API_KEY"):
        return "fireworks"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "modelslab"


# ---------------------------------------------------------------------------
# Shared LLM caller  (avoids repeating provider logic in every function)
# ---------------------------------------------------------------------------

def _call_llm(provider: str, system: str, user: str, temperature: float = 0.95) -> str:
    """Call the given provider and return raw text content."""
    if provider == "fireworks":
        from openai import OpenAI as _OAI
        api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set")
        model_id = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
        client = _OAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=2048,
            temperature=temperature,
        )
        result = (resp.choices[0].message.content or "").strip()
        if not result:
            raise RuntimeError("Fireworks AI returned empty output")
        return result

    if provider == "modelslab":
        # ModelsLab v7 LLM API is OpenAI-compatible — use OpenAI SDK with custom base_url
        from openai import OpenAI as _OAI
        api_key = os.getenv("MODELSLAB_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("MODELSLAB_API_KEY not set")
        model_id = os.getenv("MODELSLAB_LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
        client = _OAI(api_key=api_key, base_url="https://modelslab.com/api/v7/llm")
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=2048,
            temperature=temperature,
            top_p=0.95,
        )
        result = (resp.choices[0].message.content or "").strip()
        if not result:
            raise RuntimeError("ModelsLab LLM returned empty output")
        return result

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
        groq_model = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
        resp = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=2048,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def _call_llm_with_retry(
    provider: str, system: str, user: str, temperature: float = 0.95, retries: int = 3
) -> str:
    """Call _call_llm with exponential back-off retry on transient failures."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        try:
            return _call_llm(provider, system, user, temperature)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s before final raise
                logger.debug("LLM %s attempt %d failed, retrying in %ds: %s", provider, attempt + 1, wait, exc)
                import time as _time
                _time.sleep(wait)
    raise last_exc


# ---------------------------------------------------------------------------
# Character anchor — locked description injected into EVERY image prompt
# ---------------------------------------------------------------------------

_GIRL_NAMES = {
    "fatima", "fatimah", "aisha", "ayesha", "zara", "sara", "maryam", "amina", "layla", "hana",
    "sofia", "emma", "olivia", "lily", "ella", "ava", "mia", "aria",
    "noor", "hira", "sana", "aliya", "rania", "dua", "zainab", "khadija",
    "asma", "ruqayyah", "mahnoor", "iman", "saba",
    "tooba", "tuba", "hafsa", "eshal", "areeba", "laiba", "anaya", "mahira",
    "nadia", "maria", "lena", "nina", "ana", "anna", "luna", "maya",
    "lila", "leila", "yasmin", "jasmine", "rose", "ruby", "grace",
    "claire", "chloe", "sophie", "isabel", "isabella", "natasha",
}


def _build_character_anchor(child_name: str, child_age: int, avatar_desc: str, gender_override: str = "") -> str:
    """
    Build a locked character description injected VERBATIM into every FLUX scene prompt.

    Design principles:
    - Gender is always explicit ("boy"/"girl", never "child") — prevents FLUX from
      switching gender between scenes.
    - Specific clothing colours are always named — "red t-shirt and blue shorts" locks
      the outfit across 6 scenes; "colorful simple outfit" does not.
    - Chibi proportion cues ("round chubby face, big head relative to body") prevent
      FLUX from generating a teenager or adult in later scenes.
    - "same character consistent appearance throughout" is a FLUX-specific consistency
      signal that reduces drift across a long prompt sequence.
    - When describe_avatar succeeds (real photo uploaded), the full vision description
      replaces all defaults so the real child's traits are used.
    """
    age = int(child_age) if child_age else 8
    desc = (avatar_desc or "").strip().rstrip(".")
    is_fallback = not desc or desc.lower() in {"a friendly child", "friendly child", ""}

    # Pixar 3D suffix — keeps proportions cartoon-rounded (not realistic) while
    # signalling 3D volume to FLUX, matching the _STYLE_TAG's Pixar CGI request.
    _pixar3d = "Pixar-style rounded face, large expressive eyes, soft 3D volume, smooth CGI proportions"

    if is_fallback:
        ov = (gender_override or "").strip().lower()
        if ov in ("boy", "girl"):
            gender = ov
        else:
            name_parts = (child_name or "").lower().strip().split()
            name_lower = name_parts[0] if name_parts else ""
            gender = "girl" if name_lower in _GIRL_NAMES else "boy"
        outfit = ("yellow sundress and white sandals" if gender == "girl"
                  else "red t-shirt and blue shorts and white sneakers")

        return (
            f"a {age}-year-old {gender} with short black hair, dark brown eyes, "
            f"{outfit}, {_pixar3d}"
        )

    # Vision description available — strip any leading prose first
    for prefix in (
        "the child is ", "this child is ", "a child with ", "they are wearing ",
        "they have ", "child has ", "this is a child with ",
    ):
        if desc.lower().startswith(prefix):
            desc = desc[len(prefix):]
            break

    # Detect gender: explicit override > vision description > name lookup
    ov = (gender_override or "").strip().lower()
    desc_lower = desc.lower()
    if ov in ("boy", "girl"):
        gender = ov
    elif desc_lower.startswith("girl") or ", girl" in desc_lower or "female" in desc_lower:
        gender = "girl"
    elif desc_lower.startswith("boy") or ", boy" in desc_lower or "male" in desc_lower:
        gender = "boy"
    else:
        name_parts = (child_name or "").lower().strip().split()
        name_lower = name_parts[0] if name_parts else ""
        gender = "girl" if name_lower in _GIRL_NAMES else "boy"

    return f"a {age}-year-old {gender} with {desc}, {_pixar3d}"


# ---------------------------------------------------------------------------
# Pass 2 — generate image prompts from scene texts  (focused, no drift)
# ---------------------------------------------------------------------------

def _generate_image_prompts_for_scenes(
    scenes: list,
    character_anchor: str,
    child_name: str,
    provider: str,
    supporting_characters: list | None = None,
) -> list:
    """
    Pass-2: translate each scene text into a FLUX image prompt.

    Architecture:
    - The LLM generates ONLY the 5 visual fields per scene (action, props,
      setting, others, lighting). It never sees the anchor or style tag.
    - Python assembles the final prompt: anchor + fields + style_tag.
    - This makes the anchor and style tag 100% deterministic — the LLM
      cannot paraphrase or omit them.
    - A validation pass checks noun coverage and anchor integrity.
    """
    import json as _json

    if not scenes:
        return []

    scene_lines = []
    for s in scenes:
        if isinstance(s, dict):
            sid = s.get("id", "?")
            txt = s.get("text", "")
            scene_lines.append(f"Scene {sid}: \"{txt}\"")
    if not scene_lines:
        return []

    # Locked descriptions for recurring secondary characters (from Pass 1) — reused verbatim
    # in the 'others' field so they look identical every time they appear.
    _sc_block = ""
    if supporting_characters:
        _lines = [
            f"  - {c.get('name','?')}: {c.get('visual','').strip()}"
            for c in supporting_characters
            if isinstance(c, dict) and c.get("visual")
        ]
        if _lines:
            _sc_block = (
                "RECURRING CHARACTERS (locked descriptions — when a scene includes one of these "
                "people, the 'others' field MUST reuse their description below VERBATIM, never invent a new look):\n"
                + "\n".join(_lines) + "\n\n"
            )

    system = (
        "You are a professional prompt engineer specialising in FLUX text-to-image AI for "
        "children's picture books.\n\n"

        "YOUR ONLY JOB: for each scene sentence, extract the 5 pure visual fields listed below.\n"
        f"For context, the protagonist is: \"{character_anchor}\"\n"
        "Do NOT repeat the character description in your output — code prepends it automatically. "
        "The 'action' field must contain ONLY the protagonist's physical movement (no appearance, no name).\n"
        "You do NOT write the art style tag — that is added by code.\n"
        "You write ONLY what changes between scenes: action, props, setting, others, lighting.\n\n"

        + _sc_block +

        "══════════════════════════════════════════════\n"
        "THE 5 FIELDS — rules for each\n"
        "══════════════════════════════════════════════\n\n"

        "FIELD 1 — action  (REQUIRED, never empty)\n"
        "  Describe ONLY the character's precise physical movement right now — NO appearance, NO name "
        "(code adds the character description automatically).\n"
        "  Must contain: a concrete movement verb + the body part performing it.\n"
        "  ✓ Good: 'crouching on the ground, both hands cupped around a sparrow'\n"
        "  ✓ Good: 'extending both arms forward, holding a trophy toward a teacher'\n"
        "  ✓ Good: 'climbing a tree trunk, left foot on a branch, right hand clutching bark'\n"
        "  ✗ Bad: starting with the character's appearance/age/clothing (that is added by code)\n"
        "  ✗ Bad:  'feeling scared'  'thinking about'  'realizing the truth'  'learning a lesson'\n"
        "  EMOTION → BODY LANGUAGE — never write emotion words, translate them:\n"
        "    scared/terrified → wide eyes, one foot stepped back, both hands raised\n"
        "    guilty/ashamed   → head drooping forward, shoulders slumped, gaze fixed on feet\n"
        "    sad/crying       → eyes downcast, lower lip trembling, tears on cheeks\n"
        "    happy/excited    → arms raised wide, big open smile, weight on toes\n"
        "    proud/confident  → chest out, chin raised, standing tall with hands on hips\n"
        "    confused         → head tilted to one side, one hand raised touching chin\n"
        "    angry            → fists clenched at sides, eyebrows pushed down hard\n"
        "    brave/determined → one foot stepped forward, chin raised, back straight\n"
        "    surprised        → both hands at cheeks, mouth open, eyes wide\n"
        "    relieved         → shoulders dropping, long exhale, small smile\n"
        "    nervous/anxious  → shoulders raised, arms crossed tight, eyes darting side to side\n"
        "    thrilled         → jumping slightly off ground, both arms flung wide, mouth open in a grin\n"
        "    lonely/isolated  → sitting hunched, knees pulled up, chin resting on folded arms\n"
        "    embarrassed      → one hand raised covering cheek, face turned slightly away, eyes down\n"
        "    disappointed     → head hanging down, arms limp at sides, slow exhale visible\n"
        "    determined/resolved → jaw set, eyes locked forward, one fist raised at chest level\n"
        "    exhausted/tired  → leaning against wall or object, eyes half-closed, one hand on forehead\n"
        "    wondering        → chin tilted up, eyes looking upward and to one side, index finger on lips\n\n"

        "FIELD 2 — props  (REQUIRED if scene names any object; empty string if none)\n"
        "  Every physical object named in the scene text, each described with color+size+material.\n"
        "  ✓ 'a small brown leather wallet' not 'a wallet'\n"
        "  ✓ 'a tall gold trophy with a star on top' not 'a trophy'\n"
        "  ✓ 'a thick blue hardcover book with gold title lettering' not 'a book'\n"
        "  ✓ 'a folded white envelope sealed with red wax' not 'a letter'\n"
        "  ✓ 'a shiny red bicycle with silver handlebars and a black seat' not 'a bike'\n"
        "  ✓ 'a small injured brown sparrow with one drooping wing' not 'a bird'\n"
        "  If scene has multiple objects, list all: 'a blue lunchbox and a wrapped white sandwich'\n"
        "  PROP CONSISTENCY: If a prop appeared in an earlier scene, use the exact same "
        "description (same color, size, material) — do not abbreviate or vary it.\n"
        "  OUTFIT LOCK: Never change the protagonist's clothing between scenes — code handles their "
        "appearance, so do not add or alter clothing in any field.\n\n"

        "FIELD 3 — setting  (REQUIRED, never empty)\n"
        "  Location name + exactly 2 specific visible background elements.\n"
        "  Must match the story world (village market ≠ modern mall; ancient school ≠ modern classroom).\n"
        "  ✓ 'stone village market path with colorful vegetable stalls and hanging red lanterns'\n"
        "  ✓ 'school classroom with rows of wooden desks and a green chalkboard on the wall'\n"
        "  ✓ 'forest path with tall trees casting long shadows and moss-covered rocks'\n"
        "  ✓ 'home living room with a wooden cabinet and framed family photos on the wall'\n"
        "  ✓ 'grassy school playground with a metal swing set and a red brick school building'\n"
        "  SETTING CONSISTENCY: If consecutive scenes share a location, reuse the same setting "
        "description so the place looks identical between clips.\n\n"

        "FIELD 4 — others  (REQUIRED if scene mentions another person; empty string if none)\n"
        "  Visual description of every other person in the scene. NEVER use their name.\n"
        "  Format: [age group] + [gender] + [clothing: color+type] + [hair] + [expression+posture]\n"
        "  ✓ 'an elderly woman with grey hair tied back, wearing a dark green headscarf and "
        "brown salwar kameez, hands clasped together, eyes wide with relief'\n"
        "  ✓ 'a middle-aged man in a white shopkeeper apron and blue shirt, pointing at shelves'\n"
        "  ✓ 'a young boy in a red school uniform with black hair, sitting on the ground, "
        "both hands pressed to a scraped knee'\n"
        "  If multiple people, describe each separated by semicolons.\n\n"

        "FIELD 5 — lighting  (REQUIRED, never empty)\n"
        "  One phrase. Match scene emotion to lighting mood:\n"
        "    joy / triumph / pride      → warm bright golden sunlight\n"
        "    fear / danger / tension    → cold dim grey light with deep shadows\n"
        "    guilt / regret / sadness   → flat grey overcast afternoon light\n"
        "    mystery / discovery        → dappled light filtering through leaves\n"
        "    night / darkness           → cool moonlit blue light\n"
        "    calm / resolution          → soft warm afternoon sunlight\n"
        "    excitement / adventure     → bright clear morning sunlight\n"
        "    anger / conflict           → harsh orange-red sunset light\n\n"

        "══════════════════════════════════════════════\n"
        "EXAMPLES — study these carefully, they show ALL rules in action\n"
        "══════════════════════════════════════════════\n\n"

        "── COURAGE (solo, emotion→body language) ──\n"
        f"Scene: \"{child_name} was terrified but stepped forward onto the dark stage in front of the whole school\"\n"
        "Output:\n"
        "{\n"
        '  "action": "taking one slow step forward onto a dark wooden stage, one foot placed '
        'forward, chin raised, hands slightly trembling at sides, weight leaning forward",\n'
        '  "props": "",\n'
        '  "setting": "school auditorium stage with dark velvet curtains on both sides and '
        'rows of children seated in the audience below",\n'
        '  "others": "",\n'
        '  "lighting": "bright warm spotlight shining down from above"\n'
        "}\n\n"

        "── HONESTY (prop + secondary character + emotion→body language) ──\n"
        f"Scene: \"{child_name} felt guilty and returned the gold coin to the shopkeeper who had been searching everywhere\"\n"
        "Output:\n"
        "{\n"
        '  "action": "stepping forward, one arm extended, placing a shiny gold coin into an '
        'open palm, head slightly bowed, shoulders relaxed after a heavy sigh",\n'
        '  "props": "a single shiny gold coin",\n'
        '  "setting": "small village shop with wooden shelves of jars and goods and a '
        'worn wooden counter",\n'
        '  "others": "a stout middle-aged man in a white shopkeeper apron and brown cap, '
        'both hands open and extended forward, eyes wide with surprise and relief",\n'
        '  "lighting": "warm soft indoor lamp light"\n'
        "}\n\n"

        "── FRIENDSHIP (multiple props + secondary character in distress) ──\n"
        f"Scene: \"{child_name} shared her only sandwich with her friend who had forgotten her lunchbox at home\"\n"
        "Output:\n"
        "{\n"
        '  "action": "holding a wrapped sandwich out with both hands, leaning forward slightly, '
        'a small warm smile on face",\n'
        '  "props": "a triangle-shaped sandwich wrapped in white paper",\n'
        '  "setting": "school cafeteria with long wooden lunch tables and benches and '
        'other children eating in the background",\n'
        '  "others": "a girl with short black hair in a yellow school uniform, seated at '
        'the table, reaching both hands forward, eyes bright with gratitude",\n'
        '  "lighting": "warm bright indoor cafeteria light"\n'
        "}\n\n"

        "── CONCLUSION (moral lesson — no physical event, translate lesson into body language) ──\n"
        f"Scene: \"{child_name} understood that kindness always comes back to you. She felt peaceful and happy.\"\n"
        "Output:\n"
        "{\n"
        '  "action": "standing upright with a soft warm smile, both hands relaxed at sides, '
        'head slightly raised, chest open and at ease",\n'
        '  "props": "",\n'
        '  "setting": "school playground with sunlit green grass and other children '
        'playing happily in the background",\n'
        '  "others": "",\n'
        '  "lighting": "warm bright golden sunlight"\n'
        "}\n\n"

        "── DECISION (character at a choice point — show them at the exact moment of decision) ──\n"
        f"Scene: \"{child_name} found a wallet on the ground. Should he keep the money or return it?\"\n"
        "Output:\n"
        "{\n"
        '  "action": "crouching down on one knee, one hand reaching toward a wallet on the ground, '
        'head tilted as eyes look at it closely, other hand hovering mid-air",\n'
        '  "props": "a worn brown leather wallet lying on the dusty ground with a small zip pocket visible",\n'
        '  "setting": "stone village path with a dusty road surface and low mud-brick walls on both sides",\n'
        '  "others": "",\n'
        '  "lighting": "flat grey overcast afternoon light"\n'
        "}\n\n"

        "── EMOTIONAL STATIC (internal feeling — translate entirely into body posture, no physical event) ──\n"
        f"Scene: \"{child_name} felt nervous standing at the school entrance, not knowing if anyone would be her friend.\"\n"
        "Output:\n"
        "{\n"
        '  "action": "standing still near a doorway, both arms hugging own chest, shoulders raised high, '
        'eyes cast downward toward feet, weight shifted to one side",\n'
        '  "props": "",\n'
        '  "setting": "school entrance corridor with lockers along the wall and other children walking past in the background",\n'
        '  "others": "several children in blue school uniforms walking past without looking, each carrying a schoolbag",\n'
        '  "lighting": "cold flat indoor fluorescent light"\n'
        "}\n\n"

        "══════════════════════════════════════════════\n"
        "NARRATION MATCH — CRITICAL RULE\n"
        "══════════════════════════════════════════════\n"
        "The 5 fields you output MUST together make it visually obvious what the narration is about.\n"
        "A viewer looking at the generated image, without reading the scene text, must be able to tell:\n"
        "  (a) WHAT the character is physically doing right now\n"
        "  (b) WHAT object(s) are involved\n"
        "  (c) WHERE the scene takes place\n"
        "If the narration says 'returned the wallet' → the image must show a hand giving a wallet.\n"
        "If the narration says 'climbed the tree' → the image must show the character on the tree.\n"
        "NEVER generate a generic standing/smiling pose when the scene describes a specific action.\n\n"

        "══════════════════════════════════════════════\n"
        "OUTPUT FORMAT\n"
        "══════════════════════════════════════════════\n"
        "Return ONLY valid JSON. No markdown fences. No explanation. No extra keys.\n"
        "{\n"
        '  "scenes": [\n'
        '    {"action": "...", "props": "...", "setting": "...", "others": "...", "lighting": "..."},\n'
        '    {"action": "...", "props": "...", "setting": "...", "others": "...", "lighting": "..."}\n'
        "  ]\n"
        "}\n"
        "Array must have EXACTLY as many objects as scenes given — no more, no fewer."
    )

    user_msg = "Generate the 5 visual fields for each of these scenes:\n" + "\n".join(scene_lines)

    try:
        content = _call_llm_with_retry(provider, system, user_msg, temperature=0.7)
        if content:
            data = _json.loads(_clean_json(content))
            field_list = data.get("scenes", [])
            if isinstance(field_list, list) and len(field_list) == len(scenes):
                prompts = []
                for scene_data, fields in zip(scenes, field_list):
                    scene_text = scene_data.get("text", "") if isinstance(scene_data, dict) else ""
                    scene_id = scene_data.get("id", "?") if isinstance(scene_data, dict) else "?"
                    # Validate required fields — warn so logs surface LLM compliance issues
                    action_val = (fields.get("action") or "").strip()
                    setting_val = (fields.get("setting") or "").strip()
                    lighting_val = (fields.get("lighting") or "").strip()
                    if not action_val:
                        logger.warning("Pass-2 scene %s: 'action' empty — scene_text fallback will be used", scene_id)
                    elif len(action_val.split()) < 5:
                        logger.warning("Pass-2 scene %s: 'action' suspiciously short (%d words): %s", scene_id, len(action_val.split()), action_val)
                    if not setting_val:
                        logger.warning("Pass-2 scene %s: 'setting' empty — FLUX will guess background", scene_id)
                    if not lighting_val:
                        logger.warning("Pass-2 scene %s: 'lighting' empty", scene_id)
                    assembled = _assemble_scene_prompt(character_anchor, fields, scene_text)
                    validated = _validate_and_patch_prompt(assembled, scene_text, character_anchor)
                    prompts.append(validated)
                    logger.info("Pass-2 scene %s prompt (%d chars): %s…", scene_id, len(validated), validated[:80])
                return prompts
            logger.warning(
                "Pass-2 returned %d field-sets for %d scenes",
                len(field_list), len(scenes),
            )
    except Exception as e:
        logger.warning("Pass-2 image prompt generation failed: %s", e)

    # Fallback: assemble from scene text directly — still deterministic anchor+style
    fallbacks = []
    for s in scenes:
        if isinstance(s, dict):
            text = s.get("text", "")
            first = text.split(".")[0].strip()
            fields = {
                "action": first,
                "props": "",
                "setting": "",
                "others": "",
                "lighting": "warm natural light",
            }
            fallbacks.append(_assemble_scene_prompt(character_anchor, fields, text))
    return fallbacks


# ---------------------------------------------------------------------------
# Prompt assembly and validation helpers
# ---------------------------------------------------------------------------

_STYLE_TAG = (
    "Disney-Pixar CGI animated film, Pixar-quality 3D render, "
    "subsurface skin scattering, soft volumetric rim lighting, "
    "cinematic depth of field with bokeh background, "
    "rich saturated jewel-toned colors, smooth studio quality render, "
    "expressive Pixar cartoon face, full body visible head to toe, "
    "main character centered in frame, professional animation studio quality, "
    "ultra-detailed 3D surface textures, no flat shading, no cel-shading."
)

# Pass 3 constants — defined at module level for consistency and reuse
_REQUIRED_ENDING = "Character stays centered, full body in frame."
_MAX_VP_CHARS = 460

_SKIP_WORDS = frozenset({
    "once", "upon", "time", "then", "that", "this", "with", "from", "were",
    "been", "have", "their", "them", "they", "very", "into", "also", "about",
    "would", "could", "should", "because", "always", "every", "never", "when",
    "while", "where", "which", "after", "before", "since", "until", "though",
    "although", "through", "without", "toward", "towards", "already", "again",
    "there", "these", "those", "still", "just", "even", "only",
})


def _assemble_scene_prompt(anchor: str, fields: dict, scene_text: str = "") -> str:
    """
    Build the final FLUX prompt deterministically from the LLM's 5 visual fields.
    Action is placed FIRST so FLUX's early-token weighting emphasises what the
    character is doing, not just who they are. Anchor and style tag are injected
    by Python and cannot be paraphrased or omitted by the LLM.

    If the LLM returned an empty action field (invariant #3 violation), fall back
    to the first sentence of the scene text so the anchor never becomes token 1.
    """
    from ai_modules.prompt_builder import _condense_character
    condensed = _condense_character(anchor)

    action = (fields.get("action") or "").strip().rstrip(",.")
    if not action:
        # Pass 2 returned an empty action — derive a minimal one from the scene text.
        if scene_text:
            action = scene_text.split(".")[0].strip()
            logger.warning(
                "_assemble_scene_prompt: 'action' field empty — using scene text fallback: %s…",
                action[:60],
            )
        else:
            logger.warning("_assemble_scene_prompt: 'action' field empty and no scene_text")
    parts = []
    # Code-injected condensed character anchor FIRST — guarantees correct, consistent
    # identity tokens without relying on the LLM to copy the full description verbatim,
    # and frees the CLIP token budget for the scene-specific action below.
    if condensed:
        parts.append(condensed)
    if action:
        parts.append(action)
    for key in ("props", "setting", "others", "lighting"):
        val = (fields.get(key) or "").strip().rstrip(",.")
        if val:
            parts.append(val)
    return ", ".join(parts) + ". " + _STYLE_TAG


_SHORT_SKIP = frozenset({
    "the", "a", "an", "and", "but", "or", "in", "on", "at", "to", "of", "for",
    "with", "by", "from", "up", "out", "so", "as", "if", "it", "is", "was",
    "are", "were", "be", "has", "had", "did", "not", "she", "he", "his",
    "her", "its", "they", "them", "all", "one", "two", "can", "may", "will",
    "this", "that", "who", "into", "than", "him", "got", "get", "saw", "see",
    "had", "went", "did", "too", "now", "our", "its", "own", "new", "old",
})

_EMOTION_WORDS_PATCH = frozenset({
    "scared", "afraid", "guilty", "happy", "proud", "angry", "brave",
    "terrif", "excited", "worried", "nervous", "ashamed", "joyful",
    "miserable", "lonely", "confused", "deter", "hoped", "wished",
    "realiz", "decid", "wonder", "reliev", "disappoint", "thrilled",
    "embarrass", "exhaust", "frustrat", "delight", "content", "jealous",
})


def _validate_and_patch_prompt(prompt: str, scene_text: str, anchor: str) -> str:
    """
    Two-pass validation after assembly:
    1. Anchor integrity — if the prompt doesn't contain the anchor start, prepend it.
    2. Noun coverage — extract up to 8 content words from the scene text (including
       short action verbs) and patch any missing ones. Threshold is >= 1 missing word
       (was > 2) so even a single absent key term triggers a patch.
    """
    # Pass 1: anchor must appear early in the prompt (prepend the condensed form to match
    # _assemble_scene_prompt and avoid token bloat if it is somehow missing)
    if anchor[:25].lower() not in prompt.lower():
        from ai_modules.prompt_builder import _condense_character
        prompt = _condense_character(anchor) + ", " + prompt

    # Pass 2: key content-word coverage (includes short verbs like run/hit/eat/fly)
    scene_words = [
        w.lower().strip(".,!?\"'-()")
        for w in scene_text.split()
        if len(w) > 2 and w.lower().strip(".,!?\"'-()") not in _SHORT_SKIP
        and w.lower().strip(".,!?\"'-()") not in _SKIP_WORDS
    ]
    content_nouns = [
        w for w in scene_words
        if not any(w.startswith(e) for e in _EMOTION_WORDS_PATCH)
    ][:8]  # check up to 8 content words (was 6)

    prompt_lower = prompt.lower()
    missing = [w for w in content_nouns if w not in prompt_lower]
    if len(missing) >= 1:   # patch if ANY content word is absent (was > 2)
        patch = ", ".join(missing[:4])
        prompt = prompt.rstrip(". ") + f", {patch}."

    return prompt


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
    """Assemble a FLUX natural-language prompt for stub/fallback stories."""
    anchor = character_anchor or "a young child with big anime eyes, colorful outfit"
    return f"{anchor} {action_env}. {_STYLE_TAG}"


def _generate_stub_story(input_data: Dict[str, Any], character_anchor: str = "") -> Dict[str, Any]:
    child_name = input_data.get("child_name", "the child")
    moral_theme = input_data.get("moral_theme", "kindness")
    avatar_used = bool(input_data.get("avatar_path"))
    theme_lower = moral_theme.lower()

    p = lambda ae: _make_prompt(ae, character_anchor)  # noqa: E731

    if "honest" in theme_lower:
        scene1_text = f"Once upon a time, {child_name} was playing at the school playground."
        scene2_text = f"{child_name} found a lost wallet on the ground. It had someone's name on it."
        scene3_text = f"{child_name} felt torn — the wallet was full of coins, but someone must be missing it."
        scene4_text = f"{child_name} decided to return the wallet to the teacher's office right away."
        scene5_text = f"The teacher announced it over the speaker and the wallet's owner, a younger student, came running with tears in their eyes."
        scene6_text = f"The grateful student hugged {child_name} tightly. {child_name} learned that honesty is always the best choice."
        choice_a, choice_b = "Return the wallet", "Keep it for candy"
        choice2_a, choice2_b = "Give it to the teacher", "Leave it at the lost and found"
        scene1_img = p(f"{child_name} playing and running on a school playground with swings and a slide, red brick school building behind")
        scene2_img = p(f"{child_name} bending down picking up a brown leather wallet from a school playground path, other children playing behind")
        scene3_img = p(f"{child_name} standing on the playground holding the wallet and looking conflicted, coins visible inside, school building behind")
        scene4_img = p(f"{child_name} walking into a school office doorway holding a wallet out to a teacher, school corridor with lockers and sunlit windows")
        scene5_img = p(f"a small child running happily down a school corridor toward {child_name} who is standing by the office, sunlight through windows")
        scene6_img = p(f"two children hugging on school steps, {child_name} smiling warmly, golden afternoon light, potted flowers around")

    elif "courage" in theme_lower or "brave" in theme_lower:
        scene1_text = f"Once upon a time, {child_name} went on a hiking adventure in the mountains."
        scene2_text = f"{child_name} saw a dark cave entrance and heard a strange rumbling sound from inside."
        scene3_text = f"{child_name} felt frightened but curious — something glowed faintly deep within the cave."
        scene4_text = f"Taking a deep breath, {child_name} stepped inside and found a narrow passage leading deeper."
        scene5_text = f"At the heart of the cave, {child_name} discovered a chamber full of glowing blue crystals that lit up the walls."
        scene6_text = f"Heart pounding with joy, {child_name} stepped back into the sunlight. {child_name} learned that courage means facing your fears."
        choice_a, choice_b = "Enter the cave", "Wait outside"
        choice2_a, choice2_b = "Go deeper", "Stay near the entrance"
        scene1_img = p(f"{child_name} hiking up a winding mountain trail with pine trees and wildflowers, rocky peaks in the distance under blue sky")
        scene2_img = p(f"{child_name} standing at the dark entrance of a cave in a rocky mountainside, mossy rocks and ferns around the cave mouth, faint glow from inside")
        scene3_img = p(f"{child_name} peering into a cave entrance looking nervous but curious, hand resting on the rock wall, dim blue glow visible ahead")
        scene4_img = p(f"{child_name} stepping carefully through a narrow rocky cave passage, torch in hand, damp stone walls, faint light ahead")
        scene5_img = p(f"{child_name} inside a glowing crystal cave with blue crystals covering the walls and ceiling, soft blue light reflected everywhere, arms wide in wonder")
        scene6_img = p(f"{child_name} emerging from the cave holding a glowing crystal on a mountain path at sunset, orange-pink sky and valley below")

    elif "respect" in theme_lower:
        scene1_text = f"One afternoon, {child_name} was walking home through the neighborhood."
        scene2_text = f"{child_name} met an elderly neighbor struggling to carry heavy grocery bags up the steps."
        scene3_text = f"{child_name} wondered whether to stop — there was a favorite TV show waiting at home."
        scene4_text = f"{child_name} chose to help, taking the heavy bags and walking slowly beside the neighbor."
        scene5_text = f"Along the way, the neighbor told {child_name} wonderful stories about the neighborhood long ago."
        scene6_text = f"The neighbor smiled warmly at the door. {child_name} learned that showing respect makes the world kinder."
        choice_a, choice_b = "Help carry the bags", "Keep walking"
        choice2_a, choice2_b = "Carry both bags", "Help only with one"
        scene1_img = p(f"{child_name} walking along a quiet suburban street lined with blooming trees and colorful houses, afternoon sunlight")
        scene2_img = p(f"{child_name} stopping on a sidewalk watching an elderly woman struggling with two heavy grocery bags, a grocery store visible behind")
        scene3_img = p(f"{child_name} standing on the sidewalk looking uncertain, backpack on, elderly neighbor visible ahead struggling with bags")
        scene4_img = p(f"{child_name} taking heavy grocery bags from an elderly neighbor on the front steps of a house, smiling, tree-lined street behind")
        scene5_img = p(f"{child_name} and an elderly neighbor walking together along a garden path, neighbor pointing at old houses and telling stories, warm afternoon light")
        scene6_img = p(f"an elderly neighbor smiling and waving goodbye to {child_name} from the front door of a cozy house, garden full of flowers, warm afternoon light")

    else:  # Default: Kindness
        scene1_text = f"One sunny morning, {child_name} was exploring the meadow near the village."
        scene2_text = f"{child_name} found a small injured bird with a broken wing lying in the tall grass."
        scene3_text = f"{child_name} wanted to help but wasn't sure how — the bird looked frightened and in pain."
        scene4_text = f"{child_name} gently picked up the bird and carefully wrapped its wing with a soft strip of cloth."
        scene5_text = f"For the next three days, {child_name} visited the bird each morning, bringing seeds and fresh water."
        scene6_text = f"On the fourth morning, the bird hopped, stretched its wings, and flew up into the bright blue sky. {child_name} learned that kindness makes everything better."
        choice_a, choice_b = "Help the bird", "Leave it alone"
        choice2_a, choice2_b = "Care for it at home", "Leave it in the meadow"
        scene1_img = p(f"{child_name} walking through a wide sunlit meadow with tall wildflowers and golden grass, a village with red-roofed cottages on a hill in the background")
        scene2_img = p(f"{child_name} kneeling in tall grass gently holding a tiny injured bird with a drooping wing, wildflowers and insects around, soft dappled light")
        scene3_img = p(f"{child_name} crouching close to an injured bird in tall grass, hand outstretched gently, looking concerned and careful, warm meadow light")
        scene4_img = p(f"{child_name} sitting cross-legged in a meadow carefully wrapping a tiny bird's wing with a strip of cloth, warm sunlight, butterflies nearby")
        scene5_img = p(f"{child_name} kneeling beside a small wooden box with the bird inside, placing seeds and a tiny dish of water carefully, morning light")
        scene6_img = p(f"{child_name} standing in an open meadow looking up joyfully as a small bird flies upward into a bright blue sky, colorful wildflowers around")

    return {
        "title": f"{child_name} and the Lesson of {moral_theme.capitalize()}",
        "avatar_used": avatar_used,
        "character_anchor": character_anchor,
        "scenes": [
            {"id": 1, "text": scene1_text, "image_prompt": scene1_img, "decision": None},
            {"id": 2, "text": scene2_text, "image_prompt": scene2_img, "decision": None},
            {"id": 3, "text": scene3_text, "image_prompt": scene3_img, "decision": {"A": choice_a, "B": choice_b}},
            {"id": 4, "text": scene4_text, "image_prompt": scene4_img, "decision": None},
            {"id": 5, "text": scene5_text, "image_prompt": scene5_img, "decision": {"A": choice2_a, "B": choice2_b}},
            {"id": 6, "text": scene6_text, "image_prompt": scene6_img, "decision": None},
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
            # Use the assembled prompt directly — _validate_and_patch_prompt already
            # ensures full keyword coverage; _enrich_image_prompt is no longer called
            # as it prepended raw scene text and broke the action-first token order.
            scene["image_path"] = image_engine.generate_scene_image(
                raw_prompt,
                avatar_path=avatar_path,
                image_provider=image_provider,
                scene_text=scene_text,
            )
        except Exception:
            continue
    return story


# ---------------------------------------------------------------------------
# Pass 3 — generate video motion prompts from scene texts
# ---------------------------------------------------------------------------

def _generate_video_prompts_for_scenes(
    scenes: list,
    character_anchor: str,
    total_scenes: int,
    provider: str = "groq",
    allow_rule_fallback: bool = True,
) -> list:
    """
    Pass-3: for each scene text, ask the LLM to write a Kling i2v motion script.

    Each prompt describes:
      - What the character's body does (specific limb movements)
      - How the camera moves (pan / dolly / zoom / track)
      - What moves in the environment
    The image prompt covers composition/colour/style — the video prompt covers MOTION ONLY.
    Falls back to prompt_builder.build_video_prompt() if the LLM call fails.
    """
    import json as _json
    from ai_modules.prompt_builder import build_video_prompt as _rule_prompt, _condense_character

    if not scenes:
        return []

    condensed_char = _condense_character(character_anchor)

    scene_lines = []
    for i, s in enumerate(scenes, 1):
        if isinstance(s, dict):
            txt = (s.get("text") or "").strip()
            position_label = (
                "OPENING scene" if i == 1
                else "CLOSING scene" if i == total_scenes
                else f"MIDDLE scene {i}/{total_scenes}"
            )
            # Include the previous scene's text so the LLM can write a natural motion handoff
            prev_context = ""
            if i > 1 and isinstance(scenes[i - 2], dict):
                prev_txt = (scenes[i - 2].get("text") or "").strip()
                if prev_txt:
                    prev_context = f" [follows scene: \"{prev_txt[:70]}\"]"
            scene_lines.append(f"[{position_label}]{prev_context} Scene {s.get('id', i)}: \"{txt}\"")

    system = (
        "You are an AI video director writing image-to-video motion scripts for a children's Pixar-style 3D animated story.\n\n"

        f"CHARACTER appearing in every scene: {condensed_char}\n\n"

        "For each scene, write a motion script of exactly 3 sentences (max 450 characters total) that tells "
        "the image-to-video AI:\n"
        "  Sentence 1 — CHARACTER MOTION: what the character's body does in the first half of the clip. "
        "Name specific limbs, direction, and speed. The character is a 3D rendered figure with volume and weight.\n"
        "  Sentence 2 — CHARACTER CONTINUATION: what the character does in the second half (completing the action or reacting). "
        "Keep motion flowing naturally — avoid abrupt stops.\n"
        "  Sentence 3 — CAMERA + ENVIRONMENT: how the camera moves AND what moves in the background. "
        "Use cinematic 3D camera language — depth of field, bokeh, volumetric light rays.\n\n"

        "IDENTITY & STABILITY (CRITICAL for this model):\n"
        "  - Keep the character's face, proportions, and outfit IDENTICAL throughout the clip — "
        "natural weight and momentum, NO morphing, melting, warping, face-shifting, or extra limbs.\n"
        "  - Prefer SMOOTH, limited-range, physically-plausible motion. Avoid huge or fast full-body "
        "movements that distort the character. Subtle, believable motion looks best.\n"
        "  - Keep the camera SIMPLE (slow push-in, gentle pan, static wide, or pull-back). Do not request "
        "complex or fast camera moves the model cannot do cleanly.\n\n"

        "STRICT RULES:\n"
        "  - NARRATION SYNC: The motion you write must animate EXACTLY what the narration describes. "
        "If narration says 'climbed the tree' → character climbs. "
        "'returned the wallet' → arm extends handing object. "
        "'found a wallet' → character stops and crouches toward ground. "
        "NEVER write generic idle animation (swaying, blinking) when a specific action is narrated.\n"
        "  - MOTION ONLY — never describe colour, composition, or art style (the image already shows those).\n"
        "  - Be specific: 'steps forward with left foot, arms swinging naturally, then reaches out right hand toward the object' "
        "not just 'walks'.\n"
        "  - KEEP CHARACTER IN FRAME (CRITICAL): The character must remain fully visible — full body "
        "head-to-toe — for the ENTIRE clip. FORBIDDEN camera moves: zoom in on face, rack focus to face, "
        "close-up, push in tight, macro shot. These cause the character to drift off-frame mid-clip.\n"
        "  - ALLOWED camera moves: slow push in from wide to medium-full (stopping at full body), "
        "gentle side pan, cinematic pull back, overhead tilt down, static wide shot.\n"
        "  - Camera vocabulary: slow push in · cinematic pull back · side-scroll · overhead tilt down · "
        "static wide · gentle pan · dynamic tracking shot.\n"
        "  - For OPENING scenes: wide establishing shot slowly pushing in to medium-full body, bokeh background.\n"
        "  - For CLOSING scenes: cinematic pull back to reveal the full environment, warm light rays.\n"
        "  - For DECISION scenes (scene contains a choice/question): static wide shot, camera holds on character's full body with hesitating posture.\n"
        "  - For EMOTIONAL/STATIC scenes (character feeling, not doing): animate the body posture change — head drooping, shoulders curling, volumetric light dims.\n"
        "INTER-SCENE CAMERA ARC (mandatory for smooth video flow across all 6 clips):\n"
        "  Scene 1: Wide establishing shot, slow push-in to medium-full.\n"
        "  Scene 2: Medium-full, gentle pan or tilt following character movement.\n"
        "  Scene 3: Same height, tracking alongside the character.\n"
        "  Scene 4: Hold on character, slight environmental drift or overhead tilt.\n"
        "  Scene 5: Dynamic follow shot during action or decision moment.\n"
        "  Scene 6: Cinematic pull back from medium to wide, revealing full environment.\n"
        "  NEVER repeat the same camera direction in two consecutive scenes — vary between push-in, pan, tilt, track, hold, and pull-back.\n"
        "MOTION HANDOFF: If scene N shows the character arriving somewhere or finishing an action, scene N+1 should begin with them at the new location starting the next action — not repeating the same movement.\n"
        "  - End every prompt with exactly these 7 words: 'Character stays centered, full body in frame.'\n"
        "  - Max 450 characters per prompt including the closing 7 words.\n"
        "  - Return ONLY valid JSON. No markdown fences. No explanation.\n\n"

        "OUTPUT FORMAT:\n"
        "{\"scenes\": [{\"video_prompt\": \"...\"}, {\"video_prompt\": \"...\"}]}\n"
        "Array length must EXACTLY match the number of scenes given — no more, no fewer.\n\n"

        "EXAMPLES:\n"
        "[OPENING scene] \"Tooba walked through the farm watching chickens peck at the ground.\"\n"
        "→ {\"video_prompt\": \"Character walks forward, arms swinging naturally, head turning left toward chickens. "
        "Wide establishing shot pushes in slowly; chickens bob and peck, grass sways, bokeh background. "
        "Character stays centered, full body in frame.\"}\n\n"

        "[MIDDLE scene 3/6] \"She noticed the old rusty well filled with debris and decided to clean it.\"\n"
        "→ {\"video_prompt\": \"Character stops, head turns toward the well, hands settle on hips with weight. "
        "Camera tracks gently alongside at full-body distance; leaves drift past, well water ripples, volumetric dust motes. "
        "Character stays centered, full body in frame.\"}\n\n"

        "[CLOSING scene] \"Tooba smiled as the clean well sparkled in the afternoon light.\"\n"
        "→ {\"video_prompt\": \"Character turns to camera, face brightening into a warm smile, arms relaxing at sides. "
        "Cinematic pull back reveals full farm; golden light rays sweep across scene, birds take flight. "
        "Character stays centered, full body in frame.\"}\n\n"

        "[DECISION scene] \"She found a wallet on the street. Should she keep the money or return it?\"\n"
        "→ {\"video_prompt\": \"Character stops walking, crouches slowly, one hand reaching toward the wallet with hesitation. "
        "Static wide shot holds on the full body; dust particles drift, shallow depth of field blurs the background. "
        "Character stays centered, full body in frame.\"}\n\n"

        "[EMOTIONAL scene — no physical action] \"Omar felt deeply ashamed and could not look up.\"\n"
        "→ {\"video_prompt\": \"Character's head bows forward slowly, shoulders curl inward, one hand rises to cover face. "
        "Camera holds tight on face; volumetric light dims, soft shadow spreads across 3D scene. "
        "Character stays centered, full body in frame.\"}"
    )

    user_msg = (
        "These scenes form ONE continuous story played in sequence with 0.8s crossfades between clips. "
        "Write motion scripts that flow together as a single cinematic video:\n"
        "- Character's motion at the END of scene N should lead naturally into the START of scene N+1.\n"
        "- Use the camera arc prescribed in the rules — each clip must use a DIFFERENT camera direction.\n"
        "- The 6 clips must feel like ONE continuous journey, not 6 unrelated shots.\n\n"
        "Write a video motion script for each of these scenes:\n"
        + "\n".join(scene_lines)
    )

    try:
        content = _call_llm_with_retry(provider, system, user_msg, temperature=0.65)
        logger.debug("Pass-3 raw response: %s", content[:300])
        data = _json.loads(_clean_json(content))
        prompt_list = data.get("scenes", [])
        if isinstance(prompt_list, list) and len(prompt_list) == len(scenes):
            result = []
            for p in prompt_list:
                if not isinstance(p, dict):
                    result.append(None)
                    continue
                vp = (p.get("video_prompt") or "").strip()
                if vp:
                    # Enforce required ending tag
                    if not vp.endswith(_REQUIRED_ENDING):
                        base = vp.rstrip(". ")
                        candidate = base + ". " + _REQUIRED_ENDING
                        if len(candidate) <= _MAX_VP_CHARS:
                            vp = candidate
                        else:
                            available = _MAX_VP_CHARS - len(". " + _REQUIRED_ENDING)
                            vp = base[:available].rstrip() + ". " + _REQUIRED_ENDING
                    # Enforce char limit (trim if over, preserving the required ending)
                    elif len(vp) > _MAX_VP_CHARS:
                        base = vp[: -len(_REQUIRED_ENDING)].rstrip(". ")
                        available = _MAX_VP_CHARS - len(". " + _REQUIRED_ENDING)
                        vp = base[:available].rstrip() + ". " + _REQUIRED_ENDING
                    # Validate two sentences and minimum content length
                    if vp:
                        content_body = vp[:-len(_REQUIRED_ENDING)].rstrip(". ")
                        if content_body.count(". ") < 2:
                            logger.warning("Pass-3: fewer than 3 sentences (expected 3): %s…", vp[:80])
                        if len(content_body) < 40:
                            logger.warning("Pass-3: prompt body too short (%d chars before ending tag): %s", len(content_body), vp)
                result.append(vp if vp else None)
            valid = sum(1 for r in result if r)
            logger.info("Pass-3 video prompts generated: %d/%d valid", valid, len(result))
            if valid > 0:
                return result
        logger.warning(
            "Pass-3 returned %d prompts for %d scenes",
            len(prompt_list) if isinstance(prompt_list, list) else 0,
            len(scenes),
        )
        if not allow_rule_fallback:
            raise RuntimeError("Pass-3 produced no valid prompts")
    except Exception as exc:
        if not allow_rule_fallback:
            raise
        logger.warning("Pass-3 video prompt generation failed: %s — using rule fallback", exc)

    # Rule-based fallback
    fallback = []
    for i, s in enumerate(scenes, 1):
        text = (s.get("text") or "") if isinstance(s, dict) else ""
        fallback.append(
            _rule_prompt(
                text=text,
                character_anchor=character_anchor,
                position=i,
                total_scenes=total_scenes,
            )
        )
    return fallback


# ---------------------------------------------------------------------------
# Pass 1 — story text generation
# ---------------------------------------------------------------------------

def _validate_story_structure(data: dict, has_avatar: bool = False) -> list:
    """Return a list of human-readable structural problems with a Pass-1 story
    (empty list = valid). Used to trigger a single repair retry."""
    errors = []
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 6:
        errors.append(f"Must have EXACTLY 6 scenes (got {len(scenes) if isinstance(scenes, list) else 'none'}).")
        return errors  # further checks need 6 scenes

    for i, sc in enumerate(scenes, 1):
        has_decision = isinstance(sc, dict) and isinstance(sc.get("decision"), dict)
        if i in (3, 5) and not has_decision:
            errors.append(f"Scene {i} must have a non-null decision with A and B options.")
        if i not in (3, 5) and has_decision:
            errors.append(f"Scene {i} must NOT have a decision (only scenes 3 and 5 do).")
        if not (isinstance(sc, dict) and (sc.get("text") or "").strip()):
            errors.append(f"Scene {i} has empty text.")

    if (data.get("protagonist_gender") or "").strip().lower() not in ("boy", "girl"):
        errors.append("protagonist_gender must be 'boy' or 'girl'.")
    if not has_avatar and not (data.get("character_visual") or "").strip():
        errors.append("character_visual is required and must be non-empty.")

    total_words = sum(len((sc.get("text") or "").split()) for sc in scenes if isinstance(sc, dict))
    if total_words > 170:
        errors.append(f"Total story is {total_words} words; keep it to ~150 or fewer.")
    return errors


def _generate_story_with_llm(
    input_data: Dict[str, Any],
    provider: str,
    character_anchor: str,
) -> Dict[str, Any]:
    import json as _json

    child_name = input_data.get("child_name", "the child")
    child_age = input_data.get("child_age", 7)
    avatar_desc = input_data.get("avatar_description", "a friendly child")
    moral_theme = input_data.get("moral_theme", "kindness")
    preferred_language = (input_data.get("preferred_language") or "en").lower()
    language_label = "Urdu" if preferred_language == "ur" else "English"

    # PASS 1 — story text and decisions only (image_prompt is deliberately blank here)
    system_p1 = (
        "You are a children's interactive storytelling engine. "
        "Return ONLY valid JSON, no extra text.\n"
        "JSON structure:\n"
        "  {\"title\": string, \"avatar_used\": bool, \"protagonist_gender\": \"boy\" or \"girl\", \"character_visual\": string, "
        "\"supporting_characters\": [{\"name\": string, \"visual\": string}], \"scenes\": [\n"
        "    {\"id\": int, \"text\": string, \"image_prompt\": \"\", \"decision\": null or {\"A\": string, \"B\": string},\n"
        "     \"tts_prompt\": [{\"role\": string, \"text\": string}]}\n"
        "  ]}\n"
        "Rules:\n"
        "  - Exactly 6 scenes.\n"
        "  - Exactly 2 scenes with a non-null decision: scenes 3 and 5.\n"
        "  - Scene flow: (1) intro/setup, (2) rising action, (3) first challenge + decision, "
        "(4) consequence of decision, (5) climax + second decision, (6) resolution with moral lesson.\n"
        f"  - All scene `text` and decision texts in {language_label}.\n"
        "  - Each scene `text` is ~15-25 words (1-2 short sentences); the 6 scenes combined must not "
        "exceed ~150 words. Vary scene length naturally; do not pad. Each scene's length sets its clip timing.\n"
        "  - Each scene `text` must contain a VISIBLE PHYSICAL ACTION — something that can be shown in an image.\n"
        "    GOOD: 'Ali picked up the wallet from the dusty road.' (physical, visible)\n"
        "    BAD:  'Ali thought about what he should do.' (inner thought — invisible, cannot be illustrated)\n"
        "    Even emotional scenes must anchor to a physical posture: 'Sara sat alone, hugging her knees.'\n"
        "  - Decision option format: 10-20 words each, describing a SPECIFIC action the child would take.\n"
        "    One option must be the TEMPTING / easy / selfish choice, the other the VIRTUOUS choice that "
        f"embodies the moral theme ({moral_theme}). Both must be believable for a child.\n"
        "    GOOD: A: 'Keep the wallet and spend the coins on sweets for himself'\n"
        "          B: 'Return the wallet to find who it belongs to'\n"
        "    BAD:  A: 'Keep it'   B: 'Return it'  (too short — not enough for the story UI)\n"
        "  - Scene 4 (consequence) must follow the VIRTUOUS path of the scene-3 decision.\n"
        "  - Scene 6 must ONLY resolve the story and state the moral — never introduce new events.\n"
        "  - image_prompt must be an empty string \"\" — it is generated separately.\n"
        f"  - Make the story age-appropriate for a {child_age}-year-old and rich in detail.\n"
        f"  - {child_name} is the subject of every scene. Use their name in the first sentence, then pronouns "
        "within the same scene to avoid robotic repetition.\n"
        f"  - title: ≤6 words, evocative, and include {child_name}'s name (e.g. \"{child_name}'s Brave Morning\").\n"
        "  - Scene 1 must show the character actively doing something specific from the very first sentence — never open with 'Once upon a time there was...' alone.\n"
        "  - Every scene must be in a DIFFERENT location or show a clear visual change from the previous scene — no two consecutive scenes in identical settings.\n"
        f"  - protagonist_gender: REQUIRED. Must be \"boy\" or \"girl\" and MUST match the pronouns you use for {child_name} in the story (if you write \"she/her\", it is \"girl\").\n"
        f"  - character_visual: REQUIRED, NEVER empty. A single string (≤80 words) describing the protagonist's FULL appearance for image generation. "
        f"It MUST begin by stating the gender (the same gender as protagonist_gender). "
        f"Include: age, gender, hair colour/style, eye colour, skin tone, exact clothing with colours and any accessories. "
        f"Be concrete and culturally specific to {child_name}'s likely background. "
        f"Example: \"8-year-old Pakistani girl, warm olive skin, long dark hair in two braids, big round brown eyes, bright pink shalwar kameez with gold embroidery, small red earrings\". "
        f"This string is injected verbatim into every image prompt — NEVER change it between scenes.\n"
        "  - tts_prompt: For EACH scene, a list of spoken lines for text-to-speech. Rules:\n"
        "    ALWAYS start with narrator line that speaks the scene text. Add 1-2 character dialogue lines if characters speak.\n"
        "    Each text MUST be under 15 words. No punctuation in text field — plain spoken words only.\n"
        "    Available roles (use ONLY these): narrator, boy_hero, girl, old_man, villain, wise_woman, young_child, comic_relief, tough_guy, gentle_soul\n"
        "    The PROTAGONIST's own dialogue must use the role matching protagonist_gender: girl → \"girl\", boy → \"boy_hero\".\n"
        "    Example: [{\"role\": \"narrator\", \"text\": \"Ali found a lost wallet on the dusty road\"}, "
        "{\"role\": \"boy_hero\", \"text\": \"I must find who this belongs to\"}]\n"
        "  - supporting_characters: REQUIRED list (use [] if none qualify). Include ONLY non-protagonist "
        "characters who ACTUALLY appear in the text of 2 or more scenes (e.g. a friend, a bully, an old man). "
        "Do NOT invent characters who never appear. Each entry: `name` + a `visual` (one line: age group, "
        "gender, hair, clothing with colours, defining feature). "
        "This locked description is reused in every scene they appear in so they look identical throughout. "
        "Example: [{\"name\": \"Sara\", \"visual\": \"a 9-year-old girl with two black braids, a green frock and white sandals\"}]\n\n"
        "EXAMPLE (shape only — follow this structure exactly; abbreviated to 2 of the 6 scenes):\n"
        "{\n"
        '  "title": "Ali\'s Honest Choice", "avatar_used": false, "protagonist_gender": "boy",\n'
        '  "character_visual": "7-year-old Pakistani boy, light brown skin, short black hair, big brown eyes, blue kurta and white shalwar, brown sandals",\n'
        '  "supporting_characters": [{"name": "the shopkeeper", "visual": "a stout middle-aged man with a grey beard, white cap and brown apron"}],\n'
        '  "scenes": [\n'
        '    {"id": 1, "text": "Ali walked to the busy market, clutching a few coins to buy fruit for his mother.", "image_prompt": "", "decision": null, "tts_prompt": [{"role": "narrator", "text": "Ali walked to the busy market clutching his coins"}]},\n'
        '    {"id": 3, "text": "Ali saw a dropped wallet on the ground. He picked it up and looked around.", "image_prompt": "", "decision": {"A": "Keep the wallet and buy sweets for himself", "B": "Take the wallet to the shopkeeper to find its owner"}, "tts_prompt": [{"role": "narrator", "text": "Ali found a wallet lying on the ground"}, {"role": "boy_hero", "text": "Who could have dropped this"}]}\n'
        "  ]\n"
        "}"
    )

    _variation_settings = [
        "Set the story in a forest.", "Set the story in a city.", "Set the story by the ocean.",
        "Set the story in a village.", "Set the story in the mountains.", "Set the story on a farm.",
        "Set the story in a school.", "Set the story in a magical garden.", "Set the story near a river.",
        "Set the story in a snowy landscape.", "Set the story in a desert.", "Set the story in a jungle.",
    ]
    _variation = random.choice(_variation_settings)
    _seed = random.randint(1000, 9999)

    user_p1 = (
        f"Topic: {input_data.get('prompt')}\n"
        f"Child: {child_name}, Age: {child_age}, Theme: {moral_theme}\n"
        f"Variation seed: {_seed}. {_variation} "
        f"Create a UNIQUE story — do not reuse plot points from any previous story."
    )

    content_p1 = _call_llm_with_retry(provider, system_p1, user_p1)
    data = _json.loads(_clean_json(content_p1))

    # Structural validation + single repair retry — catches missing decisions, wrong scene
    # count, missing gender/character_visual, over-long stories before they reach the user.
    _has_avatar = bool(input_data.get("avatar_path"))
    _errs = _validate_story_structure(data, has_avatar=_has_avatar)
    if _errs:
        logger.warning("Pass-1 structure invalid, retrying once: %s", "; ".join(_errs))
        _repair_msg = (
            user_p1
            + "\n\nYour previous response violated these requirements:\n- "
            + "\n- ".join(_errs)
            + "\nReturn corrected JSON that fixes ALL of them."
        )
        try:
            content_p1 = _call_llm_with_retry(provider, system_p1, _repair_msg)
            _repaired = _json.loads(_clean_json(content_p1))
            if not _validate_story_structure(_repaired, has_avatar=_has_avatar):
                data = _repaired
                logger.info("Pass-1 repair retry produced a valid story")
            else:
                logger.warning("Pass-1 repair retry still invalid — proceeding with best effort")
                data = _repaired
        except Exception as _rex:
            logger.warning("Pass-1 repair retry failed (%s) — keeping original", _rex)

    data["avatar_used"] = _has_avatar

    # Use LLM-generated character_visual as anchor when no avatar is provided.
    # Avatar stories keep the vision-analysis anchor (more accurate than LLM imagination).
    character_visual = data.pop("character_visual", None)
    protagonist_gender = (data.pop("protagonist_gender", "") or "").strip().lower()
    if not input_data.get("avatar_path"):
        if character_visual:
            character_anchor = f"{character_visual.rstrip('. ')}, Pixar-style rounded face, large expressive eyes, soft 3D volume, smooth CGI proportions"
            logger.info("Character anchor updated from LLM character_visual (%d chars)", len(character_anchor))
        elif protagonist_gender in ("boy", "girl"):
            # character_visual missing — rebuild the name-based anchor with the LLM's gender
            character_anchor = _build_character_anchor(
                child_name, child_age, avatar_desc, gender_override=protagonist_gender
            )
            logger.info("Character anchor rebuilt with LLM protagonist_gender=%s", protagonist_gender)

    data["character_anchor"] = character_anchor

    # PASS 2 — image prompts derived directly from each scene text
    scenes = data.get("scenes", [])
    _supporting = data.get("supporting_characters") or []
    image_prompts = _generate_image_prompts_for_scenes(
        scenes, character_anchor, child_name, provider, supporting_characters=_supporting
    )
    for i, scene in enumerate(scenes):
        if isinstance(scene, dict) and i < len(image_prompts):
            scene["image_prompt"] = image_prompts[i]

    # PASS 3 — video motion prompts for the cinematic pipeline.
    # Try providers in reliability order for structured JSON — Groq first.
    _vp_order = ["fireworks", "groq", "gemini", provider, "modelslab"]
    _vp_tried = []
    video_prompts = None
    for _vp in _vp_order:
        if _vp in _vp_tried:
            continue
        _vp_tried.append(_vp)
        if _vp == "fireworks" and not os.getenv("FIREWORKS_API_KEY"):
            continue
        if _vp == "groq" and not os.getenv("GROQ_API_KEY"):
            continue
        if _vp == "gemini" and not os.getenv("GEMINI_API_KEY"):
            continue
        if _vp == "modelslab" and not os.getenv("MODELSLAB_API_KEY"):
            continue
        if _vp == "openai" and not os.getenv("OPENAI_API_KEY"):
            continue
        try:
            vps = _generate_video_prompts_for_scenes(
                scenes, character_anchor, len(scenes), _vp, allow_rule_fallback=False
            )
            if vps and any(v for v in vps):
                video_prompts = vps
                logger.info("Pass-3 video prompts via provider: %s", _vp)
                break
        except Exception as _vpe:
            logger.warning("Pass-3 provider %s failed: %s", _vp, _vpe)

    # All LLM providers failed — fall back to rule-based prompts once for all scenes.
    if not video_prompts:
        from ai_modules.prompt_builder import build_video_prompt as _rule
        logger.warning("Pass-3: all providers failed — using rule fallback for all scenes")
        video_prompts = [
            _rule(
                text=(s.get("text") or "") if isinstance(s, dict) else "",
                character_anchor=character_anchor,
                position=i,
                total_scenes=len(scenes),
            )
            for i, s in enumerate(scenes, 1)
        ]

    if video_prompts:
        for i, scene in enumerate(scenes):
            if isinstance(scene, dict) and i < len(video_prompts) and video_prompts[i]:
                scene["video_prompt"] = video_prompts[i]

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

    # Try primary provider, then fall through all others before going to stub
    _fallback_order = ["fireworks", "groq", "gemini", "openai", "modelslab"]
    providers_to_try = [provider] + [p for p in _fallback_order if p != provider]

    last_error = None
    for attempt_provider in providers_to_try:
        # Skip providers with no key configured
        if attempt_provider == "fireworks" and not os.getenv("FIREWORKS_API_KEY"):
            continue
        if attempt_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            continue
        if attempt_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            continue
        if attempt_provider == "groq" and not os.getenv("GROQ_API_KEY"):
            continue
        if attempt_provider == "modelslab" and not os.getenv("MODELSLAB_API_KEY"):
            continue
        try:
            logger.info("Trying LLM provider: %s", attempt_provider)
            story = _generate_story_with_llm(input_data, attempt_provider, character_anchor)
            return _attach_images(story, image_provider, avatar_path, input_data)
        except Exception as e:
            logger.warning("LLM provider %s failed: %s", attempt_provider, e)
            last_error = e

    logger.error("All LLM providers failed — using stub story. Last error: %s", last_error)
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
        "Write EXACTLY 2 new scenes that continue from the chosen option: "
        "one scene showing the immediate consequence, one scene resolving the story with the moral lesson.\n"
        "Use scene IDs that continue from the last existing scene ID.\n"
        "Focus on the specific consequences of the choice made."
    )

    user_p1 = (
        f"The child ({child_name}) chose Option {option} at Scene {decision_scene_id}.\n"
        f"Existing Story: {_json.dumps(existing_story)}"
    )

    _fallback_order = ["groq", "gemini", "openai", "modelslab"]
    providers_to_try = [provider] + [p for p in _fallback_order if p != provider]

    last_error = None
    for attempt_provider in providers_to_try:
        if attempt_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            continue
        if attempt_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            continue
        if attempt_provider == "groq" and not os.getenv("GROQ_API_KEY"):
            continue
        if attempt_provider == "modelslab" and not os.getenv("MODELSLAB_API_KEY"):
            continue
        try:
            content = _call_llm_with_retry(attempt_provider, system_p1, user_p1)
            data = _json.loads(_clean_json(content))
            data["avatar_used"] = bool(input_data.get("avatar_path"))
            data["character_anchor"] = character_anchor

            new_scenes = data.get("scenes", [])
            _supporting = existing_story.get("supporting_characters") or data.get("supporting_characters") or []
            image_prompts = _generate_image_prompts_for_scenes(
                new_scenes, character_anchor, child_name, attempt_provider, supporting_characters=_supporting
            )
            for i, scene in enumerate(new_scenes):
                if isinstance(scene, dict) and i < len(image_prompts):
                    scene["image_prompt"] = image_prompts[i]

            # Video motion prompts for the new scenes (Groq first for reliable JSON)
            for _vp in ["groq", "gemini", attempt_provider]:
                if _vp == "groq" and not os.getenv("GROQ_API_KEY"):
                    continue
                if _vp == "gemini" and not os.getenv("GEMINI_API_KEY"):
                    continue
                try:
                    vps = _generate_video_prompts_for_scenes(
                        new_scenes, character_anchor, len(new_scenes), _vp
                    )
                    if vps and any(v for v in vps):
                        for i, scene in enumerate(new_scenes):
                            if isinstance(scene, dict) and i < len(vps) and vps[i]:
                                scene["video_prompt"] = vps[i]
                        break
                except Exception:
                    pass

            # Merge: keep all scenes that came BEFORE the decision point, then append new scenes.
            # Without this merge, story_json is replaced with only 2 scenes and the video loses
            # all the earlier scenes.
            pre_decision = [
                s for s in existing_story.get("scenes", [])
                if int(s.get("id", 0)) < int(decision_scene_id)
            ]
            merged = dict(existing_story)
            merged["scenes"] = pre_decision + new_scenes
            merged["character_anchor"] = character_anchor
            merged["avatar_used"] = bool(input_data.get("avatar_path"))

            return _attach_images(merged, input_data.get("image_provider"), input_data.get("avatar_path"), input_data)
        except Exception as e:
            logger.warning("Regeneration provider %s failed: %s", attempt_provider, e)
            last_error = e

    logger.error("All regeneration providers failed: %s", last_error)
    return existing_story
