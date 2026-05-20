# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoralVerse.AI is a Django 4.2 web application that generates personalised moral story videos for children. A user submits a child's name, age, theme, language (English/Urdu), and optional avatar photo. The system runs a three-pass LLM pipeline, generates 6 scene illustrations, synthesises TTS audio, and assembles a final MP4 video.

**Production:** DigitalOcean Ubuntu 22.04 · nginx + Gunicorn (3 workers) · Supabase PostgreSQL 17.6  
**Domain:** moralverse.dev  
**Branch to deploy from:** `develop`

---

## Development Commands

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate           # Linux/macOS (production server)

# Install dependencies
pip install -r requirements.txt

# Run local dev server (SQLite by default)
python manage.py runserver

# Apply migrations
python manage.py migrate

# Create a migration after model changes
python manage.py makemigrations core

# Run the test suite
python manage.py test tests/

# Run a single test file
python manage.py test tests.test_pipeline

# Open Django shell
python manage.py shell
```

**Environment:** All secrets live in `.env` at the project root. Key variables:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Supabase session-pooler URL (port 5432) |
| `GROQ_API_KEY` | Primary LLM (llama-3.3-70b-versatile) |
| `MODELSLAB_API_KEY` | FLUX image gen, Kling v2.1 video, SadTalker |
| `OPENAI_API_KEY` | Fallback LLM + DALL-E 3 image fallback |
| `GEMINI_API_KEY` | Fallback LLM + Gemini image + TTS |
| `ELEVENLABS_API_KEY` | Urdu TTS (best quality) |
| `RESEND_API_KEY` | Transactional email via django-anymail |
| `SITE_URL` | Public base URL (used to build media URLs for APIs) |
| `DEBUG` | Default `True` locally; must be `False` in production |

---

## Architecture: The Three-Pass LLM Pipeline

Everything flows from `core/services/pipeline.py` → `ai_modules/`. The pipeline is the single most important file to understand.

### Pass 1 — Story Generation (`llm_engine.py: generate_story_json`)
- LLM produces a structured JSON: `{title, scenes[6], avatar_used}`
- Each scene has `{id, text, image_prompt:"", decision: null | {A, B}}`
- **Exactly scenes 3 and 5** have A/B decision dicts
- Temperature: `0.95` (high creativity)
- Provider priority: Groq → Gemini → OpenAI → ModelsLab → stub

### Pass 2 — Image Prompt Engineering (`llm_engine.py: generate_image_prompts`)
- LLM decomposes each scene into **5 visual fields**: `action`, `props`, `setting`, `others`, `lighting`
- Fields are assembled by Python (not LLM) into the final FLUX prompt:
  `action, character_anchor, props, setting, others, lighting. [_STYLE_TAG]`
- **Action is placed FIRST** so FLUX's early-token weighting prioritises what the character is *doing*
- Temperature: `0.70` (structured, consistent)
- After assembly, `_validate_and_patch_prompt()` checks that all key content words from the scene narration appear in the prompt (threshold: ≥ 1 missing word triggers a patch)

### Pass 3 — Video Motion Scripts (`llm_engine.py: generate_video_prompts`)
- LLM writes a 2-sentence Kling i2v motion script per scene (max 300 chars)
- Sentence 1: character body motion. Sentence 2: camera + environment
- Every prompt must end with `"Smooth 2D anime, cel-shaded."` — enforced in code post-parse
- Temperature: `0.65` (strict format compliance)
- Rule-based fallback in `ai_modules/prompt_builder.py` used if LLM fails

### Character Anchor System
- `_build_character_anchor()` in `llm_engine.py` creates a locked description:
  `"a {age}-year-old {gender} with {appearance}, chibi proportions, large anime eyes"`
- Injected into every FLUX prompt by Python — the LLM never touches it
- Gender inferred from avatar vision description, or from a name list fallback
- Default outfits: girl → yellow sundress; boy → red t-shirt + blue shorts

---

## Image Generation (`ai_modules/image_engine.py`)

- **Primary:** ModelsLab FLUX Dev (text2img, `model_id="flux"`)
- **Scene 2–6 with avatar:** FLUX Kontext Dev img2img (`model_id="flux-kontext-dev"`, `strength=0.60`)
- Inference steps: 30 · CFG text2img: 7.5 · CFG img2img: 6.5
- `_outfit_color_negatives(prompt)` appended to negative prompt for **both** text2img and Kontext
- Fallback chain: Pollinations → Stability AI → DALL-E 3 → Replicate → Gemini Imagen → stub
- `_assemble_scene_prompt(anchor, fields)` and `_validate_and_patch_prompt()` are the two key assembly functions in `llm_engine.py` (not in `image_engine.py`)

---

## Video Assembly

### Slideshow mode (`ai_modules/video_engine.py`)
- `ImageClip` per scene, crossfade 0.5s, background music ducked to 30%
- SRT subtitles burned via MoviePy `TextClip` + Pillow backend

### Cinematic mode (`ai_modules/video_engine_cinematic.py`)
- Kling v2.1 clip per scene via `ai_modules/img2video.py`
- Clip duration: audio ≥ 6s → 10s Kling clip, else 5s
- CDN polling: HEAD-poll `future_links[]` URL every 15s until `Content-Length > 10KB`
- Crossfade 0.8s, background music ducked to 12%
- Optional SadTalker lipsync face overlay (220×220, bottom-left) via `ai_modules/lipsync.py`

---

## TTS & Subtitles

- `ai_modules/tts_engine.py` — one audio file per scene
- Provider chain: ModelsLab → Gemini → OpenAI → ElevenLabs → gTTS → stub
- Urdu routes to ElevenLabs first (best multilingual quality)
- `ai_modules/subtitle_engine.py` — bilingual SRT (Urdu translation via Gemini, fallback stub)
- Audio duration measured with `mutagen` for accurate subtitle timestamps and Kling clip sizing

---

## Database Models (`core/models.py`)

- `StoryRequest` — main record; holds all user inputs, `story_json`, `status`, `video_path`
- `StoryScene` — one row per scene; holds `image_path`, `audio_path`, `video_clip_path`, `decision`
- `UserDecision` — tracks which A/B choice was made at each decision scene
- `EmailVerification` — 6-digit OTP, 10-minute expiry

Scene cache guard pattern (in `pipeline.py`): always check `file.exists() and file.stat().st_size > 10_000` before skipping regeneration — DB path alone is not enough.

---

## Deploy Flow

```bash
# On local machine
git push origin develop

# On server (via SSH / paramiko script)
cd /root/MoralVerse-FYP && git pull origin develop
systemctl restart gunicorn
systemctl is-active gunicorn   # verify: should print "active"
```

Gunicorn logs: `/root/MoralVerse-FYP/logs/gunicorn-{access,error}.log`

---

## Prompt Engineering Rules (Critical)

When modifying any LLM prompt in `llm_engine.py`:

1. **Pass 2 emotion table** — always translate emotions to concrete body language. Never pass abstract words like "felt scared" to FLUX. The emotion → body-language table lives in the Pass 2 system prompt.
2. **NARRATION MATCH** — the assembled image prompt must make the correct scene action visually obvious without reading the scene text.
3. **Pass 3 char limit** — Kling video prompts must be ≤ 460 chars and end with `"Character stays centered, full body in frame."`. Code in the Pass 3 parse loop enforces both (`_REQUIRED_ENDING`, `_MAX_VP_CHARS` constants at module level).
4. **Token order** — `action` must be first in `_assemble_scene_prompt()` so FLUX prioritises the action over the character portrait.
5. **`_validate_and_patch_prompt()`** — patches missing content words (threshold ≥ 1). Do not raise the threshold without a strong reason.
6. **`_STYLE_TAG`** and the character anchor are always injected by Python — never put style or character identity into LLM instructions for Pass 2.

---

## Known Constraints

- Supabase **session pooler** (port 5432) must be used — direct connection (`db.moyaltjrlsjdzyaatfto.supabase.co`) has IPv6-only DNS and does not work from the server.
- SadTalker lipsync tries ModelsLab v6 then v5 endpoints; may still fail — skips gracefully.
- MoviePy 2.x API: use `subclipped()`, `with_effects()`, `with_multiply_volume()` — old 1.x names (`subclip`, `volumex`) raise `AttributeError`.
- `SUBTITLE_FONT` env var overrides font search; Urdu requires NotoNaskhArabic or similar — Latin fonts do not render Urdu script.
