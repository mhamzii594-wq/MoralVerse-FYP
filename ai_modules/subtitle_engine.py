"""
Subtitle generation engine for Urdu and English subtitles.

This module:
1. Translates scene text to Urdu (using Gemini/OpenAI)
2. Creates SRT subtitle files with both languages
3. Returns the path to the SRT file
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Tuple

from django.conf import settings

logger = logging.getLogger(__name__)


def _gemini_translate(text: str, target_lang: str) -> str:
    """Translate text using the google.genai client (same SDK as rest of codebase)."""
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = os.getenv("GEMINI_TRANSLATION_MODEL", "gemini-1.5-flash")
    lang_name = "Urdu" if target_lang == "ur" else "English"
    prompt = (
        f"Translate the following text to {lang_name}. "
        f"Return ONLY the {lang_name} translation, no explanations:\n\n{text}"
    )
    resp = client.models.generate_content(model=model, contents=prompt)
    result = (resp.text or "").strip()
    if not result:
        raise ValueError("Empty translation response from Gemini")
    return result


def translate_to_urdu(text: str) -> str:
    """
    Translate English text to Urdu using available translation API.

    Priority:
    1. Gemini (if GEMINI_API_KEY available)
    2. OpenAI (if OPENAI_API_KEY available)
    3. Stub mode (returns transliterated placeholder)

    Args:
        text: English text to translate

    Returns:
        Urdu translation
    """
    # Try Gemini first (uses new google.genai SDK, same as rest of codebase)
    if os.getenv("GEMINI_API_KEY"):
        try:
            return _gemini_translate(text, "ur")
        except Exception as e:
            logger.warning(f"Gemini translation failed: {e}")
    
    # Try OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a translator. Translate English text to Urdu. Return ONLY the Urdu translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,
            )
            urdu_text = response.choices[0].message.content.strip()
            return urdu_text
        except Exception as e:
            logger.warning(f"OpenAI translation failed: {e}")
    
    # Stub mode: return transliterated placeholder
    logger.info("Translation API not available, using stub mode")
    return f"[اردو ترجمہ: {text}]"


def translate_to_english(text: str) -> str:
    """
    Translate Urdu text to English.

    Priority:
    1. Gemini (if GEMINI_API_KEY available)
    2. OpenAI (if OPENAI_API_KEY available)
    3. Stub mode (returns placeholder)
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            return _gemini_translate(text, "en")
        except Exception as e:
            logger.warning(f"Gemini translation to English failed: {e}")

    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a translator. Translate Urdu to English. Return ONLY the English translation.",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI translation to English failed: {e}")

    logger.info("English translation API not available, using stub mode")
    return f"[English translation: {text}]"


def estimate_duration(text: str, words_per_minute: int = 0, language: str = "en") -> float:
    """
    Estimate audio duration based on text length.

    Args:
        text: Text to estimate duration for
        words_per_minute: Override WPM (0 = use language default)
        language: 'en' (150 WPM) or 'ur' (100 WPM — Urdu TTS speaks slower)

    Returns:
        Duration in seconds
    """
    if words_per_minute <= 0:
        words_per_minute = 100 if language == "ur" else 150
    word_count = len(text.split())
    duration_seconds = (word_count / words_per_minute) * 60
    return max(2.0, min(10.0, duration_seconds))


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into SRT timestamp format: HH:MM:SS,mmm
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_file(
    scenes: List[Tuple[str, str]],
    output_path: Path,
    durations: List[float] | None = None,
    language: str = "en",
) -> None:
    """
    Create an SRT subtitle file with Urdu and English subtitles.
    
    Args:
        scenes: List of tuples (english_text, urdu_text)
        output_path: Path to save SRT file
        durations: Optional list of actual audio durations
    """
    current_time = 0.0
    subtitle_index = 1
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (english_text, urdu_text) in enumerate(scenes):
            duration = durations[i] if durations and i < len(durations) else estimate_duration(english_text, language=language)

            start_time = format_timestamp(current_time)
            end_time = format_timestamp(current_time + duration)

            # Show only one language per block — mixing both causes double-text rendering.
            # For Urdu stories show Urdu; for English stories show English.
            subtitle_text = urdu_text if language == "ur" else english_text

            f.write(f"{subtitle_index}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{subtitle_text}\n")
            f.write("\n")

            current_time += duration
            subtitle_index += 1


def generate_subtitles(scene_texts: List[str], source_language: str = "en", durations: List[float] | None = None) -> str:
    """
    Generate SRT subtitle file with Urdu and English subtitles for all scenes.
    
    Args:
        scene_texts: List of scene text strings
        source_language: 'en' or 'ur'
        durations: Actual durations of the audio clips for perfect sync
    """
    import time
    import hashlib
    
    subtitles_dir = Path(settings.MEDIA_ROOT) / "subtitles"
    subtitles_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = int(time.time() * 1000)
    text_hash = hashlib.md5("".join(scene_texts).encode()).hexdigest()[:8]
    output_path = subtitles_dir / f"story_{timestamp}_{text_hash}.srt"
    
    scenes_with_translations: List[Tuple[str, str]] = []
    if source_language == "ur":
        for scene_text in scene_texts:
            english_text = translate_to_english(scene_text)
            scenes_with_translations.append((english_text, scene_text))
    else:
        for scene_text in scene_texts:
            urdu_text = translate_to_urdu(scene_text)
            scenes_with_translations.append((scene_text, urdu_text))
    
    create_srt_file(scenes_with_translations, output_path, durations, language=source_language)
    logger.info(f"Generated subtitles: {output_path}")
    
    return f"subtitles/{output_path.name}"

