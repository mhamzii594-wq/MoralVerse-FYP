"""
Text-to-Speech (TTS) engine.

Supported providers:
- Gemini TTS (when GEMINI_API_KEY is present and `google-genai` is installed)
- OpenAI TTS (when OPENAI_API_KEY is present)
- ElevenLabs (when ELEVENLABS_API_KEY is present)

When no provider/key is available, a stub mode is used that saves a
plain-text file describing the would-be audio output.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Literal

from django.conf import settings

logger = logging.getLogger(__name__)

def log_api_error(msg):
    from pathlib import Path
    import time
    debug_log = Path(settings.BASE_DIR) / "logs" / "api_debug.log"
    debug_log.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_log, "a", encoding="utf-8") as f:
        f.write(f"[{time.ctime()}] {msg}\n")

TTSProvider = Literal["gemini", "openai", "elevenlabs", "coqui", "modelslab", "stub"]


def _detect_provider(override: str | None = None) -> TTSProvider:
    explicit = (override or os.getenv("TTS_PROVIDER") or "").strip().lower()
    if explicit in {"gemini", "openai", "elevenlabs", "coqui", "modelslab", "stub"}:
        return explicit  # type: ignore[return-value]

    if os.getenv("MODELSLAB_API_KEY"):
        return "modelslab"

    if os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai  # noqa: F401
            return "gemini"
        except Exception:
            pass

    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    # Only use ElevenLabs if the dependency is actually installed.
    if os.getenv("ELEVENLABS_API_KEY"):
        try:
            import elevenlabs  # noqa: F401
            return "elevenlabs"
        except Exception:
            pass
    # Coqui TTS works offline, so we can always try it
    try:
        import TTS
        return "coqui"
    except ImportError:
        pass
    return "stub"


def generate_audio(
    scene_text: str,
    voice: str = "child_friendly",
    language: str = "en",
    provider_override: str | None = None,
) -> str:
    """
    Generate speech audio for the supplied text and return a file path
    within MEDIA_ROOT (e.g. "audio/scene_123.wav").

    Args:
        scene_text: Text to convert to speech
        voice: Voice identifier (default: "child_friendly")
        language: 'en' for English, 'ur' for Urdu (used to pick a better local model if needed)
            - For OpenAI: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
            - For ElevenLabs: voice name or ID
            - For Coqui: model name or "tts_models/en/ljspeech/tacotron2-DDC"

    Returns:
        Relative path to audio file (e.g., "audio/scene_123.wav")
    """
    import time
    import hashlib
    
    provider = _detect_provider(provider_override)
    
    # Global Stub Mode (Credit Protection)
    if os.getenv("USE_STUB_MODE", "false").lower() == "true":
        provider = "stub"
        
    # Urdu routing: try ElevenLabs first if available (best multilingual support),
    # otherwise let Gemini TTS try (it supports Urdu), gTTS is the reliable free fallback.
    if language == "ur" and provider == "gemini" and os.getenv("ELEVENLABS_API_KEY"):
        try:
            import elevenlabs  # noqa: F401
            provider = "elevenlabs"
            logger.info("Urdu detected: routing to ElevenLabs (multilingual) instead of Gemini.")
        except ImportError:
            pass

    audio_dir = Path(settings.MEDIA_ROOT) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    timestamp = int(time.time() * 1000)
    text_hash = hashlib.md5(scene_text.encode()).hexdigest()[:8]
    # Default extension; will be updated by provider if needed
    ext = "wav" if provider in {"gemini", "openai"} else "mp3"
    output_path = audio_dir / f"scene_{timestamp}_{text_hash}.{ext}"
    
    if provider == "modelslab":
        try:
            import requests as _req, time as _time
            api_key = os.getenv("MODELSLAB_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("MODELSLAB_API_KEY not set")

            # ModelsLab TTS only reliably supports English; route Urdu to next provider
            if language == "ur":
                raise RuntimeError("ModelsLab TTS does not support Urdu; routing to Gemini")

            voice_id = os.getenv("MODELSLAB_TTS_VOICE_ID", "madison")  # clear female voice
            payload = {
                "key": api_key,
                "prompt": scene_text,
                "voice_id": voice_id,
                "language": "american english",
                "speed": 1,
                "emotion": True,   # expressive narration
            }
            resp = _req.post(
                "https://modelslab.com/api/v6/voice/text_to_speech",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "error":
                raise RuntimeError(f"ModelsLab TTS error: {data.get('message', data)}")

            audio_url = None
            if data.get("status") == "success":
                out = data.get("output")
                audio_url = (out[0] if isinstance(out, list) else out) or ""
            elif data.get("status") == "processing":
                fetch_id = data.get("id")
                eta = int(data.get("eta", 10))
                logger.info("ModelsLab TTS processing (id=%s, eta=%ss)...", fetch_id, eta)
                _time.sleep(min(eta, 15))
                for _ in range(8):
                    fetch_resp = _req.post(
                        f"https://modelslab.com/api/v6/voice/fetch/{fetch_id}",
                        json={"key": api_key},
                        timeout=30,
                    )
                    fdata = fetch_resp.json()
                    if fdata.get("status") == "success":
                        out = fdata.get("output")
                        audio_url = (out[0] if isinstance(out, list) else out) or ""
                        break
                    if fdata.get("status") == "error":
                        raise RuntimeError(f"ModelsLab TTS poll error: {fdata.get('message', fdata)}")
                    _time.sleep(5)

            if not audio_url:
                raise RuntimeError("ModelsLab TTS returned no audio URL")

            dl = _req.get(audio_url, timeout=60)
            dl.raise_for_status()
            if len(dl.content) < 500:
                raise RuntimeError(f"ModelsLab TTS audio too small ({len(dl.content)} bytes)")

            output_path = audio_dir / f"scene_{timestamp}_{text_hash}_ml.mp3"
            output_path.write_bytes(dl.content)
            logger.info("ModelsLab TTS saved: %s (%d bytes)", output_path.name, len(dl.content))
            return f"audio/{output_path.name}"

        except Exception as e:
            logger.warning("ModelsLab TTS failed: %s. Falling back to Gemini...", e)
            log_api_error(f"ModelsLab TTS failed: {e}")
            provider = "gemini"

    if provider == "gemini":
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            )
            # Use .wav for Gemini multimodal audio output (usually PCM/WAV)
            output_path = audio_dir / f"scene_{timestamp}_{text_hash}.wav"

            # Prefer gemini-2.0-flash which has better multimodal support
            gemini_tts_model = (os.getenv("GEMINI_TTS_MODEL") or "gemini-2.0-flash").strip()
            # Ensure model name is correctly prefixed if needed by the client library version
            if not gemini_tts_model.startswith("models/"):
                actual_model = f"models/{gemini_tts_model}"
            else:
                actual_model = gemini_tts_model

            gemini_voice = (
                voice
                if voice != "child_friendly"
                else (os.getenv("GEMINI_TTS_VOICE") or "Puck")
            )

            import threading
            import queue

            def call_gemini(q, text, model, voice_name):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=text,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=voice_name
                                    )
                                )
                            ),
                        ),
                    )
                    q.put(response)
                except Exception as e:
                    q.put(e)

            res_queue = queue.Queue()
            thread = threading.Thread(target=call_gemini, args=(res_queue, scene_text, actual_model, gemini_voice))
            thread.daemon = True
            thread.start()

            try:
                # Wait for 60 seconds for Gemini to respond (multimodal can be slow)
                response = res_queue.get(timeout=60)
                if isinstance(response, Exception):
                    raise response
            except queue.Empty:
                logger.error("Gemini TTS timed out after 60s. High traffic or slow connection suspected.")
                raise RuntimeError("Gemini TTS timed out.")

            # Collect audio bytes
            audio_bytes = bytearray()
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        audio_bytes.extend(part.inline_data.data)
                    elif hasattr(part, 'audio'):
                        audio_bytes.extend(part.audio)

            if not audio_bytes:
                raise RuntimeError("Gemini TTS returned no audio data.")

            with open(output_path, "wb") as f:
                f.write(bytes(audio_bytes))
            
            # If the file is actually an MP3 but we named it WAV, or vice versa, 
            # ffmpeg can usually handle it if we are lucky, but let's be consistent.
            # Most GenAI audio responses are WAV/PCM.
            logger.info(f"Generated Gemini TTS audio: {output_path} (size: {len(audio_bytes)} bytes)")
            return f"audio/{output_path.name}"
        except Exception as e:
            error_msg = f"Gemini TTS failed: {e}. Falling back to Google Translate TTS."
            logger.warning(error_msg)
            log_api_error(error_msg)
            provider = "fallback_google"
            # Switch to mp3 for google translate
            output_path = output_path.with_suffix(".mp3")
            try:
                import requests
                
                # Split text into chunks of 200 characters (limit for public Google TTS)
                def split_text(text, limit=200):
                    import re
                    # Split by sentences if possible
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    chunks = []
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < limit:
                            current_chunk += (" " if current_chunk else "") + sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            # Handle case where a single sentence > limit
                            while len(sentence) > limit:
                                chunks.append(sentence[:limit])
                                sentence = sentence[limit:]
                            current_chunk = sentence
                    if current_chunk:
                        chunks.append(current_chunk)
                    return chunks

                text_chunks = split_text(scene_text)
                audio_data = bytearray()
                
                url = "https://translate.google.com/translate_tts"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                for chunk in text_chunks:
                    params = {
                        "ie": "UTF-8",
                        "q": chunk,
                        "tl": language,
                        "client": "tw-ob"
                    }
                    res = requests.get(url, params=params, headers=headers, timeout=10)
                    logger.info(f"Google Translate TTS request for language '{language}' (chunk length: {len(chunk)})")
                    if res.status_code == 200:
                        audio_data.extend(res.content)
                    else:
                        error_msg = f"Google Translate TTS chunk failed with status {res.status_code} for language '{language}': {res.text}"
                        logger.warning(error_msg)
                        log_api_error(error_msg)

                if audio_data:
                    with open(output_path, "wb") as f:
                        f.write(bytes(audio_data))
                    logger.info(f"Generated Google Translate TTS audio (chunked): {output_path}")
                    return f"audio/{output_path.name}"
            except Exception as e2:
                logger.error(f"Google Translate TTS fallback also failed: {e2}")
            # Both Gemini and Google Translate failed — continue to remaining providers
            provider = _detect_provider()
            if provider == "gemini":
                # Gemini was the only detected provider; nothing left to try
                provider = "stub"

    if provider == "openai":
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            output_path = audio_dir / f"scene_{timestamp}_{text_hash}.wav"
            with open(output_path, "wb") as f:
                audio = client.audio.speech.create(
                    model="tts-1",
                    voice=voice if voice != "child_friendly" else "nova",  # "nova" is more child-friendly
                    input=scene_text,
                )
                f.write(audio.read())
            return f"audio/{output_path.name}"
        except Exception as e:
            logger.warning(f"OpenAI TTS failed: {e}")
            # Fall through to next provider
            pass

    if provider == "elevenlabs":
        try:
            from elevenlabs import ElevenLabs, save

            client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
            output_path = audio_dir / f"scene_{timestamp}_{text_hash}.mp3"
            # Use child-friendly voice if available
            # Note: voice availability for Urdu varies by ElevenLabs library.
            voice_id = voice if voice != "child_friendly" else ("Rachel" if language == "en" else "Charlie")
            audio = client.generate(text=scene_text, voice=voice_id, model="eleven_multilingual_v2")
            save(audio, str(output_path))
            return f"audio/{output_path.name}"
        except Exception as e:
            logger.warning(f"ElevenLabs TTS failed: {e}")
            # Fall through to next provider
            pass

    if provider == "coqui":
        try:
            from TTS.api import TTS
            
            # Initialize Coqui TTS
            # Default model is English; allow override for Urdu.
            model_name = (
                (os.getenv("COQUI_MODEL_UR") if language == "ur" else None)
                or os.getenv("COQUI_MODEL_EN")
                or "tts_models/en/ljspeech/tacotron2-DDC"
            )
            tts = TTS(model_name=model_name, progress_bar=False)
            
            output_path = audio_dir / f"scene_{timestamp}_{text_hash}.wav"
            tts.tts_to_file(text=scene_text, file_path=str(output_path))
            return f"audio/{output_path.name}"
        except ImportError:
            logger.warning("Coqui TTS not installed. Install with: pip install TTS")
        except Exception as e:
            logger.warning(f"Coqui TTS failed: {e}")
            # Fall through to stub
            pass

    # Final cleanup: ensure the file exists and has size
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        return f"audio/{output_path.name}"

    # Last Resort: gTTS (Google Text-to-Speech library) — supports English and Urdu
    try:
        from gtts import gTTS
        output_path = audio_dir / f"scene_{timestamp}_{text_hash}_gtts.mp3"
        lang_code = "ur" if language == "ur" else "en"
        tts_obj = gTTS(text=scene_text, lang=lang_code, slow=False)
        tts_obj.save(str(output_path))
        if output_path.stat().st_size > 500:
            logger.info("Generated audio via gTTS (%s): %s", lang_code, output_path.name)
            return f"audio/{output_path.name}"
    except Exception as e:
        logger.warning("gTTS fallback failed: %s", e)

    # Stub: create a short silent WAV so MoviePy/ffmpeg can read it.
    logger.warning(f"All TTS providers failed or returned empty data for '{language}'. Creating silent stub.")
    # This keeps the pipeline "fully functional" even without API quota/keys.
    word_count = len(scene_text.split())
    words_per_minute = 150
    estimated_seconds = (word_count / words_per_minute) * 60
    duration = max(2.0, min(7.0, float(estimated_seconds)))

    import wave
    import struct

    sample_rate = 16000
    n_frames = int(duration * sample_rate)
    output_path = audio_dir / f"scene_{timestamp}_{text_hash}_stub.wav"

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        silence_frame = struct.pack("<h", 0)
        wf.writeframes(silence_frame * n_frames)

    # Optional: write the would-be text alongside for debugging.
    try:
        output_path.with_suffix(".txt").write_text(
            f"TTS output here:\n\n{scene_text}",
            encoding="utf-8",
        )
    except Exception:
        pass

    return f"audio/{output_path.name}"


# Backward compatibility alias
def generate_tts(text: str, voice: str | None = None) -> str:
    """Backward compatibility wrapper for generate_audio."""
    return generate_audio(text, voice or "child_friendly")


