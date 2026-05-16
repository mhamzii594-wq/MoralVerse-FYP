"""
Talking-head lipsync clip generation via ModelsLab SadTalker.

Given an avatar portrait image and a narration audio file, this module
produces a short animated talking-head video whose mouth movements are
lip-synced to the audio.  The result is composited onto each Kling scene
clip by the cinematic assembly engine.

Tries v6 endpoint first; falls back to v5 if v6 returns an error response.
"""

import os
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# ModelsLab has two known SadTalker endpoints — try v6 first, fall back to v5
_SADTALKER_SUBMIT_V6 = "https://modelslab.com/api/v6/video/sad_talker"
_SADTALKER_SUBMIT_V5 = "https://modelslab.com/api/v5/live_portraits/sadtalker"
_SADTALKER_FETCH_V6  = "https://modelslab.com/api/v6/video/fetch"
_SADTALKER_FETCH_V5  = "https://modelslab.com/api/v5/live_portraits/fetch"

# Keep legacy names for backward compatibility
_SADTALKER_SUBMIT = _SADTALKER_SUBMIT_V6
_SADTALKER_FETCH  = _SADTALKER_FETCH_V6


def _modelslab_key() -> str:
    return os.getenv("MODELSLAB_API_KEY", "")


def _public_media_url(local_path: str) -> str:
    """Convert a local media file path to its public HTTPS URL (served by nginx)."""
    try:
        from django.conf import settings as dj_settings
        media_root = Path(dj_settings.MEDIA_ROOT).resolve()
        abs_path = Path(local_path).resolve()
        relative = abs_path.relative_to(media_root)
        site_url = getattr(dj_settings, "SITE_URL", "").rstrip("/") or "https://moralverse.dev"
        media_url = dj_settings.MEDIA_URL.rstrip("/")
        return f"{site_url}{media_url}/{relative}"
    except Exception as exc:
        raise RuntimeError(f"Could not build public media URL for {local_path}: {exc}") from exc


def generate_lipsync_clip(avatar_path: str, audio_path: str, output_path: str) -> str:
    """
    Generate a lip-synced talking-head clip using SadTalker.

    Args:
        avatar_path: Absolute local path to the avatar portrait image (JPG/PNG).
        audio_path:  Absolute local path to the narration audio file (MP3/WAV).
        output_path: Absolute local path for the resulting .mp4 clip.

    Returns:
        output_path on success.

    Raises:
        RuntimeError if the API key is missing or all attempts fail.
    """
    key = _modelslab_key()
    if not key:
        raise RuntimeError("MODELSLAB_API_KEY not set — cannot generate lipsync clip")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    avatar_url = _public_media_url(avatar_path)
    audio_url  = _public_media_url(audio_path)

    logger.info("lipsync [SadTalker]: avatar=%s audio=%s", avatar_url, audio_url)

    payload = {
        "key": key,
        "init_image": avatar_url,
        "init_audio": audio_url,
        "face_restore": "true",
        "size": 256,
        "still_mode": False,
        "webhook": None,
        "track_id": None,
    }

    # Try v6 first, fall back to v5 if v6 returns an error status
    fetch_url = _SADTALKER_FETCH_V6
    for submit_url, fetch_endpoint in [
        (_SADTALKER_SUBMIT_V6, _SADTALKER_FETCH_V6),
        (_SADTALKER_SUBMIT_V5, _SADTALKER_FETCH_V5),
    ]:
        logger.info("lipsync [SadTalker]: trying endpoint %s", submit_url)
        try:
            resp = requests.post(submit_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            logger.info("SadTalker submit response (%s): status=%s", submit_url, data.get("status"))
        except Exception as exc:
            logger.warning("SadTalker submit failed (%s): %s", submit_url, exc)
            continue

        if data.get("status") == "error":
            logger.warning("SadTalker %s returned error: %s — trying next endpoint", submit_url, data.get("message"))
            continue

        if data.get("status") == "success" and data.get("output"):
            return _download(data["output"][0], output_path)

        fetch_id = data.get("id") or data.get("fetch_id")
        if not fetch_id:
            logger.warning("SadTalker %s returned no fetch_id: %s", submit_url, data)
            continue

        fetch_url = fetch_endpoint
        return _poll(fetch_id, output_path, key, fetch_url=fetch_url)

    raise RuntimeError("SadTalker: all endpoints failed — lipsync unavailable")


def _poll(fetch_id: str, output_path: str, key: str, timeout: int = 240,
          fetch_url: str = _SADTALKER_FETCH_V6) -> str:
    deadline = time.time() + timeout
    payload = {"key": key, "request_id": fetch_id}

    while time.time() < deadline:
        time.sleep(15)
        resp = requests.post(fetch_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "")
        logger.info("SadTalker poll [%s] status: %s", fetch_id, status)

        if status == "success":
            url = (data.get("output") or [None])[0]
            if url:
                return _download(url, output_path)
            raise RuntimeError("SadTalker success but no output URL")

        if status == "error":
            raise RuntimeError(f"SadTalker generation error: {data.get('message')}")

    raise RuntimeError(f"SadTalker timed out after {timeout}s (fetch_id={fetch_id})")


def _download(url: str, output_path: str) -> str:
    logger.info("Downloading lipsync clip: %s -> %s", url, output_path)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(output_path) / 1e6
    logger.info("Lipsync clip downloaded: %s (%.1f MB)", output_path, size_mb)
    return output_path
