"""
Image-to-video clip generation for the cinematic pipeline.

Provider priority:
  1. ModelsLab Kling v2.1 i2v (fast, 1080p, character-consistent animation)
  2. ModelsLab basic img2video (v6, cheap fallback)
  3. Stability AI SVD (last-resort fallback)

Returns a local .mp4 path on success, raises on failure.
"""

import os
import time
import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Kling v2.1 via ModelsLab video-fusion API (v7)
_KLING_V21_SUBMIT = "https://modelslab.com/api/v7/video-fusion/image-to-video"
_KLING_V21_FETCH  = "https://modelslab.com/api/v7/video-fusion/fetch"

# ModelsLab basic img2video (v6, fallback)
_MODELSLAB_SUBMIT = "https://modelslab.com/api/v6/video/img2video"
_MODELSLAB_FETCH  = "https://modelslab.com/api/v6/video/fetch"

_STABILITY_SUBMIT = "https://api.stability.ai/v2beta/image-to-video"
_STABILITY_FETCH  = "https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"


def _modelslab_key() -> str:
    # Read lazily so Django's load_dotenv() has already run
    return os.getenv("MODELSLAB_API_KEY", "")


def _stability_key() -> str:
    return os.getenv("STABILITY_API_KEY", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_clip(
    image_path: str,
    output_path: str,
    duration: int = 5,
    prompt: str = "",
    audio_duration: Optional[float] = None,
) -> str:
    """
    Generate an animated video clip from a still image.

    Args:
        image_path: Absolute local path to the source scene image (PNG/JPG).
        output_path: Absolute local path for the resulting .mp4.
        duration: Fallback clip length in seconds when audio_duration is not given.
        prompt: Scene description to guide the animation.
        audio_duration: Actual narration audio length in seconds. When provided,
            selects Kling clip length automatically: >= 6s → 10s clip, else → 5s clip.

    Returns:
        output_path on success.

    Raises:
        RuntimeError if all providers fail.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not prompt:
        prompt = (
            "Animate this children's story scene with gentle cinematic motion, "
            "smooth camera movement, vivid colors, characters moving naturally"
        )

    # Choose clip length: match audio when available so Kling generates enough content
    effective_duration = duration
    if audio_duration is not None:
        effective_duration = 10 if audio_duration >= 6.0 else 5

    ml_key = _modelslab_key()
    if ml_key:
        try:
            return _kling_v21(image_path, output_path, effective_duration, prompt, ml_key)
        except Exception as exc:
            logger.error("Kling v2.1 failed: %s — trying ModelsLab basic", exc)

        try:
            return _modelslab_basic(image_path, output_path, effective_duration, ml_key)
        except Exception as exc:
            logger.error("ModelsLab basic img2video failed: %s — trying Stability", exc)
    else:
        logger.error("MODELSLAB_API_KEY not set — skipping ModelsLab providers")

    st_key = _stability_key()
    if st_key:
        try:
            return _stability_svd(image_path, output_path, st_key)
        except Exception as exc:
            logger.error("Stability SVD failed: %s", exc)
    else:
        logger.error("STABILITY_API_KEY not set — skipping Stability SVD")

    raise RuntimeError("All img2video providers failed — check API keys and logs.")


# ---------------------------------------------------------------------------
# URL helper — use our own publicly-served media instead of upload endpoint
# ---------------------------------------------------------------------------

def _public_image_url(image_path: str) -> str:
    """
    Convert a local absolute media path to the public HTTPS URL served by nginx.

    e.g. /root/MoralVerse-FYP/media/images/foo.png
         → https://moralverse.dev/media/images/foo.png
    """
    try:
        from django.conf import settings as dj_settings
        media_root = Path(dj_settings.MEDIA_ROOT).resolve()
        abs_path = Path(image_path).resolve()
        relative = abs_path.relative_to(media_root)
        site_url = getattr(dj_settings, "SITE_URL", "").rstrip("/")
        if not site_url:
            site_url = "https://moralverse.dev"
        media_url = dj_settings.MEDIA_URL.rstrip("/")
        url = f"{site_url}{media_url}/{relative}"
        logger.debug("Public image URL: %s", url)
        return url
    except Exception as exc:
        raise RuntimeError(f"Could not build public image URL for {image_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Kling v2.1 provider (primary)
# ---------------------------------------------------------------------------

def _kling_v21(image_path: str, output_path: str, duration: int, prompt: str, key: str) -> str:
    image_url = _public_image_url(image_path)
    logger.info("img2video [Kling v2.1]: %s", image_url)

    kling_duration = "10" if duration >= 8 else "5"

    payload = {
        "key": key,
        "model_id": "kling-v2-1-i2v",
        "init_image": image_url,
        "prompt": prompt[:500],
        "duration": kling_duration,
    }

    resp = requests.post(_KLING_V21_SUBMIT, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    logger.info("Kling v2.1 submit response: %s", data)

    if data.get("status") == "error":
        raise RuntimeError(f"Kling v2.1 submit error: {data.get('message')}")

    if data.get("status") == "success" and data.get("output"):
        return _download(data["output"][0], output_path)

    fetch_id = data.get("id") or data.get("fetch_id")
    if not fetch_id:
        raise RuntimeError(f"Kling v2.1 returned no fetch_id: {data}")

    return _poll(fetch_id, output_path, fetch_url=_KLING_V21_FETCH, key=key, timeout=300)


# ---------------------------------------------------------------------------
# ModelsLab basic img2video provider (fallback)
# ---------------------------------------------------------------------------

def _modelslab_basic(image_path: str, output_path: str, duration: int, key: str) -> str:
    image_url = _public_image_url(image_path)
    logger.info("img2video [ModelsLab basic]: %s", image_url)

    payload = {
        "key": key,
        "init_image": image_url,
        "motion_bucket_id": 40,
        "noise_aug_strength": 0.02,
        "fps": 8,
        "num_frames": duration * 8,
        "width": 768,
        "height": 768,
        "webhook": None,
        "track_id": None,
    }

    resp = requests.post(_MODELSLAB_SUBMIT, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    logger.info("ModelsLab basic submit response: %s", data)

    if data.get("status") == "error":
        raise RuntimeError(f"ModelsLab submit error: {data.get('message')}")

    if data.get("status") == "success" and data.get("output"):
        return _download(data["output"][0], output_path)

    fetch_id = data.get("id") or data.get("fetch_id")
    if not fetch_id:
        raise RuntimeError(f"ModelsLab returned no fetch_id: {data}")

    return _poll(fetch_id, output_path, fetch_url=_MODELSLAB_FETCH, key=key, timeout=180)


# ---------------------------------------------------------------------------
# ModelsLab poll helper
# ---------------------------------------------------------------------------

def _poll(fetch_id: str, output_path: str, fetch_url: str, key: str, timeout: int = 180) -> str:
    deadline = time.time() + timeout
    payload = {"key": key, "request_id": fetch_id}

    while time.time() < deadline:
        time.sleep(15)
        resp = requests.post(fetch_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "")
        logger.info("ModelsLab poll [%s] status: %s", fetch_id, status)

        if status == "success":
            url = (data.get("output") or [None])[0]
            if url:
                return _download(url, output_path)
            raise RuntimeError("ModelsLab success but no output URL")

        if status == "error":
            raise RuntimeError(f"ModelsLab generation error: {data.get('message')}")

    raise RuntimeError(f"ModelsLab img2video timed out after {timeout}s (fetch_id={fetch_id})")


# ---------------------------------------------------------------------------
# Stability AI SVD provider (last resort)
# ---------------------------------------------------------------------------

def _stability_svd(image_path: str, output_path: str, key: str) -> str:
    logger.info("img2video [Stability SVD]: %s", image_path)

    with open(image_path, "rb") as f:
        resp = requests.post(
            _STABILITY_SUBMIT,
            headers={"authorization": f"Bearer {key}"},
            files={"image": (Path(image_path).name, f, "image/png")},
            data={"seed": 0, "cfg_scale": 1.8, "motion_bucket_id": 40},
            timeout=30,
        )
    resp.raise_for_status()
    generation_id = resp.json().get("id")
    if not generation_id:
        raise RuntimeError(f"Stability SVD returned no generation id: {resp.text}")

    return _stability_poll(generation_id, output_path, key)


def _stability_poll(generation_id: str, output_path: str, key: str, timeout: int = 180) -> str:
    deadline = time.time() + timeout
    url = _STABILITY_FETCH.format(generation_id=generation_id)

    while time.time() < deadline:
        time.sleep(10)
        resp = requests.get(
            url,
            headers={"authorization": f"Bearer {key}", "accept": "video/*"},
            timeout=30,
        )
        if resp.status_code == 202:
            logger.debug("Stability SVD still processing...")
            continue
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        logger.info("Stability SVD clip saved: %s", output_path)
        return output_path

    raise RuntimeError(f"Stability SVD timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Shared download helper
# ---------------------------------------------------------------------------

def _download(url: str, output_path: str) -> str:
    logger.info("Downloading clip: %s → %s", url, output_path)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(output_path) / 1e6
    logger.info("Clip downloaded: %s (%.1f MB)", output_path, size_mb)
    return output_path
