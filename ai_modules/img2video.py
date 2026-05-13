"""
Image-to-video clip generation for the cinematic pipeline.

Provider priority:
  1. ModelsLab (existing API key, cheap, async job-based)
  2. Stability AI SVD (existing key, fallback)

Returns a local .mp4 path on success, raises on failure.
"""

import os
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

MODELSLAB_API_KEY = os.getenv("MODELSLAB_API_KEY", "")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")

_MODELSLAB_SUBMIT = "https://modelslab.com/api/v6/video/img2video"
_MODELSLAB_FETCH  = "https://modelslab.com/api/v6/video/fetch"

_STABILITY_SUBMIT = "https://api.stability.ai/v2beta/image-to-video"
_STABILITY_FETCH  = "https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_clip(image_path: str, output_path: str, duration: int = 4) -> str:
    """
    Generate an animated video clip from a still image.

    Args:
        image_path: Absolute path to the source scene image (PNG/JPG).
        output_path: Where to save the resulting .mp4.
        duration: Desired clip length in seconds (3–6 recommended).

    Returns:
        output_path on success.

    Raises:
        RuntimeError if all providers fail.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if MODELSLAB_API_KEY:
        try:
            return _modelslab(image_path, output_path, duration)
        except Exception as exc:
            logger.warning("ModelsLab img2video failed: %s — trying Stability", exc)

    if STABILITY_API_KEY:
        try:
            return _stability_svd(image_path, output_path)
        except Exception as exc:
            logger.warning("Stability SVD img2video failed: %s", exc)

    raise RuntimeError("All img2video providers failed — check API keys and logs.")


# ---------------------------------------------------------------------------
# ModelsLab provider
# ---------------------------------------------------------------------------

def _modelslab(image_path: str, output_path: str, duration: int) -> str:
    logger.info("img2video: submitting to ModelsLab — %s", image_path)

    # ModelsLab requires a publicly accessible image URL or base64.
    # We upload via their file hosting endpoint first.
    image_url = _modelslab_upload(image_path)

    payload = {
        "key": MODELSLAB_API_KEY,
        "init_image": image_url,
        "motion_bucket_id": 40,        # 1-255; higher = more motion
        "noise_aug_strength": 0.02,    # subtle noise for realism
        "fps": 8,
        "num_frames": duration * 8,    # 8 fps × duration seconds
        "width": 512,
        "height": 512,
        "webhook": None,
        "track_id": None,
    }

    resp = requests.post(_MODELSLAB_SUBMIT, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"ModelsLab submit error: {data.get('message')}")

    # Immediate result (rare but possible)
    if data.get("status") == "success" and data.get("output"):
        return _download(data["output"][0], output_path)

    # Async job — poll with fetch_id
    fetch_id = data.get("id") or data.get("fetch_id")
    if not fetch_id:
        raise RuntimeError(f"ModelsLab returned no fetch_id: {data}")

    return _modelslab_poll(fetch_id, output_path)


def _modelslab_upload(image_path: str) -> str:
    """Upload a local image to ModelsLab and return its hosted URL."""
    upload_url = "https://modelslab.com/api/v6/realtime/upload"
    with open(image_path, "rb") as f:
        resp = requests.post(
            upload_url,
            data={"key": MODELSLAB_API_KEY},
            files={"file": (Path(image_path).name, f, "image/png")},
            timeout=30,
        )
    resp.raise_for_status()
    data = resp.json()
    url = data.get("url") or (data.get("output") or [None])[0]
    if not url:
        raise RuntimeError(f"ModelsLab upload returned no URL: {data}")
    return url


def _modelslab_poll(fetch_id: str, output_path: str, timeout: int = 180) -> str:
    """Poll ModelsLab fetch endpoint until the clip is ready."""
    deadline = time.time() + timeout
    payload = {"key": MODELSLAB_API_KEY, "request_id": fetch_id}

    while time.time() < deadline:
        time.sleep(10)
        resp = requests.post(_MODELSLAB_FETCH, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "")
        logger.debug("ModelsLab poll status: %s", status)

        if status == "success":
            url = (data.get("output") or [None])[0]
            if url:
                return _download(url, output_path)
            raise RuntimeError("ModelsLab success but no output URL")

        if status == "error":
            raise RuntimeError(f"ModelsLab generation error: {data.get('message')}")

        # status == "processing" or "queued" — keep waiting

    raise RuntimeError(f"ModelsLab img2video timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Stability AI SVD provider
# ---------------------------------------------------------------------------

def _stability_svd(image_path: str, output_path: str) -> str:
    logger.info("img2video: submitting to Stability SVD — %s", image_path)

    with open(image_path, "rb") as f:
        resp = requests.post(
            _STABILITY_SUBMIT,
            headers={"authorization": f"Bearer {STABILITY_API_KEY}"},
            files={"image": (Path(image_path).name, f, "image/png")},
            data={"seed": 0, "cfg_scale": 1.8, "motion_bucket_id": 40},
            timeout=30,
        )
    resp.raise_for_status()
    generation_id = resp.json().get("id")
    if not generation_id:
        raise RuntimeError(f"Stability SVD returned no generation id: {resp.text}")

    return _stability_poll(generation_id, output_path)


def _stability_poll(generation_id: str, output_path: str, timeout: int = 180) -> str:
    deadline = time.time() + timeout
    url = _STABILITY_FETCH.format(generation_id=generation_id)

    while time.time() < deadline:
        time.sleep(10)
        resp = requests.get(
            url,
            headers={"authorization": f"Bearer {STABILITY_API_KEY}", "accept": "video/*"},
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
# Shared helper
# ---------------------------------------------------------------------------

def _download(url: str, output_path: str) -> str:
    logger.info("Downloading clip: %s → %s", url, output_path)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Clip downloaded: %s (%.1f MB)", output_path, os.path.getsize(output_path) / 1e6)
    return output_path
