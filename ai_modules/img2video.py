"""
Image-to-video clip generation for the cinematic pipeline.

Provider priority:
  1. Wan 2.1 on Modal.com  (MODAL_WAN_ENDPOINT)  — best open-source quality, 480p, 16fps
  2. LTX-Video 2.3 on Modal.com (MODAL_LTX_ENDPOINT) — fast fallback
  3. ModelsLab Kling v2.1 i2v  — best commercial, 1080p, prompt-guided
  4. ModelsLab basic img2video  — cheap fallback
  5. Stability AI SVD           — last resort

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


def _modal_ltx_endpoint() -> str:
    return os.getenv("MODAL_LTX_ENDPOINT", "").strip()


def _modal_wan_endpoint() -> str:
    return os.getenv("MODAL_WAN_ENDPOINT", "").strip()


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

    # Clip length follows the scene's narration so video and audio finish together.
    effective_duration = audio_duration if audio_duration else duration

    # 1. Wan 2.1 on Modal (best open-source quality)
    if _modal_wan_endpoint():
        try:
            return _wan_modal(image_path, output_path, prompt, duration=effective_duration)
        except Exception as exc:
            logger.error("Wan 2.1 Modal failed: %s — trying LTX", exc)

    # 2. LTX-Video 2.3 on Modal (fast fallback)
    if _modal_ltx_endpoint():
        try:
            return _ltx_modal(image_path, output_path, prompt, duration=effective_duration)
        except Exception as exc:
            logger.error("LTX Modal failed: %s — trying Kling", exc)

    # 3. Kling v2.1 via ModelsLab
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

    # 5. Stability SVD last resort
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
# Wan 2.1 on Modal.com (primary provider)
# ---------------------------------------------------------------------------

def _wan_modal(image_path: str, output_path: str, prompt: str, duration: float = 5.0) -> str:
    """Generate a clip via Wan 2.1 running on Modal.com A10G GPU."""
    import base64, random
    endpoint = _modal_wan_endpoint()
    if not endpoint:
        raise RuntimeError("MODAL_WAN_ENDPOINT not set")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # Wan 2.1 runs at 16fps and requires (num_frames - 1) % 4 == 0. Round the clip length
    # UP to the next valid frame count so the video always covers the narration audio
    # (never ends mid-sentence). Clamped to 17..121 frames (~1-7.5s, model limit).
    import math
    raw = max(17, min(121, int(math.ceil(duration * 16))))
    num_frames = math.ceil((raw - 1) / 4) * 4 + 1
    num_frames = min(121, num_frames)

    logger.info("img2video [Wan 2.1 Modal]: %d frames (%.1fs) posting to %s", num_frames, duration, endpoint)
    resp = requests.post(
        endpoint,
        json={
            "image": image_b64,
            "prompt": prompt[:500],
            "negative_prompt": (
                "flat 2D, anime, cel-shaded, hand-drawn, sketch, "
                "worst quality, inconsistent motion, blurry, jittery, distorted, static, no movement"
            ),
            "num_frames": num_frames,
            "fps": 16,
            "seed": random.randint(0, 2**32 - 1),  # unique per scene
        },
        timeout=900,  # cold start ~2 min + 25-step generation ~7 min; must exceed Modal's runtime
    )
    resp.raise_for_status()
    data = resp.json()

    video_bytes = base64.b64decode(data["video"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(video_bytes)

    size_mb = len(video_bytes) / 1e6
    logger.info("Wan 2.1 Modal clip saved: %s (%.1f MB)", output_path, size_mb)
    return output_path


# ---------------------------------------------------------------------------
# LTX-Video 2.3 on Modal.com (fallback)
# ---------------------------------------------------------------------------

def _ltx_modal(image_path: str, output_path: str, prompt: str, duration: float = 5.0) -> str:
    """Generate a clip via LTX-Video 2.3 running on Modal.com A10G GPU."""
    import base64, random
    endpoint = _modal_ltx_endpoint()
    if not endpoint:
        raise RuntimeError("MODAL_LTX_ENDPOINT not set")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # LTX runs at 24fps and requires (num_frames - 1) % 8 == 0. Clip length follows the
    # scene narration, clamped to 49..193 frames (~2-8s) — LTX loses coherence on longer clips.
    fps = 24
    raw = max(49, min(193, int(round(duration * fps))))
    num_frames = ((raw - 1) // 8) * 8 + 1

    logger.info("img2video [LTX Modal]: %d frames (%.1fs @ %dfps) posting to %s", num_frames, duration, fps, endpoint)
    resp = requests.post(
        endpoint,
        json={
            "image": image_b64,
            "prompt": prompt[:500],
            "num_frames": num_frames,
            "fps": fps,
            "height": 768,
            "width": 1024,
            "num_inference_steps": 40,
            "guidance_scale": 3.5,
            "negative_prompt": (
                "flat 2D, anime, cel-shaded, hand-drawn, sketch, low resolution, "
                "warped face, extra limbs, morphing, worst quality, inconsistent motion, "
                "blurry, jittery, distorted"
            ),
            "seed": random.randint(0, 2**32 - 1),  # unique per scene
        },
        timeout=480,
    )
    resp.raise_for_status()
    data = resp.json()

    video_bytes = base64.b64decode(data["video"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(video_bytes)

    size_mb = len(video_bytes) / 1e6
    logger.info("LTX Modal clip saved: %s (%.1f MB)", output_path, size_mb)
    return output_path


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

    kling_duration = "10" if (duration and duration >= 6.0) else "5"

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

    # ModelsLab provides a pre-signed CDN URL in future_links — poll that with HEAD
    # until the video is ready (returns 200) then download.  This is more reliable
    # than the fetch_result endpoint which currently returns {"status":"error","message":""}
    # regardless of the job state.
    future_links = data.get("future_links") or []
    if future_links:
        return _kling_poll_cdn(future_links[0], output_path, timeout=300)

    # Fallback to the fetch_result URL approach (GET-based, no POST)
    fetch_result_url = data.get("fetch_result")
    fetch_id = data.get("id") or data.get("fetch_id")
    if not fetch_result_url and not fetch_id:
        raise RuntimeError(f"Kling v2.1 returned no fetch info: {data}")

    return _kling_poll(fetch_result_url, fetch_id, output_path, key, timeout=300)


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
        "fps": 16,
        "num_frames": min(duration * 16, 120),  # API caps at 120 frames max
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
# Kling v2.1 CDN poll — HEAD-poll the future_links URL until the clip is ready
# ---------------------------------------------------------------------------

def _kling_poll_cdn(cdn_url: str, output_path: str, timeout: int = 300) -> str:
    """
    Poll the pre-signed CDN URL from future_links[] until the video file is ready.

    The CDN returns HTTP 200 with Content-Length: 0 as a placeholder while the
    generation is still running. Only download when Content-Length > 0 (actual video).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(15)
        try:
            resp = requests.head(cdn_url, timeout=15, allow_redirects=True)
            content_length = int(resp.headers.get("Content-Length", 0))
            logger.info(
                "Kling CDN poll: HTTP %s  bytes=%d  url=%s",
                resp.status_code, content_length, cdn_url,
            )
            if resp.status_code == 200 and content_length > 100_000:
                return _download(cdn_url, output_path)
        except Exception as exc:
            logger.debug("Kling CDN HEAD check error: %s", exc)

    raise RuntimeError(f"Kling clip not ready at CDN after {timeout}s — url={cdn_url}")


# ---------------------------------------------------------------------------
# Kling v2.1 GET-based poll (v7 fetch endpoint only accepts GET, not POST)
# ---------------------------------------------------------------------------

def _kling_poll(
    fetch_result_url: Optional[str],
    fetch_id: Optional[str],
    output_path: str,
    key: str,
    timeout: int = 300,
) -> str:
    """Poll the Kling v2.1 fetch endpoint via GET until the clip is ready."""
    # Build the URL: prefer the explicit fetch_result returned by the submit API
    # (already has the ID in the path); fall back to constructing it from base + id.
    if fetch_result_url:
        url = fetch_result_url
    else:
        url = f"{_KLING_V21_FETCH}/{fetch_id}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(15)
        resp = requests.get(url, params={"key": key}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "")
        logger.info("Kling v2.1 poll [%s] status: %s", fetch_id or url, status)

        if status == "success":
            clip_url = (data.get("output") or [None])[0]
            if clip_url:
                return _download(clip_url, output_path)
            raise RuntimeError("Kling v2.1 success but no output URL in response")

        if status == "error":
            raise RuntimeError(f"Kling v2.1 generation error: {data.get('message')}")

        # "processing" or "queued" — keep waiting

    raise RuntimeError(f"Kling v2.1 timed out after {timeout}s (id={fetch_id})")


# ---------------------------------------------------------------------------
# ModelsLab v6 POST-based poll helper (basic img2video uses POST fetch)
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
