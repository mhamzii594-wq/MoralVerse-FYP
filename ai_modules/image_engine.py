"""
Image generation engine for scene illustrations.

Supported options (via IMAGE_PROVIDER env var, future extensible):
- "openai"        -> OpenAI Images API (DALL-E) - uses OPENAI_API_KEY
- "gemini"        -> Google Imagen API (via google-genai) - uses GEMINI_API_KEY
- "stability"     -> Stability AI - uses IMAGE_API_KEY
- "flux" (future) -> Black Forest Labs FLUX - uses IMAGE_API_KEY
- "local_sd"      -> Local Stable Diffusion (GPU/CPU), requires diffusers+torch

NOTE: Imagen access must be enabled for your Google AI Studio key/project.

For the current 30% implementation, these integrations are outlined and
wrapped in try/except so the project remains runnable even without keys.
Stub mode always exists and returns a placeholder path.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import logging
logger = logging.getLogger(__name__)
import base64
from typing import Any, Dict
from django.conf import settings
from .runtime import get_render_preset

# Also log to a file for easy debugging
debug_log = Path(settings.BASE_DIR) / "logs" / "api_debug.log"
debug_log.parent.mkdir(parents=True, exist_ok=True)
def log_api_error(msg):
    with open(debug_log, "a", encoding="utf-8") as f:
        f.write(f"[{time.ctime()}] {msg}\n")


def _provider(override: str | None = None) -> str:
    """
    Resolve image provider.

    Priority:
    1) Explicit override (e.g. from UI/API)
    2) IMAGE_PROVIDER env var
    3) Default: "gemini"
    """
    if override:
        return override.lower()
    return (os.getenv("IMAGE_PROVIDER") or "gemini").lower()


def _image_dir() -> Path:
    path = Path(settings.MEDIA_ROOT) / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stub_image(prompt: str) -> str:
    """
    Stub implementation that saves a tiny placeholder PNG (1x1) so the UI can render
    an image even when real providers are unavailable.
    """
    folder = _image_dir()
    filename = "placeholder_image.png"
    file_path = folder / filename

    try:
        from PIL import Image

        # Create a simple valid image so OpenCV/MoviePy can always load it.
        img = Image.new("RGB", (512, 512), color=(240, 244, 255))
        img.save(file_path, format="PNG")
    except Exception:
        # As a last resort, fall back to a text file
        file_path = folder / "placeholder_image.txt"
        file_path.write_text(f"Image stub for prompt:\n{prompt}", encoding="utf-8")
        return "images/placeholder_image.txt"

    return f"images/{filename}"


def _modelslab_poll(api_key: str, prediction_id: str, prefix: str, max_attempts: int = 12) -> str | None:
    """Poll Modelslab fetch endpoint until success/error. Returns image URL or None."""
    import requests
    for attempt in range(max_attempts):
        fetch_resp = requests.post(
            "https://modelslab.com/api/v6/images/fetch",
            json={"key": api_key, "request_id": prediction_id},
            timeout=30,
        )
        fetch_resp.raise_for_status()
        fdata = fetch_resp.json()
        if fdata.get("status") == "success":
            urls = fdata.get("output") or []
            return urls[0] if urls else None
        if fdata.get("status") == "error":
            raise RuntimeError(f"{prefix} poll error: {fdata.get('message', fdata)}")
        logger.info("%s still processing (attempt %d/%d)...", prefix, attempt + 1, max_attempts)
        time.sleep(5)
    raise RuntimeError(f"{prefix} timed out after polling")


def _modelslab_save(image_url: str, prefix: str) -> str:
    """Download an image URL with CDN retry, verify it is valid, and save to images dir."""
    import requests
    from PIL import Image as _PILImg
    from io import BytesIO as _BytesIO

    for _attempt in range(4):
        img_resp = requests.get(image_url, timeout=60)
        content = img_resp.content if img_resp.status_code == 200 else b""

        if len(content) >= 5000:
            try:
                pil_img = _PILImg.open(_BytesIO(content))
                pil_img.verify()
            except Exception as _e:
                if _attempt < 3:
                    logger.info("%s CDN not ready yet (attempt %d/4), retrying...", prefix, _attempt + 1)
                    time.sleep(8)
                    continue
                raise RuntimeError(f"{prefix}: downloaded content is not a valid image: {_e}")

            folder = _image_dir()
            filename = f"{prefix}_{int(time.time() * 1000)}.jpg"
            file_path = folder / filename
            file_path.write_bytes(content)
            logger.info("%s image saved: %s", prefix, filename)
            return f"images/{filename}"

        logger.info("%s CDN not ready (%d bytes, attempt %d/4), retrying...", prefix, len(content), _attempt + 1)
        time.sleep(8)

    raise RuntimeError(f"{prefix}: image CDN failed after 4 attempts for {image_url}")


def _generate_modelslab(prompt: str, options: Dict[str, Any]) -> str:
    """
    Generate image via Modelslab.

    - With avatar (options["image_data"]): FLUX Kontext Dev img2img — preserves character
      face/hair across all scenes with superior character retention (12B params).
    - Without avatar: FLUX realtime text2img — fast, free, high quality.
    """
    import requests

    api_key = os.getenv("MODELSLAB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MODELSLAB_API_KEY not set")

    width = str(int(options.get("width", 1024)))
    height = str(int(options.get("height", 768)))
    image_data = options.get("image_data")

    if image_data:
        # ModelsLab img2img requires a public URL for init_image — base64 is rejected
        # ("init image must be a valid url when base64 is a representation of false").
        # Decode the base64 bytes and upload to a public host once per avatar (cached).
        if isinstance(image_data, str) and "," in image_data:
            image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        cache_key = options.get("avatar_cache_key") or f"kontext_{hash(image_data[:80])}"
        init_image_url = _upload_image_for_api(img_bytes, cache_key=cache_key)

        _kontext_base_neg = (
            "photorealistic, realistic photograph, realistic skin texture, real person, "
            "photography, 3d render, CGI, blurry, low quality, distorted, bad anatomy, ugly, "
            "totoro, spirited away characters, no-face, calcifer, howl, ghibli mascots, "
            "anime mascots, copyright characters, brand mascots, "
            "adult, man, woman, grown-up, teenager, elderly, adult face, mature features, "
            "realistic human proportions, tall person, full grown adult, "
            "text, watermark, label, logo, signature, cropped, cut off, out of frame, "
            "duplicate characters, extra person, multiple children, crowd"
        )
        neg = options.get(
            "negative_prompt",
            _kontext_base_neg + ", " + _outfit_color_negatives(prompt),
        )
        payload = {
            "key": api_key,
            "model_id": "flux-kontext-dev",
            "prompt": prompt,
            "negative_prompt": neg,
            "init_image": init_image_url,
            "strength": options.get("strength", 0.60),
            "width": width,
            "height": height,
            "samples": "1",
            "num_inference_steps": "30",
            "guidance_scale": 6.5,
            "safety_checker": "no",
            "enhance_prompt": "no",
        }
        if options.get("seed") is not None:
            payload["seed"] = int(options["seed"])
        logger.info("ModelsLab FLUX Kontext Dev img2img: %s...", prompt[:60])
        resp = requests.post("https://modelslab.com/api/v6/images/img2img", json=payload, timeout=90)
        log_prefix = "kontext"
    else:
        _base_neg = (
            "photorealistic, realistic photograph, blurry, low quality, distorted, bad anatomy, ugly, "
            "totoro, spirited away characters, no-face, calcifer, howl, ghibli mascots, "
            "anime mascots, copyright characters, brand mascots, "
            "adult, man, woman, grown-up, teenager, elderly, adult face, mature features, "
            "realistic human proportions, tall person, full grown adult, "
            "text, watermark, label, logo, signature, cropped, cut off, out of frame, "
            "duplicate characters, extra person, multiple children, crowd"
        )
        neg = options.get("negative_prompt", _base_neg + ", " + _outfit_color_negatives(prompt))
        # text2img: FLUX Dev — highest quality scene images
        payload = {
            "key": api_key,
            "model_id": "flux",
            "prompt": prompt,
            "negative_prompt": neg,
            "width": width,
            "height": height,
            "samples": "1",
            "num_inference_steps": "30",
            "guidance_scale": 7.5,
            "safety_checker": "no",
            "enhance_prompt": "no",
        }
        if options.get("seed") is not None:
            payload["seed"] = int(options["seed"])
        logger.info("ModelsLab FLUX text2img: %s...", prompt[:60])
        resp = requests.post("https://modelslab.com/api/v6/images/text2img", json=payload, timeout=90)
        log_prefix = "modelslab"

    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"Modelslab error: {data.get('message', data)}")

    if data.get("status") == "success":
        urls = data.get("output") or []
        if not urls:
            raise RuntimeError("Modelslab success but no output URLs")
        return _modelslab_save(urls[0], log_prefix)

    if data.get("status") == "processing":
        prediction_id = data.get("id")
        eta = int(data.get("eta", 10))
        logger.info("Modelslab processing (id=%s, eta=%ss)...", prediction_id, eta)
        time.sleep(min(eta, 15))
        image_url = _modelslab_poll(api_key, prediction_id, log_prefix)
        if image_url:
            return _modelslab_save(image_url, log_prefix)
        raise RuntimeError("Modelslab poll returned no URL")

    raise RuntimeError(f"Modelslab unexpected response status: {data.get('status')}")


def _generate_modelslab_ghibli(prompt: str, options: Dict[str, Any]) -> str:
    """
    Generate image via ModelsLab using anything-v5 (ghibli model taken offline).
    Uses img2img when an avatar is provided, text2img otherwise.
    """
    import requests

    api_key = os.getenv("MODELSLAB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MODELSLAB_API_KEY not set")

    width = str(int(options.get("width", 768)))
    height = str(int(options.get("height", 768)))
    neg = options.get(
        "negative_prompt",
        "photorealistic, realistic, 3d render, CGI, western cartoon, low quality, blurry, distorted, bad anatomy",
    )
    image_data = options.get("image_data")

    if image_data:
        if isinstance(image_data, str) and "," in image_data:
            image_data = image_data.split(",")[1]
        payload = {
            "key": api_key,
            "model_id": "anything-v5",
            "prompt": prompt,
            "negative_prompt": neg,
            "init_image": image_data,
            "strength": options.get("strength", 0.60),
            "base64": True,
            "width": width,
            "height": height,
            "samples": "1",
            "num_inference_steps": "20",
            "guidance_scale": 7.5,
            "safety_checker": "no",
            "enhance_prompt": "no",
        }
        logger.info("ModelsLab anything-v5 img2img: %s...", prompt[:60])
        resp = requests.post("https://modelslab.com/api/v6/images/img2img", json=payload, timeout=90)
        log_prefix = "anythingv5_i2i"
    else:
        payload = {
            "key": api_key,
            "model_id": "anything-v5",
            "prompt": prompt,
            "negative_prompt": neg,
            "width": width,
            "height": height,
            "samples": "1",
            "num_inference_steps": "20",
            "guidance_scale": 7.5,
            "safety_checker": "no",
            "enhance_prompt": "no",
        }
        logger.info("ModelsLab anything-v5 text2img: %s...", prompt[:60])
        resp = requests.post("https://modelslab.com/api/v6/images/text2img", json=payload, timeout=90)
        log_prefix = "anythingv5"

    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"ModelsLab Ghibli error: {data.get('message', data)}")

    if data.get("status") == "success":
        urls = data.get("output") or []
        if not urls:
            raise RuntimeError("ModelsLab Ghibli success but no output URLs")
        return _modelslab_save(urls[0], log_prefix)

    if data.get("status") == "processing":
        prediction_id = data.get("id")
        eta = int(data.get("eta", 15))
        logger.info("ModelsLab Ghibli processing (id=%s, eta=%ss)...", prediction_id, eta)
        time.sleep(min(eta, 20))
        image_url = _modelslab_poll(api_key, prediction_id, log_prefix)
        if image_url:
            return _modelslab_save(image_url, log_prefix)
        raise RuntimeError("ModelsLab Ghibli poll returned no URL")

    raise RuntimeError(f"ModelsLab Ghibli unexpected status: {data.get('status')}")


def _generate_modelslab_facegen(prompt: str, options: Dict[str, Any]) -> str:
    """
    Generate scene image via ModelsLab face_gen (ai-avatar-generatorface-gen).

    Injects the child's actual face from the original photo into a chibi-style
    scene illustration. Provides much stronger face/identity consistency than
    FLUX img2img because the face is extracted directly from the photo rather
    than approximated through a style transfer.

    Requires options["face_image_b64"]: base64-encoded original photo (no prefix).
    Falls back gracefully — caller should catch RuntimeError and use FLUX img2img.
    """
    import requests

    api_key = os.getenv("MODELSLAB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MODELSLAB_API_KEY not set")

    face_b64 = options.get("face_image_b64")
    if not face_b64:
        raise RuntimeError("face_gen requires face_image_b64")

    style = options.get("facegen_style", "chibi")

    neg = options.get(
        "negative_prompt",
        "drawing, big nose, long nose, fat, ugly, big lips, big mouth, "
        "face proportion mismatch, unrealistic, monochrome, lowres, bad anatomy, "
        "worst quality, low quality, blurry, adult, man, woman, grown-up, teenager, "
        "elderly, adult face, mature features, realistic human proportions, tall person, "
        "full grown adult",
    )

    payload = {
        "key": api_key,
        "model_id": "ai-avatar-generatorface-gen",
        "face_image": face_b64,
        "prompt": prompt,
        "style": style,
        "negative_prompt": neg,
        "num_inference_steps": "31",
        "guidance_scale": 7.0,
        "base64": True,
    }
    if options.get("seed") is not None:
        payload["seed"] = int(options["seed"])

    logger.info("ModelsLab FaceGen (style=%s): %s...", style, prompt[:60])
    resp = requests.post(
        "https://modelslab.com/api/v6/image_editing/face_gen",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"ModelsLab FaceGen error: {data.get('message', data)}")

    if data.get("status") == "success":
        urls = data.get("output") or []
        if not urls:
            raise RuntimeError("ModelsLab FaceGen success but no output URLs")
        return _modelslab_save(urls[0], "facegen")

    if data.get("status") == "processing":
        prediction_id = data.get("id")
        eta = int(data.get("eta", 20))
        # FaceGen is slow — initial wait up to 30s, then poll 24×5s = 120s more (150s total).
        logger.info("ModelsLab FaceGen processing (id=%s, eta=%ss)...", prediction_id, eta)
        time.sleep(min(eta, 30))
        image_url = _modelslab_poll(api_key, prediction_id, "facegen", max_attempts=24)
        if image_url:
            return _modelslab_save(image_url, "facegen")
        raise RuntimeError("ModelsLab FaceGen poll returned no URL")

    raise RuntimeError(f"ModelsLab FaceGen unexpected status: {data.get('status')}")


def _generate_modelslab_toonyou(prompt: str, options: Dict[str, Any]) -> str:
    """
    Generate scene image via ToonYou (SD 1.5 cartoon model) + IP-Adapter Plus Face.

    ToonYou produces flat cel-shaded cartoon/chibi artwork that matches the project's
    picture-book aesthetic. IP-Adapter Plus Face injects face identity directly from the
    avatar photo so the child's likeness appears in every scene.

    Requires options["ip_adapter_image_url"]: public URL of the avatar photo.
    Falls back gracefully — caller should catch RuntimeError and use FaceGen.
    """
    import requests

    api_key = os.getenv("MODELSLAB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MODELSLAB_API_KEY not set")

    # SD 1.5 models are most reliable at 768x512 (landscape, avoids anatomy artefacts)
    width = str(int(options.get("width", 768)))
    height = str(int(options.get("height", 512)))

    _toonyou_base_neg = (
        "photorealistic, realistic photograph, 3d render, CGI, "
        "low quality, blurry, distorted, bad anatomy, extra limbs, "
        "adult, man, woman, grown-up, elderly, mature features, "
        "ugly, deformed, noisy, out of focus, watermark, signature"
    )
    neg = options.get("negative_prompt", _toonyou_base_neg + ", " + _outfit_color_negatives(prompt))

    payload: Dict[str, Any] = {
        "key": api_key,
        "model_id": "toonyou",
        "prompt": prompt,
        "negative_prompt": neg,
        "width": width,
        "height": height,
        "samples": "1",
        "num_inference_steps": "30",
        "guidance_scale": 7.0,
        "clip_skip": 2,
        "scheduler": "DPMSolverMultistepScheduler",
        "safety_checker": "no",
        "enhance_prompt": "no",
    }

    if options.get("seed") is not None:
        payload["seed"] = int(options["seed"])

    avatar_url = options.get("ip_adapter_image_url")
    if not avatar_url:
        raise RuntimeError("ToonYou requires ip_adapter_image_url for IP-Adapter Face")
    payload["ip_adapter_id"] = "ip-adapter-plus-face_sd15"
    payload["ip_adapter_scale"] = float(options.get("ip_adapter_scale", 0.7))
    payload["ip_adapter_image"] = avatar_url

    logger.info("ModelsLab ToonYou + IP-Adapter Plus Face: %s...", prompt[:60])
    resp = requests.post(
        "https://modelslab.com/api/v6/images/text2img",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"ModelsLab ToonYou error: {data.get('message', data)}")

    if data.get("status") == "success":
        urls = data.get("output") or []
        if not urls:
            raise RuntimeError("ModelsLab ToonYou success but no output URLs")
        return _modelslab_save(urls[0], "toonyou")

    if data.get("status") == "processing":
        prediction_id = data.get("id")
        eta = int(data.get("eta", 20))
        logger.info("ModelsLab ToonYou processing (id=%s, eta=%ss)...", prediction_id, eta)
        time.sleep(min(eta, 25))
        image_url = _modelslab_poll(api_key, prediction_id, "toonyou")
        if image_url:
            return _modelslab_save(image_url, "toonyou")
        raise RuntimeError("ModelsLab ToonYou poll returned no URL")

    raise RuntimeError(f"ModelsLab ToonYou unexpected status: {data.get('status')}")


def _generate_pollinations(prompt: str, options: Dict[str, Any]) -> str:
    """
    Generate image via Pollinations.AI — completely free, no API key required.
    Uses FLUX model which produces high-quality anime/Ghibli style images.
    """
    import requests
    from urllib.parse import quote

    model = options.get("pollinations_model", "flux")
    width = int(options.get("width", 1024))
    height = int(options.get("height", 1024))
    seed = int(time.time() * 1000) % 99999

    encoded = quote(prompt, safe="")
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&nologo=true&seed={seed}&model={model}&enhance=true"
    )

    logger.info("Calling Pollinations.AI (model=%s): %s...", model, prompt[:60])
    resp = requests.get(url, timeout=90)
    if resp.status_code == 200 and len(resp.content) > 5000:
        folder = _image_dir()
        filename = f"pollinations_{int(time.time() * 1000)}.jpg"
        file_path = folder / filename
        file_path.write_bytes(resp.content)
        logger.info("Pollinations image saved: %s", filename)
        return f"images/{filename}"
    raise RuntimeError(f"Pollinations failed: {resp.status_code}, size={len(resp.content)}")


def _fallback_all_or_stub(prompt: str, options: Dict[str, Any]) -> str:
    current = _provider(options.get("image_provider"))

    # Try Pollinations first (always free, no quota)
    if current != "pollinations":
        try:
            return _generate_pollinations(prompt, options)
        except Exception as e:
            logger.warning("Pollinations fallback failed: %s", e)

    # Try Modelslab (30 free/day)
    if os.getenv("MODELSLAB_API_KEY") and current != "modelslab":
        try:
            return _generate_modelslab(prompt, options)
        except Exception as e:
            logger.warning("Modelslab fallback failed: %s", e)

    # Try Gemini
    if os.getenv("GEMINI_API_KEY") and current != "gemini":
        logger.info("Falling back to Gemini image generation")
        fallback_options = dict(options)
        fallback_options["image_provider"] = "gemini"
        try:
            return generate_image(prompt, fallback_options)
        except Exception:
            pass

    # Try OpenAI
    if os.getenv("OPENAI_API_KEY") and current != "openai":
        logger.info("Falling back to OpenAI image generation")
        fallback_options = dict(options)
        fallback_options["image_provider"] = "openai"
        try:
            return generate_image(prompt, fallback_options)
        except Exception:
            pass

    logger.warning("All image providers failed. Returning placeholder stub.")
    return _stub_image(prompt)


_ALL_SHIRT_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "brown", "grey", "gray", "teal", "cyan",
]

_OUTFIT_GARMENTS = ["t-shirt", "shirt", "dress", "skirt", "uniform", "salwar", "kameez"]

def _outfit_color_negatives(prompt: str) -> str:
    """
    Extract the outfit color from the prompt and return a negative-prompt
    string blocking all other outfit colors. Prevents FLUX/SD from drifting the
    outfit color between scenes. Covers shirts, dresses, skirts, and ethnic wear.
    """
    prompt_lower = prompt.lower()
    for color in _ALL_SHIRT_COLORS:
        for garment in _OUTFIT_GARMENTS:
            if f"{color} {garment}" in prompt_lower:
                wrong = [c for c in _ALL_SHIRT_COLORS if c != color]
                return (
                    ", ".join(f"{c} {garment}" for c in wrong)
                    + ", wrong outfit color, outfit color change, different outfit"
                )
    return "wrong outfit color, outfit color change"


_SD_STYLE_KEYWORDS = {
    # Use full phrases only — avoids misclassifying "ghibli-style" in character anchors
    "ghibli anime", "studio ghibli", "anime style", "hand-drawn", "hand drawn",
    "masterpiece", "vibrant colors", "high quality", "cinematic depth",
    "panoramic shot", "wide angle", "wide shot", "full body figure",
    "character small", "small in frame", "fills the frame",
    "detailed environment", "detailed background", "cinematic",
}

def _reorder_prompt_for_sd(prompt: str) -> str:
    """
    Local SD/Ghibli models use CLIP with a 77-token hard limit.
    CLIP weights early tokens most heavily, so if style tags come first
    ("Studio Ghibli anime style...") the model attends to style and ignores
    the scene action. This function moves action/content parts before style
    tags so the action is what the model focuses on.
    """
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    action_parts, style_parts = [], []
    for part in parts:
        part_lower = part.lower()
        if any(kw in part_lower for kw in _SD_STYLE_KEYWORDS):
            style_parts.append(part)
        else:
            action_parts.append(part)
    if not action_parts:
        return prompt
    return ", ".join(action_parts + style_parts)


def _generate_local(prompt: str, options: Dict[str, Any]) -> str:
    """Generate image using a local API endpoint."""
    api_url = os.getenv("LOCAL_IMAGE_API_URL", "http://127.0.0.1:5000/generate")
    logger.info(f"Generating image via Local API: {api_url}")

    try:
        # Reorder: action first, style tags last — critical for CLIP 77-token limit
        enhanced_prompt = _reorder_prompt_for_sd(prompt)
        # Append Ghibli style at the END if not already present
        if "ghibli" not in enhanced_prompt.lower():
            enhanced_prompt = f"{enhanced_prompt}, ghibli anime style, masterpiece, high quality, hand-drawn, vibrant colors"

        payload = {"prompt": enhanced_prompt}
        # Add common params in case the local API supports them
        payload.update({
            "width": options.get("width", 1024),
            "height": options.get("height", 1024),
            "negative_prompt": options.get(
                "negative_prompt",
                "photorealistic, realistic photograph, 3d render, CGI, western cartoon, "
                "low quality, blurry, distorted, bad anatomy, missing limbs, extra limbs",
            ),
            "num_inference_steps": 30
        })
        
        # If this is an img2img request (e.g. for avatar conversion)
        if options.get("image_data"):
            image_data = options.get("image_data")
            # Strip base64 prefix if present (e.g. "data:image/jpeg;base64,")
            if isinstance(image_data, str) and "," in image_data:
                image_data = image_data.split(",")[1]
            payload["image"] = image_data
            # For img2img, we usually want lower denoising if the model supports it, 
            # but for style transfer we want it high enough to change the style.
            payload["strength"] = options.get("strength", 0.6)
        
        import requests
        headers = {"ngrok-skip-browser-warning": "true"}
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            folder = _image_dir()
            filename = f"local_{int(time.time() * 1000)}.jpg"
            file_path = folder / filename
            
            # Check if response is JSON (common for FastAPI/Flask base64)
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/json" in content_type:
                data = response.json()
                # Try common keys like 'image', 'images', 'output'
                img_data_raw = data.get("image") or data.get("images", [None])[0] or data.get("output")
                if img_data_raw:
                    if isinstance(img_data_raw, list): img_data_raw = img_data_raw[0]
                    # Remove data:image/...;base64, prefix if present
                    if isinstance(img_data_raw, str) and "," in img_data_raw:
                        img_data_raw = img_data_raw.split(",")[1]
                    
                    decoded_data = base64.b64decode(img_data_raw)
                    
                    # SAFETY CHECK: If the image is suspiciously small (e.g. < 5KB for a large image), 
                    # it's likely a black/empty square from a failed local model or NSFW filter.
                    if len(decoded_data) < 5000:
                        logger.warning(f"Local API returned suspiciously small image ({len(decoded_data)} bytes). Possible black box. Falling back.")
                        return "" # Return empty to trigger fallback

                    with open(file_path, "wb") as f:
                        f.write(decoded_data)
                    return str(file_path.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")
                else:
                    raise ValueError("No image data found in local API JSON response")
            else:
                # Assume direct binary image response
                with open(file_path, "wb") as f:
                    f.write(response.content)
            
            return str(file_path.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")
        else:
            raise RuntimeError(f"Local API failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"Local Image API hard failure: {e}")
        log_api_error(f"Local Image API hard failure: {e}")
        raise


def generate_image(prompt: str, options: Dict[str, Any] | None = None) -> str:
    """
    Generate an image for the given prompt and return a relative path under MEDIA_ROOT.

    If no working provider is configured, falls back to stub mode.
    """
    options = options or {}
    provider = _provider(options.get("image_provider"))

    # Pollinations — free, no key required, try first
    if provider == "pollinations":
        try:
            return _generate_pollinations(prompt, options)
        except Exception as e:
            logger.warning("Pollinations failed: %s. Falling back...", e)
            return _fallback_all_or_stub(prompt, dict(options, image_provider="pollinations"))

    # ModelsLab FaceGen — injects real face into chibi/anime scene (no cross-provider fallback)
    if provider == "modelslab_facegen":
        last_err = None
        for attempt in range(3):
            try:
                return _generate_modelslab_facegen(prompt, options)
            except Exception as e:
                last_err = e
                logger.warning("ModelsLab FaceGen attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(10)
        logger.error("ModelsLab FaceGen failed after 3 attempts: %s", last_err)
        return _stub_image(prompt)

    # Modelslab Ghibli — retry within ModelsLab only (no cross-provider fallback)
    if provider == "modelslab_ghibli":
        last_err = None
        for attempt in range(3):
            try:
                return _generate_modelslab_ghibli(prompt, options)
            except Exception as e:
                last_err = e
                logger.warning("ModelsLab Ghibli attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(10)
        logger.error("ModelsLab Ghibli failed after 3 attempts: %s", last_err)
        return _stub_image(prompt)

    # Modelslab FLUX — retry within ModelsLab only (no cross-provider fallback)
    if provider == "modelslab":
        last_err = None
        for attempt in range(3):
            try:
                return _generate_modelslab(prompt, options)
            except Exception as e:
                last_err = e
                logger.warning("ModelsLab attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(10)
        logger.error("ModelsLab failed after 3 attempts: %s", last_err)
        return _stub_image(prompt)

    # Check for local API first if provider is local
    if provider == "local":
        try:
            return _generate_local(prompt, options)
        except Exception as e:
            logger.warning(f"Local Image API failed: {e}. Falling back...")
            return _fallback_all_or_stub(prompt, options)

    # Global Stub Mode (Credit Protection)
    if os.getenv("USE_STUB_MODE", "false").lower() == "true":
        provider = "stub"
        
    render_preset = get_render_preset()

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.images.generate(
                model=options.get("model", "dall-e-3"),
                prompt=prompt,
                size=options.get("size", "1024x1024"),
                n=1,
            )
            image_url = response.data[0].url

            import requests

            folder = _image_dir()
            filename = f"openai_{hash(prompt) % 10000}.png"
            file_path = folder / filename
            
            img_response = requests.get(image_url)
            with open(file_path, "wb") as f:
                f.write(img_response.content)
            return f"images/{filename}"
        except Exception as e:
            # Fallback to stub if OpenAI API fails
            logger.warning("OpenAI image generation failed: %s", e)
            return _stub_image(prompt)

    if provider == "replicate" and os.getenv("REPLICATE_API_TOKEN"):
        try:
            import requests

            model_name = options.get("model") or os.getenv("REPLICATE_MODEL_NAME") or "black-forest-labs/flux-schnell"
            model_version = options.get("model_version") or os.getenv("REPLICATE_MODEL_VERSION")
            width = options.get("width") or os.getenv("REPLICATE_WIDTH")
            height = options.get("height") or os.getenv("REPLICATE_HEIGHT")

            headers = {
                "Authorization": f"Token {os.getenv('REPLICATE_API_TOKEN')}",
                "Content-Type": "application/json",
            }
            input_payload = {"prompt": prompt}
            if width:
                input_payload["width"] = width
            if height:
                input_payload["height"] = height

            if model_version:
                payload = {
                    "version": model_version,
                    "input": input_payload,
                }
                create_url = "https://api.replicate.com/v1/predictions"
            else:
                payload = {"input": input_payload}
                create_url = f"https://api.replicate.com/v1/models/{model_name}/predictions"

            create_resp = requests.post(create_url, json=payload, headers=headers, timeout=15)
            if create_resp.status_code >= 400:
                # Retry with model endpoint if supplied version is invalid.
                if model_version and create_resp.status_code == 422:
                    logger.warning("Replicate version rejected, retrying without pinned version.")
                    payload = {"input": input_payload}
                    create_url = f"https://api.replicate.com/v1/models/{model_name}/predictions"
                    create_resp = requests.post(create_url, json=payload, headers=headers, timeout=15)

            if create_resp.status_code >= 400:
                logger.warning(
                    "Replicate create failed (%s): %s | model=%s version=%s",
                    create_resp.status_code,
                    create_resp.text,
                    model_name,
                    model_version,
                )
                return _fallback_all_or_stub(prompt, options)

            prediction = create_resp.json()
            prediction_url = prediction.get("urls", {}).get("get")
            if not prediction_url:
                logger.warning("Replicate: missing prediction URL in response")
                return _stub_image(prompt)

            for _ in range(20):  # ~20s max wait
                poll_resp = requests.get(prediction_url, headers=headers, timeout=15)
                if poll_resp.status_code >= 400:
                    logger.warning("Replicate poll failed (%s): %s", poll_resp.status_code, poll_resp.text)
                    return _fallback_all_or_stub(prompt, options)

                data = poll_resp.json()
                status = data.get("status")
                if status == "succeeded":
                    outputs = data.get("output") or []
                    if not outputs:
                        logger.warning("Replicate: succeeded but no outputs returned")
                        break
                    image_url = outputs[0]
                    folder = _image_dir()
                    filename = f"replicate_{hash(prompt) % 10000}.png"
                    file_path = folder / filename
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code >= 400:
                        logger.warning("Replicate image download failed (%s): %s", img_response.status_code, img_response.text)
                        return _fallback_all_or_stub(prompt, options)
                    with open(file_path, "wb") as f:
                        f.write(img_response.content)
                    return f"images/{filename}"
                if status in {"failed", "canceled"}:
                    logger.warning("Replicate prediction %s: %s", status, data.get("error") or data)
                    break
                time.sleep(1)
        except Exception as e:
            logger.exception("Replicate image generation failed: %s", e)
        return _fallback_all_or_stub(prompt, options)

    if provider == "local_sd":
        try:
            import torch
            from diffusers import StableDiffusionPipeline

            model_id = options.get("model") or os.getenv("LOCAL_SD_MODEL") or "runwayml/stable-diffusion-v1-5"
            folder = _image_dir()
            filename = f"local_sd_{hash(prompt) % 10000}.png"
            file_path = folder / filename

            use_cuda = torch.cuda.is_available()
            torch_dtype = torch.float16 if use_cuda else torch.float32
            device = "cuda" if use_cuda else "cpu"

            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe = pipe.to(device)

            width = int(options.get("width") or render_preset["width"])
            height = int(options.get("height") or render_preset["height"])
            # Keep generation size practical on smaller VRAM GPUs.
            width = min(width, 768)
            height = min(height, 768)

            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=int(options.get("steps", 25)),
                    guidance_scale=float(options.get("guidance_scale", 7.5)),
                    width=width,
                    height=height,
                ).images[0]
            image.save(file_path)
            return f"images/{filename}"
        except Exception as e:
            logger.warning("local_sd image generation failed: %s", e)
            return _stub_image(prompt)

    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai
            from google.genai import types as _gtypes

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            imagen_model = (
                os.getenv("GEMINI_IMAGE_MODEL")
                or "gemini-2.0-flash-preview-image-generation"
            )

            response = client.models.generate_content(
                model=imagen_model,
                contents=prompt,
                config=_gtypes.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            image_data = None
            for candidate in (response.candidates or []):
                for part in (getattr(candidate.content, "parts", None) or []):
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        image_data = inline.data
                        break
                if image_data:
                    break

            if not image_data:
                raise RuntimeError("Gemini image generation returned no image data")

            folder = _image_dir()
            filename = f"gemini_img_{int(time.time() * 1000)}.png"
            file_path = folder / filename
            file_path.write_bytes(image_data)
            logger.info("Gemini image generated: %s", filename)
            return f"images/{filename}"
        except Exception as e:
            logger.warning("Gemini image generation failed: %s", e)
            log_api_error(f"Gemini image generation failed: {e}")
            if os.getenv("REPLICATE_API_TOKEN"):
                fallback_options = dict(options)
                fallback_options["image_provider"] = "replicate"
                return generate_image(prompt, fallback_options)
            if os.getenv("OPENAI_API_KEY"):
                fallback_options = dict(options)
                fallback_options["image_provider"] = "openai"
                return generate_image(prompt, fallback_options)
            return _stub_image(prompt)

    if provider == "stability" and os.getenv("STABILITY_API_KEY"):
        try:
            import requests

            api_key = os.getenv("STABILITY_API_KEY", "").strip()
            image_data = options.get("image_data") # base64 with prefix

            # Handle Image-to-Image (img2img) if an image is provided
            if image_data:
                # Use SDXL for reliable img2img
                engine_id = "stable-diffusion-xl-1024-v1-0"
                url = f"https://api.stability.ai/v1/generation/{engine_id}/image-to-image"
                
                # Extract raw bytes from base64
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                img_bytes = base64.b64decode(image_data)

                logger.info(f"Calling Stability AI img2img ({engine_id})")
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json"
                    },
                    files={"init_image": img_bytes},
                    data={
                        "image_strength": options.get("strength", 0.45), # Higher value = more like original
                        "init_image_mode": "IMAGE_STRENGTH",
                        "text_prompts[0][text]": prompt,
                        "text_prompts[0][weight]": 1,
                        "cfg_scale": 7,
                        "samples": 1,
                        "steps": 30,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    folder = _image_dir()
                    filename = f"stability_i2i_{int(time.time() * 1000)}.png"
                    file_path = folder / filename
                    with open(file_path, "wb") as f:
                        f.write(base64.b64decode(data["artifacts"][0]["base64"]))
                    return f"images/{filename}"
                else:
                    logger.warning(f"Stability img2img failed ({response.status_code}), falling back to text-to-image")

            # Standard Text-to-Image Path (v2beta)
            model = options.get("model") or os.getenv("STABILITY_MODEL") or "stable-image-ultra"
            url = f"https://api.stability.ai/v2beta/stable-image/generate/{model.split('-')[-1]}"
            if model == "stable-image-ultra":
                url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
            elif model == "stable-image-core":
                url = "https://api.stability.ai/v2beta/stable-image/generate/core"
            elif model == "sd3":
                url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

            logger.info("Calling Stability AI (%s) with prompt: %s", model, prompt[:50])
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "image/*"
                },
                files={"none": ""},
                data={
                    "prompt": prompt,
                    "output_format": options.get("output_format", "png"),
                },
            )

            if response.status_code == 200:
                logger.info("Stability AI generation successful")
                folder = _image_dir()
                filename = f"stability_{hash(prompt) % 10000}.png"
                file_path = folder / filename
                with open(file_path, "wb") as f:
                    f.write(response.content)
                return f"images/{filename}"
            else:
                error_msg = f"Stability AI generation failed. Status: {response.status_code}, Response: {response.text}"
                logger.error(error_msg)
                log_api_error(error_msg)
                return _fallback_all_or_stub(prompt, options)
        except Exception as e:
            logger.exception("Stability AI image generation failed: %s", e)
            return _fallback_all_or_stub(prompt, options)

    if provider == "flux" and os.getenv("IMAGE_API_KEY"):
        # Placeholder for future Black Forest Labs FLUX integration
        return _stub_image(prompt)

    return _stub_image(prompt)


def _build_facegen_prompt(prompt: str, anchor: str) -> str:
    """
    Strip hair/eye/skin traits from the character anchor when the prompt will be
    sent to face_gen — that model extracts face appearance directly from the photo,
    so text-based face descriptors conflict with the extracted face and cause distortion.

    Keeps only age/gender and the last two chibi proportion cues:
      "a 8-year-old boy with short curly black hair, dark brown eyes, white shirt, chibi proportions, large anime eyes"
      → "a 8-year-old boy, chibi proportions, large anime eyes"
    """
    if not anchor or anchor not in prompt:
        return prompt
    parts = [p.strip() for p in anchor.split(",")]
    # Anchor always ends with "chibi proportions" and "large anime eyes" — keep those.
    # The first part is "a N-year-old {gender} with ..." — strip the "with ..." portion.
    if len(parts) >= 3 and "chibi" in parts[-2]:
        base = parts[0].split(" with ")[0].strip()
        facegen_anchor = base + ", " + ", ".join(parts[-2:])
        return prompt.replace(anchor, facegen_anchor, 1)
    return prompt


# Cache avatar public URLs within a process so we upload once per story, not once per scene
_avatar_url_cache: dict[str, str] = {}


def _upload_image_for_api(image_bytes: bytes, cache_key: str | None = None) -> str:
    """
    Upload image bytes to a public host and return a URL ModelsLab can fetch.
    Tries three hosts in order so a single host outage never blocks all scenes.

    Hosts tried:
      1. 0x0.st       — primary (3 attempts, exponential backoff)
      2. tmpfiles.org — secondary (1 attempt)
      3. litterbox.catbox.moe — tertiary (1-hour temp file, 1 attempt)
    """
    import requests as _req

    if cache_key and cache_key in _avatar_url_cache:
        logger.info("Reusing cached avatar URL for key: %s", cache_key[:40])
        return _avatar_url_cache[cache_key]

    last_err: Exception | None = None

    # ── Host 1: 0x0.st (3 attempts) ─────────────────────────────────────────
    for _attempt in range(3):
        try:
            resp = _req.post(
                "https://0x0.st",
                files={"file": ("avatar.jpg", image_bytes, "image/jpeg")},
                timeout=30,
            )
            resp.raise_for_status()
            url = resp.text.strip()
            if url.startswith("http"):
                logger.info("Avatar uploaded to 0x0.st: %s", url)
                if cache_key:
                    _avatar_url_cache[cache_key] = url
                return url
        except Exception as _e:
            last_err = _e
            logger.warning("0x0.st upload attempt %d/3 failed: %s", _attempt + 1, _e)
            if _attempt < 2:
                time.sleep(4 * (_attempt + 1))

    # ── Host 2: tmpfiles.org ─────────────────────────────────────────────────
    try:
        resp = _req.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": ("avatar.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_url = (data.get("data") or {}).get("url", "")
        if raw_url.startswith("http"):
            # tmpfiles serves downloads under /dl/ not the page URL
            url = raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
            logger.info("Avatar uploaded to tmpfiles.org: %s", url)
            if cache_key:
                _avatar_url_cache[cache_key] = url
            return url
    except Exception as _e:
        last_err = _e
        logger.warning("tmpfiles.org upload failed: %s", _e)

    # ── Host 3: litterbox.catbox.moe (1-hour temp) ───────────────────────────
    try:
        resp = _req.post(
            "https://litterbox.catbox.moe/resources/internals/api.php",
            data={"reqtype": "fileupload", "time": "1h"},
            files={"fileToUpload": ("avatar.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        resp.raise_for_status()
        url = resp.text.strip()
        if url.startswith("http"):
            logger.info("Avatar uploaded to litterbox.catbox.moe: %s", url)
            if cache_key:
                _avatar_url_cache[cache_key] = url
            return url
    except Exception as _e:
        last_err = _e
        logger.warning("litterbox.catbox.moe upload failed: %s", _e)

    raise RuntimeError(f"All upload hosts failed. Last error: {last_err}")


def _generate_modelslab_scene_with_avatar(
    scene_text: str,
    image_prompt: str,
    avatar_url: str,
    api_key: str,
) -> str:
    """
    Generate a scene image via ModelsLab anything-v5 img2img.

    Uses the Ghibli-converted avatar as init_image with strength 0.55:
    - Low enough to keep character face/hair/outfit style from the avatar
    - High enough for the prompt to drive the full scene action and background
    - anything-v5 is the best active anime model on ModelsLab (ghibli model offline)

    ModelsLab has no IP-Adapter endpoint — img2img at the right strength is the
    correct approach for character consistency with their current API.
    """
    import requests as _req

    # LLM generates a complete anything-v5 optimised prompt (quality + action + style tags).
    # Do not append duplicate style tags here — they waste the 77-token CLIP budget.
    prompt = image_prompt
    neg = (
        "photorealistic, realistic photograph, 3d render, CGI, low quality, "
        "blurry, distorted, bad anatomy, extra limbs, "
        "(close-up face portrait:1.4), (cropped:1.3), missing body, "
        "text, watermark, signature, username"
    )

    payload = {
        "key": api_key,
        "model_id": "anything-v5",
        "prompt": prompt,
        "negative_prompt": neg,
        "init_image": avatar_url,
        "strength": 0.55,            # 0.55: avatar anchors character appearance, prompt drives scene action
        "width": "768",
        "height": "768",
        "samples": "1",
        "num_inference_steps": "30",
        "guidance_scale": 7.5,
        "safety_checker": "no",
        "enhance_prompt": "no",      # off — preserves character anchor injected in prompt
    }

    logger.info("ModelsLab anything-v5 scene img2img: %s...", scene_text[:60])
    resp = _req.post(
        "https://modelslab.com/api/v6/images/img2img",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    logger.info("ModelsLab scene img2img status: %s", data.get("status"))

    if data.get("status") == "error":
        raise RuntimeError(f"ModelsLab scene img2img error: {data.get('message', data)}")

    image_url = None
    if data.get("status") == "success":
        image_url = (data.get("output") or [None])[0]
    elif data.get("status") == "processing":
        pred_id = data.get("id")
        eta = int(data.get("eta", 20))
        logger.info("ModelsLab scene img2img processing (eta=%ss)...", eta)
        time.sleep(min(eta, 25))
        image_url = _modelslab_poll(api_key, pred_id, "scene_anythingv5")

    if image_url:
        return _modelslab_save(image_url, "scene_anythingv5")
    raise RuntimeError(f"ModelsLab scene img2img returned no image. Response: {data}")


def generate_scene_image(
    prompt: str,
    avatar_path: str | None = None,
    image_provider: str | None = None,
    scene_text: str | None = None,
    seed: int | None = None,
    character_anchor: str | None = None,
) -> str:
    """
    Generate a scene image from a text prompt.

    Priority chain when avatar is present (highest → lowest quality):
      1. ToonYou + IP-Adapter Plus Face — chibi cartoon + face identity from avatar
      2. FLUX Kontext Dev — style-transfer img2img, URL
      3. FLUX text2img    — no avatar, text-only

    Args:
        prompt: Assembled image prompt (anchor + scene fields + style tag)
        avatar_path: Relative path to the Ghibli-converted avatar (optional)
        image_provider: Optional provider override
        scene_text: Raw story text (informational only)
        seed: Shared seed for all 6 scenes — ensures colour/style consistency
        character_anchor: Character description injected into the scene prompt

    Returns:
        Relative path to generated image under MEDIA_ROOT (e.g., "images/scene_123.jpg")
    """
    api_key = os.getenv("MODELSLAB_API_KEY", "").strip()

    if avatar_path and api_key:
        try:
            avatar_full_path = Path(settings.MEDIA_ROOT) / avatar_path
            if avatar_full_path.exists():
                with open(avatar_full_path, "rb") as _f:
                    avatar_bytes = _f.read()
                avatar_b64 = base64.b64encode(avatar_bytes).decode("utf-8")

                # Upload avatar once; all tiers that need a URL reuse the cached value.
                avatar_url = _upload_image_for_api(avatar_bytes, cache_key=avatar_path)

                # ── Tier 1: FLUX Kontext Dev img2img ────────────────────────
                # Uses avatar as structural reference + full text prompt for scene.
                # 12B param model that follows _STYLE_TAG and character anchor properly.
                logger.info("Scene img2img with FLUX Kontext Dev (base64 avatar): %s", avatar_path)
                i2i_opts: Dict[str, Any] = {
                    "image_data": avatar_b64,
                    "avatar_cache_key": avatar_path,
                }
                if seed is not None:
                    i2i_opts["seed"] = seed
                try:
                    return _generate_modelslab(prompt, i2i_opts)
                except Exception as kontext_err:
                    logger.warning("FLUX Kontext failed, falling back to ToonYou: %s", kontext_err)

                # ── Tier 2: ToonYou + IP-Adapter Plus Face ───────────────────
                # SD 1.5 cartoon fallback with face-identity injection from the avatar.
                logger.info("Scene ToonYou + IP-Adapter Face (fallback): %s", avatar_path)
                ty_opts: Dict[str, Any] = {"ip_adapter_image_url": avatar_url}
                if seed is not None:
                    ty_opts["seed"] = seed
                return _generate_modelslab_toonyou(prompt, ty_opts)
        except Exception as e:
            logger.warning("ModelsLab scene generation failed, falling back to text2img: %s", e)

    options = {}
    provider = _provider(image_provider)
    if provider != "stub":
        options["image_provider"] = provider
    if seed is not None:
        options["seed"] = seed
    return generate_image(prompt, options)
