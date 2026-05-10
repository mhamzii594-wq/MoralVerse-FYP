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


def _modelslab_poll(api_key: str, prediction_id: str, prefix: str) -> str | None:
    """Poll Modelslab fetch endpoint until success/error. Returns image URL or None."""
    import requests
    for attempt in range(12):
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
        logger.info("%s still processing (attempt %d/12)...", prefix, attempt + 1)
        time.sleep(5)
    raise RuntimeError(f"{prefix} timed out after polling")


def _modelslab_save(image_url: str, prefix: str) -> str:
    """Download an image URL, verify it is a valid image, and save to images dir."""
    import requests
    from PIL import Image as _PILImg
    from io import BytesIO as _BytesIO

    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()
    content = img_resp.content

    if len(content) < 5000:
        raise RuntimeError(f"{prefix}: downloaded image too small ({len(content)} bytes)")

    # Verify the bytes are actually a valid image before saving
    try:
        pil_img = _PILImg.open(_BytesIO(content))
        pil_img.verify()  # raises if not a valid image format
    except Exception as _e:
        raise RuntimeError(f"{prefix}: downloaded content is not a valid image: {_e}")

    folder = _image_dir()
    filename = f"{prefix}_{int(time.time() * 1000)}.jpg"
    file_path = folder / filename
    file_path.write_bytes(content)
    logger.info("%s image saved: %s", prefix, filename)
    return f"images/{filename}"


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
    height = str(int(options.get("height", 1024)))
    neg = options.get(
        "negative_prompt",
        "photorealistic, realistic photograph, blurry, low quality, distorted, bad anatomy, ugly",
    )

    image_data = options.get("image_data")

    if image_data:
        # img2img: FLUX Kontext Dev — best character consistency across scenes
        if isinstance(image_data, str) and "," in image_data:
            image_data = image_data.split(",")[1]
        payload = {
            "key": api_key,
            "model_id": "flux-kontext-dev",
            "prompt": prompt,
            "negative_prompt": neg,
            "init_image": image_data,
            "strength": options.get("strength", 0.60),
            "base64": True,
            "width": width,
            "height": height,
            "samples": "1",
            "num_inference_steps": "30",
            "guidance_scale": 7.5,
            "safety_checker": "no",
            "enhance_prompt": "yes",
        }
        logger.info("ModelsLab FLUX Kontext Dev img2img: %s...", prompt[:60])
        resp = requests.post("https://modelslab.com/api/v6/images/img2img", json=payload, timeout=90)
        log_prefix = "kontext"
    else:
        # text2img: FLUX Dev — highest quality scene images
        payload = {
            "key": api_key,
            "model_id": "flux",
            "prompt": prompt,
            "negative_prompt": neg,
            "width": width,
            "height": height,
            "samples": "1",
            "num_inference_steps": "31",
            "guidance_scale": 7.5,
            "safety_checker": "no",
            "enhance_prompt": "yes",
        }
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
    Generate image via ModelsLab using the dedicated Ghibli Diffusion model.
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
            "model_id": "ghibli",
            "prompt": prompt,
            "negative_prompt": neg,
            "init_image": image_data,
            "strength": options.get("strength", 0.65),
            "base64": True,
            "width": width,
            "height": height,
            "samples": "1",
            "num_inference_steps": "30",
            "guidance_scale": 7.5,
            "safety_checker": "no",
            "enhance_prompt": "yes",
        }
        logger.info("ModelsLab Ghibli img2img: %s...", prompt[:60])
        resp = requests.post("https://modelslab.com/api/v6/images/img2img", json=payload, timeout=90)
        log_prefix = "ghibli_i2i"
    else:
        payload = {
            "key": api_key,
            "model_id": "ghibli",
            "prompt": prompt,
            "negative_prompt": neg,
            "width": width,
            "height": height,
            "samples": "1",
            "num_inference_steps": "30",
            "guidance_scale": 7.5,
            "safety_checker": "no",
            "enhance_prompt": "yes",
        }
        logger.info("ModelsLab Ghibli text2img: %s...", prompt[:60])
        resp = requests.post("https://modelslab.com/api/v6/images/text2img", json=payload, timeout=90)
        log_prefix = "ghibli"

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

    # Modelslab Ghibli — dedicated ghibli-diffusion model for authentic Ghibli art style
    if provider == "modelslab_ghibli":
        try:
            return _generate_modelslab_ghibli(prompt, options)
        except Exception as e:
            logger.warning("ModelsLab Ghibli failed: %s. Falling back to FLUX...", e)
            try:
                return _generate_modelslab(prompt, options)
            except Exception:
                return _fallback_all_or_stub(prompt, dict(options, image_provider="modelslab_ghibli"))

    # Modelslab FLUX — high quality scene images
    if provider == "modelslab":
        try:
            return _generate_modelslab(prompt, options)
        except Exception as e:
            logger.warning("Modelslab failed: %s. Falling back...", e)
            return _fallback_all_or_stub(prompt, dict(options, image_provider="modelslab"))

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


def generate_scene_image(
    prompt: str,
    avatar_path: str | None = None,
    image_provider: str | None = None,
) -> str:
    """
    Generate a scene image from a text prompt.
    
    For character consistency, if avatar_path is provided, uses img2img
    with the avatar image to maintain visual consistency across scenes.
    
    Args:
        prompt: Image generation prompt (already contains character description)
        avatar_path: Path to avatar image for img2img consistency (optional)
        image_provider: Optional provider override
    
    Returns:
        Relative path to generated image (e.g., "images/local_123456.jpg")
    """
    options = {}
    provider = _provider(image_provider)
    if provider != "stub":
        options["image_provider"] = provider
    
    # If avatar_path provided, use img2img for consistency
    if avatar_path:
        try:
            from pathlib import Path
            from django.conf import settings
            avatar_full_path = Path(settings.MEDIA_ROOT) / avatar_path
            if avatar_full_path.exists():
                with open(avatar_full_path, "rb") as f:
                    img_bytes = f.read()
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    options["image_data"] = f"data:image/jpeg;base64,{img_base64}"
                    options["strength"] = 0.7  # Balance consistency with scene changes
                    logger.info(f"Using avatar for scene consistency: {avatar_path}")
        except Exception as e:
            logger.warning(f"Failed to load avatar for img2img: {e}")
    
    return generate_image(prompt, options)
