"""
Avatar processing utilities with Ghibli-style conversion.

Responsibilities:
- Detect a face using OpenCV Haar Cascade (when available)
- Convert to Ghibli-style using AI (Replicate/OpenAI)
- Resize the avatar image
- Save processed avatar into /media/avatars/
- Return the relative file path for later use in stories
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any
import time
import base64
from io import BytesIO
from PIL import Image

from django.conf import settings

logger = logging.getLogger(__name__)


def convert_to_ghibli_style(image_path: Path) -> Path:
    """
    Convert an image to Ghibli/anime style using AI.

    Priority:
    1. ModelsLab dreamshaper-8 img2img (base64, strength 0.45) — preserves identity best
    2. ModelsLab dreamshaper-8 text2img — only if Gemini Vision can describe the person
    3. Pollinations.AI FLUX text2img — only if Gemini Vision can describe the person
    4. Return original unchanged (never generates from a generic fallback description)
    """
    if isinstance(image_path, Path) and image_path.is_absolute():
        full_image_path = image_path
    else:
        full_image_path = Path(settings.MEDIA_ROOT) / image_path

    avatars_dir = Path(settings.MEDIA_ROOT) / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    ext = ".jpg"

    neg_prompt = (
        "photorealistic, realistic photograph, realistic skin texture, soft shading, "
        "smooth gradient, subsurface scattering, 3d render, CGI, photography, real photo, "
        "blurry, low quality, bad anatomy, western cartoon, extra limbs, distorted face, "
        "ugly, deformed, angry expression, scary face, frightening, (gender swap:1.5), "
        "(wrong gender:1.5), (different person:1.4), "
        "adult, man, woman, teenager, young adult, mature face, old face, "
        "muscular, broad shoulders, adult body proportions, tall figure, "
        "different hair color, different hair style, "
        "different clothes, different outfit, outfit change, wardrobe change, naked, "
        "black background, dark background, grey background, busy background, "
        "cluttered background, colorful background, complex background, "
        "black areas, black fill, dark fill, "
        "text, watermark, cropped, cut off, out of frame"
    )

    with open(full_image_path, "rb") as _f:
        _avatar_bytes = _f.read()

    # Pre-resize to exactly 512×512 center-cropped square before any upload.
    # This prevents ModelsLab from receiving a large photo and generating a
    # half-black canvas (which happened because the model couldn't fill the
    # composition when given a 1080px+ close-up portrait).
    try:
        _pil_src = Image.open(BytesIO(_avatar_bytes)).convert("RGB")
        _w, _h = _pil_src.size
        _min_dim = min(_w, _h)
        _left = (_w - _min_dim) // 2
        _top = (_h - _min_dim) // 2
        _pil_src = _pil_src.crop((_left, _top, _left + _min_dim, _top + _min_dim))
        _pil_src = _pil_src.resize((512, 512), Image.Resampling.LANCZOS)
        _resized_buf = BytesIO()
        _pil_src.save(_resized_buf, format="JPEG", quality=92)
        _upload_bytes = _resized_buf.getvalue()
        logger.info("Ghibli: pre-resized avatar to 512×512 (%d bytes)", len(_upload_bytes))
    except Exception as _re:
        logger.warning("Ghibli: pre-resize failed (%s), using original bytes", _re)
        _upload_bytes = _avatar_bytes

    from .image_engine import log_api_error as _log_api_err

    # Style tokens come FIRST — SD/anything-v5 is CLIP-based and weights early tokens most.
    # Removing "preserve all facial features" — it directly fights anime stylization and
    # causes near-zero style change (pixel diff stays under 70). High-level identity cues
    # (gender, hair color, clothing colors) are enough for character recognition.
    _img2img_prompt = (
        "(anime illustration:1.5), (cel shading:1.4), (flat color shading:1.3), "
        "(thick black outlines:1.3), (simplified cartoon face:1.2), "
        "(large expressive anime eyes:1.4), vibrant saturated colors, "
        "clean linework, bright vivid colors, "
        "(pure white background:1.5), (white background only:1.4), simple background, "
        "(young child:1.4), (toddler face:1.3), small child, soft round face, "
        "same gender as reference, same hair color and length, same hair texture, "
        "same clothing colors and style, same skin tone, gentle smile, "
        "Hayao Miyazaki hand-drawn illustration style, masterpiece, best quality"
    )

    def _poll_for_result(req, api_key, prediction_id, eta):
        """Poll ModelsLab fetch endpoint until image URL is returned or error.
        FaceGen can take 90-120s — 30s initial wait + 25×5s = 155s total budget.
        """
        time.sleep(min(eta, 30))
        for _p in range(25):
            fdata = req.post(
                "https://modelslab.com/api/v6/images/fetch",
                json={"key": api_key, "request_id": prediction_id},
                timeout=30,
            ).json()
            if fdata.get("status") == "success":
                return (fdata.get("output") or [None])[0]
            if fdata.get("status") == "error":
                raise RuntimeError(f"ModelsLab poll error: {fdata.get('message')}")
            logger.info("FaceGen avatar still processing (attempt %d/25)...", _p + 1)
            time.sleep(5)
        return None

    def _save_from_url(req, image_url, out_rel):
        """Download image from URL and save to media dir. Retries if CDN not ready yet."""
        for _attempt in range(6):
            img_resp = req.get(image_url, timeout=60)
            if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                try:
                    _test = Image.open(BytesIO(img_resp.content))
                    _test.verify()
                except Exception as _ve:
                    raise RuntimeError(f"Downloaded file is not a valid image: {_ve}")
                out_abs = Path(settings.MEDIA_ROOT) / out_rel
                out_abs.write_bytes(img_resp.content)
                return out_rel
            # CDN not ready yet — wait and retry (6 × 15s = 90s total budget)
            logger.info("Ghibli download attempt %d/6: status=%d size=%d — waiting for CDN...",
                        _attempt + 1, img_resp.status_code, len(img_resp.content))
            time.sleep(15)
        raise RuntimeError(f"Download failed after retries: status={img_resp.status_code} size={len(img_resp.content)}")

    def _run_modelslab_img2img(req, api_key, init_image_val, use_base64_param=False):
        """Run ModelsLab img2img with given init_image. Returns image URL or raises."""
        payload = {
            "key": api_key,
            "model_id": "anything-v5",
            "prompt": _img2img_prompt,
            "negative_prompt": neg_prompt,
            "init_image": init_image_val,
            "strength": 0.55,
            "width": "512",
            "height": "512",
            "samples": "1",
            "num_inference_steps": "30",
            "guidance_scale": 8.0,
            "safety_checker": "no",
            "enhance_prompt": "yes",
        }
        if use_base64_param:
            payload["base64"] = "true"
        resp = req.post("https://modelslab.com/api/v6/images/img2img", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        _log_api_err(f"Ghibli img2img response: status={data.get('status')} msg={str(data.get('message',''))[:200]}")
        logger.info("Ghibli img2img: status=%s msg=%s", data.get("status"), str(data.get("message", ""))[:120])
        if data.get("status") == "success":
            url = (data.get("output") or [None])[0]
            if url:
                return url
            raise RuntimeError(f"status=success but no output URL. Response: {data}")
        if data.get("status") == "processing":
            url = _poll_for_result(req, api_key, data.get("id"), int(data.get("eta", 20)))
            if url:
                return url
            raise RuntimeError("Polling timed out — no image URL returned")
        raise RuntimeError(f"ModelsLab img2img error: {data.get('message', data)}")

    # --- Path 1: ModelsLab anything-v5 img2img (URL upload, no webhook required) ---
    # Upload the pre-resized avatar to a public host and pass as init_image URL.
    # Avoids FaceGen's webhook/timeout issues while still preserving identity via img2img.
    if os.getenv("MODELSLAB_API_KEY"):
        try:
            import requests as _req1
            from .image_engine import _upload_image_for_api
            _api_key1 = os.getenv("MODELSLAB_API_KEY", "").strip()
            _init_url = _upload_image_for_api(_upload_bytes, cache_key=f"ghibli_{stem}")
            _img2img_result_url = _run_modelslab_img2img(_req1, _api_key1, _init_url)
            if _img2img_result_url:
                out_rel = Path("avatars") / f"{stem}_ghibli_ml{ext}"
                result = _save_from_url(_req1, _img2img_result_url, out_rel)
                logger.info("Ghibli via anything-v5 img2img: %s", result)
                return result
        except Exception as _p1_err:
            _log_api_err(f"img2img conversion failed: {type(_p1_err).__name__}: {_p1_err}")
            logger.warning("img2img conversion failed: %s — trying text2img", _p1_err)

    # --- Path 2: ModelsLab text2img with vision-based person description ---
    # Uses Groq Vision first (already working), falls back to Gemini Vision.
    # Skipped entirely if no vision provider can describe the person
    # (avoids generating a random anime image from a fake "friendly child" description).
    if os.getenv("MODELSLAB_API_KEY"):
        _char_desc = None

        # 2a: Groq Vision (llama-3.2-11b-vision-instruct)
        _groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if _groq_key and not _char_desc:
            try:
                from groq import Groq as _Groq
                _gclient = _Groq(api_key=_groq_key)
                _b64_for_groq = base64.b64encode(_upload_bytes).decode()
                _gr = _gclient.chat.completions.create(
                    model="llama-3.2-11b-vision-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{_b64_for_groq}"}},
                            {"type": "text", "text": (
                                "Describe this person's appearance in 30 words max, comma-separated. "
                                "Include: gender, approximate age, hair color and style, eye color, skin tone, "
                                "shirt/top (color + type), pants/skirt (color + type), any visible accessories. "
                                "Example: 'boy, 8 years old, short black hair, dark brown eyes, olive skin, "
                                "yellow striped t-shirt, dark blue jeans, white sneakers'. "
                                "Output ONLY the comma-separated list, no sentences."
                            )},
                        ],
                    }],
                    max_tokens=80,
                )
                _desc = _gr.choices[0].message.content.strip().rstrip(".")
                if _desc:
                    _char_desc = _desc
                    logger.info("Ghibli: Groq Vision described person as: %s", _char_desc[:80])
            except Exception as _ge:
                logger.warning("Ghibli: Groq Vision failed: %s", _ge)

        # 2b: Gemini Vision fallback
        if not _char_desc and os.getenv("GEMINI_API_KEY"):
            try:
                from google import genai as _g2
                from google.genai import types as _gt2
                _dc = _g2.Client(api_key=os.getenv("GEMINI_API_KEY"))
                _dr = _dc.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        _gt2.Part.from_bytes(data=_upload_bytes, mime_type="image/jpeg"),
                        "Describe this person's appearance in 30 words max, comma-separated traits. "
                        "Include: gender, approximate age, hair color and style, eye color, skin tone, "
                        "shirt/top (color + type), pants/skirt (color + type), any visible accessories. "
                        "Example: 'boy, 8 years old, short black hair, dark brown eyes, olive skin, yellow striped t-shirt, dark blue jeans'. "
                        "Output ONLY the comma-separated list, no sentences.",
                    ],
                )
                if _dr.text:
                    _char_desc = _dr.text.strip().rstrip(".")
                    logger.info("Ghibli: Gemini Vision described person as: %s", _char_desc[:80])
            except Exception as _de:
                logger.warning("Ghibli: Gemini Vision also failed: %s", _de)

        if _char_desc:
            try:
                import requests as _req2
                _api_key2 = os.getenv("MODELSLAB_API_KEY", "").strip()
                _t2i_prompt = (
                    f"(masterpiece:1.2), best quality, highres, "
                    f"(anime style:1.3), (studio ghibli style:1.2), "
                    f"detailed anime portrait of {_char_desc}, "
                    "Hayao Miyazaki hand-drawn illustration, large expressive anime eyes, "
                    "soft warm lighting, clean thick outlines, vibrant saturated colors, "
                    "gentle expression, pure white background"
                )
                _t2i_payload = {
                    "key": _api_key2,
                    "model_id": "anything-v5",
                    "prompt": _t2i_prompt,
                    "negative_prompt": neg_prompt,
                    "width": "512",
                    "height": "512",
                    "samples": "1",
                    "num_inference_steps": "20",
                    "guidance_scale": 7.5,
                    "safety_checker": "no",
                    "enhance_prompt": "no",
                }
                logger.info("Ghibli: text2img with description: %s", _char_desc[:60])
                _t2i_resp = _req2.post(
                    "https://modelslab.com/api/v6/images/text2img",
                    json=_t2i_payload, timeout=90,
                )
                _t2i_resp.raise_for_status()
                _t2i_data = _t2i_resp.json()
                _log_api_err(f"Ghibli text2img: status={_t2i_data.get('status')} msg={str(_t2i_data.get('message',''))[:150]}")

                _t2i_url = None
                if _t2i_data.get("status") == "success":
                    _t2i_url = (_t2i_data.get("output") or [None])[0]
                elif _t2i_data.get("status") == "processing":
                    _t2i_url = _poll_for_result(_req2, _api_key2, _t2i_data.get("id"), int(_t2i_data.get("eta", 20)))
                elif _t2i_data.get("status") == "error":
                    raise RuntimeError(f"text2img error: {_t2i_data.get('message', _t2i_data)}")

                if _t2i_url:
                    out_rel = Path("avatars") / f"{stem}_ghibli_ml_t2i{ext}"
                    result = _save_from_url(_req2, _t2i_url, out_rel)
                    logger.info("Ghibli via ModelsLab text2img: %s", result)
                    return result
            except Exception as _e2:
                _log_api_err(f"Ghibli text2img failed: {type(_e2).__name__}: {_e2}")
                logger.warning("Ghibli text2img failed: %s", _e2)
        else:
            logger.info("Ghibli: no vision description available — skipping text2img (avoids generating unrelated image)")

    # --- Path 3: Pollinations.AI FLUX — only with a real person description ---
    _poll_char_desc = _char_desc if os.getenv("MODELSLAB_API_KEY") else None
    if not _poll_char_desc:
        # Try Groq Vision if we haven't already
        _groq_key3 = os.getenv("GROQ_API_KEY", "").strip()
        if _groq_key3:
            try:
                from groq import Groq as _Groq3
                _gc3 = _Groq3(api_key=_groq_key3)
                _b64_p = base64.b64encode(_upload_bytes).decode()
                _gr3 = _gc3.chat.completions.create(
                    model="llama-3.2-11b-vision-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64_p}"}},
                            {"type": "text", "text": "Describe this person: gender, age, hair, eyes, skin. 20 words max, comma-separated list only."},
                        ],
                    }],
                    max_tokens=80,
                )
                _d3 = _gr3.choices[0].message.content.strip().rstrip(".")
                if _d3:
                    _poll_char_desc = _d3
            except Exception:
                pass

    if _poll_char_desc:
        try:
            logger.info("Ghibli: trying Pollinations fallback with vision description...")
            poll_prompt = (
                f"Studio Ghibli anime style close-up portrait of {_poll_char_desc}, "
                "Hayao Miyazaki hand-drawn illustration, large expressive eyes, "
                "soft warm lighting, clean thick outlines, vibrant saturated colors, "
                "detailed facial features, gentle expression, pure white background, "
                "masterpiece quality, anime art style"
            )
            from .image_engine import _generate_pollinations
            result_rel = _generate_pollinations(
                poll_prompt,
                {"width": 512, "height": 512, "pollinations_model": "flux"},
            )
            if result_rel and "placeholder" not in result_rel:
                import shutil as _shutil
                src = Path(settings.MEDIA_ROOT) / result_rel
                out_rel = Path("avatars") / f"{stem}_ghibli_poll{ext}"
                out_abs = Path(settings.MEDIA_ROOT) / out_rel
                _shutil.copy2(src, out_abs)
                logger.info("Ghibli via Pollinations.AI: %s", out_rel)
                return out_rel
        except Exception as e:
            logger.warning("Ghibli Pollinations fallback failed: %s", e)
    else:
        logger.info("Ghibli: no vision description available — skipping Pollinations fallback")

    # --- Final fallback ---
    logger.info("Ghibli: all paths failed, returning original image")
    return image_path


def process_avatar(
    uploaded_file: Any,
    user_identifier: str | None = None,
    convert_ghibli: bool = False,
) -> tuple[str, str]:
    """
    Process an uploaded avatar image with optional Ghibli-style conversion.

    - Saves into MEDIA_ROOT / "avatars"
    - Attempts basic face detection using OpenCV Haar Cascade (optional)
    - Saves the original photo as ``_orig.jpg`` (used by describe_avatar — vision
      models extract real traits from photos, not from anime drawings)
    - Optionally converts to Ghibli style and saves result as ``_final.jpg``
    - Returns (orig_path, final_path) where:
        * orig_path  — relative path to original photo (for describe_avatar)
        * final_path — relative path to Ghibli output, or orig if conversion failed
                       (for FLUX img2img scene consistency)
    """
    from io import BytesIO
    from PIL import Image
    
    avatars_dir = Path(settings.MEDIA_ROOT) / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = user_identifier or "avatar"
    timestamp = int(time.time() * 1000)
    
    # Handle both file uploads and base64 data
    if isinstance(uploaded_file, str):
        # Base64 string from webcam
        try:
            # Check if it's a URL instead of data
            if uploaded_file.startswith('http') or uploaded_file.startswith('/media/') or uploaded_file.startswith('blob:'):
                logger.warning(f"Avatar processing received a URL or Blob instead of data: {uploaded_file}")
                # If it's already a media path, return it relative to media root
                media_url = getattr(settings, 'MEDIA_URL', '/media/')
                if uploaded_file.startswith(media_url):
                    return uploaded_file.replace(media_url, "", 1)
                return uploaded_file

            # Remove data URL prefix if present
            if ',' in uploaded_file:
                uploaded_file = uploaded_file.split(',')[1]
            
            # Clean string
            uploaded_file = uploaded_file.strip().replace("\n", "").replace("\r", "")
            
            # Robust padding: base64 strings must have a length multiple of 4
            missing_padding = len(uploaded_file) % 4
            if missing_padding:
                uploaded_file += '=' * (4 - missing_padding)
                
            image_data = base64.b64decode(uploaded_file)
            img = Image.open(BytesIO(image_data)).convert("RGB")
            ext = ".jpg"
        except Exception as e:
            logger.exception(f"Failed to decode base64 image: {e}")
            # Also log to our debug file
            from .image_engine import log_api_error
            log_api_error(f"Avatar Base64 Decode Error: {e}. String length: {len(uploaded_file)}. Content start: {uploaded_file[:100]}")
            # If it's very short, log the whole thing
            if len(uploaded_file) < 500:
                log_api_error(f"Full problematic string: {uploaded_file}")
            raise ValueError(f"Invalid image data: {str(e)}")
    else:
        # File upload
        image_data = uploaded_file.read()
        img = Image.open(BytesIO(image_data)).convert("RGB")
        ext = os.path.splitext(uploaded_file.name)[1].lower() if hasattr(uploaded_file, 'name') else ".jpg"
    
    # Save initial image
    initial_filename = f"{base_name}_avatar_{timestamp}{ext}"
    initial_path = avatars_dir / initial_filename
    img.save(initial_path, format="JPEG", quality=90)
    
    # Try OpenCV face detection (optional best-effort)
    try:
        import cv2
        import numpy as np
        
        # Convert PIL image to OpenCV format
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            # Add padding around face
            padding = int(w * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(cv_img.shape[1] - x, w + 2 * padding)
            h = min(cv_img.shape[0] - y, h + 2 * padding)
            cv_face = cv_img[y : y + h, x : x + w]
            img = Image.fromarray(cv2.cvtColor(cv_face, cv2.COLOR_BGR2RGB))
            # Save cropped version
            img.save(initial_path, format="JPEG", quality=90)
    except Exception:
        # Any error in OpenCV flow -> fall back to original image
        pass
    
    # Keep a copy of the current PIL image before Ghibli conversion
    # so we can fall back to it if the converted file is unreadable.
    original_img = img.copy()

    # Save the original photo (pre-Ghibli, post-crop) at 512×512.
    # describe_avatar MUST run on this file — vision models extract real traits
    # (hair colour, outfit, skin tone) from photos, NOT from anime drawings.
    orig_filename = f"{base_name}_avatar_{timestamp}_orig.jpg"
    orig_abs = avatars_dir / orig_filename
    original_img.resize((512, 512), Image.Resampling.LANCZOS).save(
        orig_abs, format="JPEG", quality=95
    )
    orig_rel = f"avatars/{orig_filename}"
    logger.info("Saved original avatar for describe_avatar: %s", orig_rel)

    # Convert to Ghibli style if requested
    if convert_ghibli:
        try:
            ghibli_path = convert_to_ghibli_style(initial_path)
            if ghibli_path != initial_path:
                initial_path = ghibli_path
                logger.info(f"Using Ghibli-style avatar: {ghibli_path}")
        except Exception as e:
            logger.warning(f"Ghibli conversion failed, using original: {e}")

    # Resize the (possibly Ghibli-converted) image to 512×512 for the final file
    if not Path(initial_path).is_absolute():
        load_path = Path(settings.MEDIA_ROOT) / initial_path
    else:
        load_path = Path(initial_path)

    try:
        img = Image.open(load_path).convert("RGB")
    except Exception as e:
        logger.warning(
            "Could not open processed avatar at %s (%s). Using in-memory original.",
            load_path, e,
        )
        img = original_img

    img = img.resize((512, 512), Image.Resampling.LANCZOS)

    # Final save — contains Ghibli output (if conversion succeeded) or original
    final_filename = f"{base_name}_avatar_{timestamp}_final.jpg"
    final_path_abs = avatars_dir / final_filename
    img.save(final_path_abs, format="JPEG", quality=95)
    final_rel = f"avatars/{final_filename}"

    # Return (original_photo_path, ghibli_or_final_path)
    return orig_rel, final_rel


def process_webcam_avatar(
    image_data: str,
    user_identifier: str | None = None,
    convert_ghibli: bool = True,
) -> tuple[str, str]:
    """
    Process a webcam-captured image (base64) with Ghibli conversion.

    Returns:
        (orig_path, ghibli_or_final_path) — same contract as process_avatar.
    """
    return process_avatar(image_data, user_identifier, convert_ghibli=convert_ghibli)


def create_default_avatar(user_identifier: str | None = None) -> str:
    """
    SRDS FR-8: Provide a default generic avatar when the user does not upload an image.

    We create a simple cartoon-style placeholder image locally (no AI keys required).
    """
    from PIL import Image, ImageDraw

    avatars_dir = Path(settings.MEDIA_ROOT) / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)

    # Stable filename so we don't regenerate repeatedly.
    filename = "default_avatar.jpg"
    final_path = avatars_dir / filename

    if not final_path.exists():
        img = Image.new("RGB", (256, 256), (255, 245, 220))
        draw = ImageDraw.Draw(img)

        # Simple head + face
        draw.ellipse((32, 40, 224, 232), fill=(255, 224, 189), outline=(240, 160, 120), width=4)
        # Eyes
        draw.ellipse((78, 105, 110, 137), fill=(60, 60, 60))
        draw.ellipse((146, 105, 178, 137), fill=(60, 60, 60))
        # Smile
        draw.arc((90, 130, 166, 186), start=20, end=160, fill=(200, 90, 90), width=8)

        # Optional label (kept subtle)
        if user_identifier:
            try:
                short = str(user_identifier).strip()[:10]
                draw.text((20, 10), short, fill=(120, 120, 120))
            except Exception:
                pass

        img.save(final_path, format="JPEG", quality=95)

    return f"avatars/{filename}"


def describe_avatar(avatar_path_rel: str) -> str:
    """
    Describe the avatar's appearance (face + clothing) for use as the locked
    character anchor injected into every scene image prompt.

    Tries Groq Vision first (quota-safe), falls back to Gemini Vision.
    Returns a comma-separated trait string including outfit details so that
    every scene prompt can enforce the same clothing across the full story.
    """
    full_path = Path(settings.MEDIA_ROOT) / avatar_path_rel
    if not full_path.exists():
        return "a friendly child"

    with open(full_path, "rb") as f:
        image_bytes = f.read()

    _vision_prompt = (
        "Describe this child's appearance for an anime illustration. "
        "Output ONLY a comma-separated list of visual traits, maximum 35 words. "
        "IMPORTANT: The FIRST word must be either 'girl' or 'boy' — never skip this. "
        "Then include: approximate age, "
        "hair color AND texture — use exactly one word from (straight/curly/wavy/afro/coily), "
        "hair length (short/medium/long), eye color, skin tone, "
        "shirt/top (color + type), pants/skirt (color + type), shoes, any visible accessories. "
        "Example: 'girl, 7 years old, short curly black hair, dark brown eyes, olive skin, "
        "white shirt, dark blue jeans, white sneakers'. "
        "Do NOT write sentences — only the comma-separated list. Start with girl or boy."
    )

    def _clean(text: str) -> str:
        desc = text.strip().rstrip(".")
        for prefix in (
            "the child is ", "this child is ", "a child with ",
            "they are wearing ", "they have ", "child has ",
        ):
            if desc.lower().startswith(prefix):
                desc = desc[len(prefix):]
                break
        return desc

    # Primary: Groq Vision (llama-3.2-11b-vision-instruct) — quota-safe
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            from groq import Groq as _Groq
            _b64 = base64.b64encode(image_bytes).decode()
            _gc = _Groq(api_key=groq_key)
            _gr = _gc.chat.completions.create(
                model="llama-3.2-11b-vision-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{_b64}"}},
                        {"type": "text", "text": _vision_prompt},
                    ],
                }],
                max_tokens=100,
            )
            desc = (_gr.choices[0].message.content or "").strip()
            if desc:
                logger.info("describe_avatar (Groq): %s", desc[:100])
                return _clean(desc)
        except Exception as e:
            logger.warning("describe_avatar Groq Vision failed: %s", e)

    # Fallback: Gemini Vision
    if os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai as _gai
            from google.genai import types as _gt
            _dc = _gai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            _dr = _dc.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    _gt.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    _vision_prompt,
                ],
            )
            if _dr.text:
                logger.info("describe_avatar (Gemini): %s", _dr.text[:100])
                return _clean(_dr.text)
        except Exception as e:
            logger.warning("describe_avatar Gemini Vision failed: %s", e)

    return "a friendly child"
