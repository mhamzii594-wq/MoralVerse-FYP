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
    Convert an image to Ghibli-style using AI.

    Priority:
    1. Modelslab FLUX Kontext Dev img2img — passes photo directly, superior character retention
    2. Gemini Image Generation (gemini-2.0-flash-preview-image-generation) — img2img fallback
    3. Gemini Vision (describe) + Pollinations.AI FLUX (generate) — free text fallback
    4. Return original unchanged
    """
    if isinstance(image_path, Path) and image_path.is_absolute():
        full_image_path = image_path
    else:
        full_image_path = Path(settings.MEDIA_ROOT) / image_path

    avatars_dir = Path(settings.MEDIA_ROOT) / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    ext = ".jpg"

    ghibli_prompt = (
        "Studio Ghibli anime portrait, same face same hair same facial features as the person in the photo, "
        "Hayao Miyazaki hand-drawn illustration style, large expressive eyes, soft warm lighting, "
        "clean thick outlines, vibrant saturated colors, gentle smile, pure white background, "
        "masterpiece quality anime art, no text, no watermark"
    )
    neg_prompt = (
        "photorealistic, realistic photograph, 3d render, CGI, blurry, low quality, "
        "bad anatomy, western cartoon, extra limbs, distorted face, ugly, deformed"
    )

    # --- Path 1: Modelslab FLUX Kontext Dev img2img ---
    if os.getenv("MODELSLAB_API_KEY"):
        try:
            import requests as _req

            api_key = os.getenv("MODELSLAB_API_KEY", "").strip()
            with open(full_image_path, "rb") as _f:
                img_b64 = base64.b64encode(_f.read()).decode("utf-8")

            payload = {
                "key": api_key,
                "model_id": "flux-kontext-dev",
                "prompt": ghibli_prompt,
                "negative_prompt": neg_prompt,
                "init_image": img_b64,
                "strength": 0.65,
                "base64": True,
                "width": "512",
                "height": "512",
                "samples": "1",
                "num_inference_steps": "30",
                "guidance_scale": 7.5,
                "safety_checker": "no",
                "enhance_prompt": "yes",
            }

            logger.info("Ghibli: calling Modelslab FLUX Kontext Dev img2img...")
            resp = _req.post(
                "https://modelslab.com/api/v6/images/img2img",
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()

            image_url = None
            if data.get("status") == "success":
                image_url = (data.get("output") or [None])[0]
            elif data.get("status") == "processing":
                prediction_id = data.get("id")
                eta = int(data.get("eta", 15))
                logger.info("Ghibli Kontext: processing (eta=%ss)...", eta)
                time.sleep(min(eta, 15))
                for _ in range(12):
                    fetch = _req.post(
                        "https://modelslab.com/api/v6/images/fetch",
                        json={"key": api_key, "request_id": prediction_id},
                        timeout=30,
                    )
                    fdata = fetch.json()
                    if fdata.get("status") == "success":
                        image_url = (fdata.get("output") or [None])[0]
                        break
                    if fdata.get("status") == "error":
                        raise RuntimeError(f"Kontext poll error: {fdata.get('message')}")
                    time.sleep(5)
            elif data.get("status") == "error":
                raise RuntimeError(f"Modelslab error: {data.get('message', data)}")

            if image_url:
                img_resp = _req.get(image_url, timeout=60)
                if img_resp.status_code == 200 and len(img_resp.content) > 5000:
                    out_rel = Path("avatars") / f"{stem}_ghibli_kontext{ext}"
                    out_abs = Path(settings.MEDIA_ROOT) / out_rel
                    out_abs.write_bytes(img_resp.content)
                    logger.info("Ghibli via Modelslab Kontext: %s", out_rel)
                    return out_rel
            raise RuntimeError("Kontext returned no valid image")
        except Exception as e:
            logger.warning("Ghibli Modelslab Kontext failed: %s. Trying Pollinations fallback.", e)

    # --- Path 2: Gemini Image Generation (img2img via gemini-2.0-flash-preview-image-generation) ---
    if os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai as _genai
            from google.genai import types as _gtypes

            logger.info("Ghibli: trying Gemini Image Generation img2img...")
            _gclient = _genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            with open(full_image_path, "rb") as _f:
                _img_bytes = _f.read()

            gemini_img_model = (
                os.getenv("GEMINI_IMAGE_MODEL") or "gemini-2.0-flash-preview-image-generation"
            )
            _gresp = _gclient.models.generate_content(
                model=gemini_img_model,
                contents=[
                    _gtypes.Part.from_bytes(data=_img_bytes, mime_type="image/jpeg"),
                    (
                        "Convert this photo into a Studio Ghibli anime-style portrait. "
                        "Keep exactly the same face shape, hair color, hair style, eye color, "
                        "and facial features as the person in the photo. "
                        "Use Hayao Miyazaki hand-drawn illustration style: large expressive eyes, "
                        "soft warm lighting, clean thick outlines, vibrant saturated colors, "
                        "gentle smile, pure white background. Output only the image."
                    ),
                ],
                config=_gtypes.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            for _part in (_gresp.candidates or [{}])[0].content.parts:
                if hasattr(_part, "inline_data") and _part.inline_data:
                    _raw = _part.inline_data.data
                    if len(_raw) > 5000:
                        out_rel = Path("avatars") / f"{stem}_ghibli_gemini{ext}"
                        out_abs = Path(settings.MEDIA_ROOT) / out_rel
                        out_abs.write_bytes(_raw)
                        logger.info("Ghibli via Gemini Image Gen: %s", out_rel)
                        return out_rel
            raise RuntimeError("Gemini Image Gen returned no image data")
        except Exception as e:
            logger.warning("Ghibli Gemini Image Gen failed: %s. Trying Pollinations fallback.", e)

    # --- Path 3: Gemini Vision (describe) + Pollinations.AI FLUX (generate) ---
    try:
        logger.info("Ghibli: describing child with Gemini Vision...")
        char_desc = "a friendly child"

        if os.getenv("GEMINI_API_KEY"):
            from google import genai
            from google.genai import types as _gtypes
            _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            with open(full_image_path, "rb") as _f:
                _img_bytes = _f.read()
            desc_resp = _client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    _gtypes.Part.from_bytes(data=_img_bytes, mime_type="image/jpeg"),
                    "Describe this child for a Ghibli anime illustration. "
                    "Output ONLY a comma-separated trait list, max 25 words. "
                    "Include: gender, approximate age, hair color and style, eye color, "
                    "shirt color and type, pants/skirt color. "
                    "Example: 'boy, 8 years old, short black bowl-cut hair, dark brown eyes, "
                    "yellow striped t-shirt, blue jeans'. "
                    "Do NOT write sentences — only the comma-separated list.",
                ],
            )
            if desc_resp.text:
                char_desc = desc_resp.text.strip().rstrip(".")
                logger.info("Ghibli: child described as: %s", char_desc[:80])

        poll_prompt = (
            f"Studio Ghibli anime style close-up portrait of {char_desc}, "
            "Hayao Miyazaki hand-drawn illustration, large expressive eyes, "
            "soft warm lighting, clean thick outlines, vibrant saturated colors, "
            "detailed facial features, gentle smile, pure white background, "
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

    # --- Final fallback ---
    logger.info("Ghibli: all paths failed, returning original image")
    return image_path


def process_avatar(uploaded_file: Any, user_identifier: str | None = None, convert_ghibli: bool = False) -> str:
    """
    Process an uploaded avatar image with optional Ghibli-style conversion.
    
    - Saves into MEDIA_ROOT / "avatars"
    - Attempts basic face detection using OpenCV Haar Cascade
    - Optionally converts to Ghibli style
    - Resizes to a square thumbnail (256x256)
    - Returns a relative media path, e.g. "avatars/ali_avatar.jpg"
    
    Args:
        uploaded_file: File object or image data
        user_identifier: Optional identifier for filename
        convert_ghibli: If True, convert to Ghibli style
        
    Returns:
        Relative path to processed avatar
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
    
    # Convert to Ghibli style if requested
    if convert_ghibli:
        try:
            ghibli_path = convert_to_ghibli_style(initial_path)
            if ghibli_path != initial_path:
                # Use Ghibli version
                initial_path = ghibli_path
                logger.info(f"Using Ghibli-style avatar: {ghibli_path}")
        except Exception as e:
            logger.warning(f"Ghibli conversion failed, using original: {e}")
    
    # Resize to 256x256
    # Ensure initial_path is absolute for opening
    if not Path(initial_path).is_absolute():
        load_path = Path(settings.MEDIA_ROOT) / initial_path
    else:
        load_path = initial_path
        
    img = Image.open(load_path)
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    
    # Final save
    final_filename = f"{base_name}_avatar_{timestamp}_final.jpg"
    final_path = avatars_dir / final_filename
    img.save(final_path, format="JPEG", quality=95)
    
    # Return relative path
    return f"avatars/{final_filename}"


def process_webcam_avatar(image_data: str, user_identifier: str | None = None, convert_ghibli: bool = True) -> str:
    """
    Process a webcam-captured image (base64) with Ghibli conversion.
    
    Args:
        image_data: Base64 encoded image data
        user_identifier: Optional identifier for filename
        convert_ghibli: Convert to Ghibli style (default: True)
        
    Returns:
        Relative path to processed avatar
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
    Use Gemini Vision to generate a Ghibli-style, SD-prompt-ready character
    description.  Returns a comma-separated trait list used as the locked
    character anchor injected into every scene image prompt.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return "a friendly child"

    try:
        from google import genai
        from google.genai import types
        from pathlib import Path

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        full_path = Path(settings.MEDIA_ROOT) / avatar_path_rel
        if not full_path.exists():
            return "a friendly child"

        with open(full_path, "rb") as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "Describe this child for a Ghibli anime illustration. "
                "Output ONLY a comma-separated list of visual traits, maximum 20 words. "
                "Include: gender, approximate age, hair color and style, eye color, "
                "shirt/top (color + type), pants/skirt (color + type). "
                "Example: 'boy, 8 years old, short black bowl-cut hair, dark brown eyes, yellow striped t-shirt, blue jeans'. "
                "Do NOT write sentences — only the comma-separated list.",
            ],
        )

        if response.text:
            desc = response.text.strip().rstrip(".")
            for prefix in (
                "the child is ", "this child is ", "a child with ",
                "they are wearing ", "they have ", "child has ",
            ):
                if desc.lower().startswith(prefix):
                    desc = desc[len(prefix):]
                    break
            return desc

    except Exception as e:
        logger.warning(f"Failed to describe avatar: {e}")

    return "a friendly child"
