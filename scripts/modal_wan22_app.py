"""
Wan 2.2 Image-to-Video endpoint on Modal.com.

Wan 2.2 is the MoE upgrade over Wan 2.1 (A14B = two 14B experts: high-noise +
low-noise denoisers). Native 720p, much better motion fidelity, less morphing,
better camera adherence. Best self-hosted open i2v model as of late 2025.

Deploy (run once from any terminal with Modal CLI authenticated):
    modal deploy scripts/modal_wan22_app.py

After deploy, copy the printed web endpoint URL into:
    .env  ->  MODAL_WAN22_ENDPOINT=https://...modal.run

The endpoint accepts a JSON POST body and returns base64-encoded MP4 bytes.
Same contract as the Wan 2.1 / LTX endpoints, so img2video.py can call it
identically and use Wan 2.1 / LTX as automatic fallbacks.
"""

import io
import base64
import modal

app = modal.App("moralverse-wan22-video")

# Wan 2.2 I2V — Mixture-of-Experts (A14B). Native 720p, SOTA among open i2v models.
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def _download_model():
    """Pre-download model weights into the image at build time (one-time cost)."""
    import os
    from huggingface_hub import snapshot_download
    os.makedirs("/model", exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir="/model",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )


wan22_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "diffusers>=0.34.0",       # Wan 2.2 MoE support landed in 0.34
        "transformers>=4.46.0",
        "accelerate>=0.33.0",
        "sentencepiece",
        "imageio[ffmpeg]",
        "Pillow",
        "fastapi[standard]",
        "ftfy",
        "huggingface_hub",
        "peft>=0.13.0",
    )
    .run_function(
        _download_model,
        secrets=[modal.Secret.from_name("huggingface")],
        timeout=3600,              # ~60GB of weights — give the download time
    )
)

with wan22_image.imports():
    import torch
    from diffusers import WanImageToVideoPipeline
    from PIL import Image
    import imageio
    import numpy as np


@app.cls(
    gpu="A100-80GB",               # 80GB needed for the MoE i2v even with offload
    image=wan22_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1500,                  # 25 min headroom: 35 steps x ~30-45s + cold-start
    scaledown_window=300,
)
class Wan22Model:
    @modal.enter()
    def load(self):
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # MoE i2v: keep cpu-offload on — both experts share VRAM peakily and the
        # text encoder (UMT5-XXL) is large. Offload keeps peak VRAM safe.
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "/model",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()  # MUST be called BEFORE any GPU transfer

    @modal.method()
    def generate(
        self,
        image_b64: str,
        prompt: str = "gentle cinematic motion, smooth camera, vivid colors, natural physical motion",
        negative_prompt: str = (
            "flat 2D, anime, cel-shaded, hand-drawn, sketch, low resolution, "
            "warped face, melting face, extra limbs, extra fingers, morphing, "
            "ghosting, smearing, double subject, duplicate, jittery, distorted, "
            "blurry, worst quality, inconsistent motion"
        ),
        num_frames: int = 121,     # 121 @ 24fps = ~5s. Max ~241 (~10s).
        fps: int = 24,
        height: int = 720,         # 720p native (Wan 2.2 strength — do NOT downscale)
        width: int = 1280,
        num_inference_steps: int = 35,    # more steps -> better MoE convergence
        guidance_scale: float = 5.0,
        seed: int = -1,            # -1 = random
    ) -> bytes:
        import os
        import tempfile
        import random

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Center-crop the source to the target aspect BEFORE resize so faces are
        # never stretched (4:3 input -> 16:9 output without distortion).
        target_ar = width / height
        sw, sh = image.size
        src_ar = sw / sh
        if abs(src_ar - target_ar) > 0.01:
            if src_ar > target_ar:
                nw = int(round(sh * target_ar)); left = (sw - nw) // 2
                image = image.crop((left, 0, left + nw, sh))
            else:
                nh = int(round(sw / target_ar)); top = (sh - nh) // 2
                image = image.crop((0, top, sw, top + nh))
        image = image.resize((width, height), Image.LANCZOS)

        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)

        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        frames = output.frames[0]   # list of PIL.Image or float arrays

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        try:
            writer = imageio.get_writer(
                tmp_path, fps=fps, codec="libx264",
                output_params=["-crf", "18", "-pix_fmt", "yuv420p"],  # crf 18 = higher quality
            )
            for frame in frames:
                arr = np.array(frame)
                if arr.dtype != np.uint8:
                    # diffusers may return float32 in [0,1] — scale explicitly
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                writer.append_data(arr)
            writer.close()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@app.function(image=wan22_image, timeout=1500)
@modal.fastapi_endpoint(method="POST")
def generate_i2v(item: dict) -> dict:
    """
    POST body (JSON):
        image               - base64-encoded PNG/JPG (required)
        prompt              - motion description (optional)
        negative_prompt     - what to avoid (optional)
        num_frames          - default 121 (~5s @ 24fps)
        fps                 - default 24
        height / width      - default 720 x 1280 (overridable)
        num_inference_steps - default 35
        guidance_scale      - default 5.0
        seed                - default -1 (random per clip)

    Response (JSON):
        video               - base64-encoded mp4 bytes
    """
    kwargs = dict(
        image_b64=item["image"],
        prompt=item.get("prompt",
            "gentle cinematic motion, smooth camera, vivid colors, natural physical motion"),
        num_frames=item.get("num_frames", 121),
        fps=item.get("fps", 24),
        seed=item.get("seed", -1),
    )
    for key in ("negative_prompt", "height", "width", "num_inference_steps", "guidance_scale"):
        if item.get(key) is not None:
            kwargs[key] = item[key]

    video_bytes = Wan22Model().generate.remote(**kwargs)
    return {"video": base64.b64encode(video_bytes).decode()}
