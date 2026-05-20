"""
Wan 2.1 Image-to-Video endpoint on Modal.com.

Deploy (run once from any terminal with Modal CLI authenticated):
    modal deploy scripts/modal_wan_app.py

After deploy, copy the printed web endpoint URL into:
    .env  →  MODAL_WAN_ENDPOINT=https://...modal.run

The endpoint accepts a JSON POST body and returns base64-encoded MP4 bytes.
Same contract as the LTX endpoint so img2video.py can call it identically.
"""

import io
import base64
import modal

app = modal.App("moralverse-wan-video")

# Wan 2.1 480p i2v — best open-source img2video model (Jan 2025)
# 480p variant fits on A10G (24GB). Use 720p on A100 if you upgrade.
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

def _download_model():
    """Pre-download model weights into the image at build time."""
    import os
    from huggingface_hub import snapshot_download
    os.makedirs("/model", exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir="/model",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )

wan_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "diffusers>=0.33.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "sentencepiece",
        "imageio[ffmpeg]",
        "Pillow",
        "fastapi[standard]",
        "ftfy",
        "huggingface_hub",
    )
    .run_function(
        _download_model,
        secrets=[modal.Secret.from_name("huggingface")],
    )
)

with wan_image.imports():
    import torch
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image
    import imageio
    import numpy as np


@app.cls(
    gpu="A100",           # 40GB VRAM — required for 14B model (A10G 22GB is too small)
    image=wan_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=900,
    scaledown_window=300,
)
class WanModel:
    @modal.enter()
    def load(self):
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Load directly with CPU offload — NEVER call .to("cuda") on the full pipeline.
        # enable_model_cpu_offload() moves layers to GPU one at a time during inference,
        # which keeps peak VRAM under 16GB even for the 14B model.
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "/model",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()  # must be called BEFORE any GPU transfer

    @modal.method()
    def generate(
        self,
        image_b64: str,
        prompt: str = "gentle cinematic motion, smooth camera movement, vivid colors, characters moving naturally",
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted, static, no movement",
        num_frames: int = 81,    # 81 frames at 16fps = ~5s. Max is 121 (~7.5s)
        fps: int = 16,
        height: int = 480,
        width: int = 832,        # Wan 2.1 native aspect ratio 832x480
        num_inference_steps: int = 20,  # quality saturates ~20-25; fewer steps = faster
        guidance_scale: float = 5.0,
        seed: int = -1,          # -1 = random seed per clip
    ) -> bytes:
        import os
        import tempfile
        import random

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Center-crop to the target aspect ratio BEFORE resizing so the frame is never
        # stretched (e.g. a 4:3 source into 16:9 would otherwise squash faces ~30%).
        target_ar = width / height
        sw, sh = image.size
        src_ar = sw / sh
        if abs(src_ar - target_ar) > 0.01:
            if src_ar > target_ar:        # too wide — trim sides
                nw = int(round(sh * target_ar)); left = (sw - nw) // 2
                image = image.crop((left, 0, left + nw, sh))
            else:                          # too tall — trim top/bottom
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

        frames = output.frames[0]   # list of PIL Images

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        try:
            writer = imageio.get_writer(
                tmp_path, fps=fps, codec="libx264",
                output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
            )
            for frame in frames:
                arr = np.array(frame)
                if arr.dtype != np.uint8:
                    # diffusers returns float32 frames in [0, 1] — scale to uint8 explicitly
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                writer.append_data(arr)
            writer.close()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@app.function(image=wan_image, timeout=900)
@modal.fastapi_endpoint(method="POST")
def generate_i2v(item: dict) -> dict:
    """
    POST body (JSON):
        image           - base64-encoded PNG/JPG (required)
        prompt          - scene motion description (optional)
        negative_prompt - what to avoid (optional)
        num_frames      - default 81 (~5s at 16fps). Use 121 for ~7.5s
        fps             - default 16
        seed            - default -1 (random per clip)

    Response (JSON):
        video           - base64-encoded mp4 bytes
    """
    video_bytes = WanModel().generate.remote(
        image_b64=item["image"],
        prompt=item.get(
            "prompt",
            "gentle cinematic motion, smooth camera movement, vivid colors, characters moving naturally",
        ),
        negative_prompt=item.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted, static, no movement",
        ),
        num_frames=item.get("num_frames", 81),
        fps=item.get("fps", 16),
        seed=item.get("seed", -1),
    )
    return {"video": base64.b64encode(video_bytes).decode()}
