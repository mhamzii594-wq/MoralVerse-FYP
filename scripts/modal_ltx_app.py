"""
LTX-Video 2.3 Image-to-Video endpoint on Modal.com.

Deploy (run once from any terminal with Modal CLI authenticated):
    modal deploy scripts/modal_ltx_app.py

After deploy, copy the printed web endpoint URL into:
    .env  →  MODAL_LTX_ENDPOINT=https://...modal.run

The endpoint accepts a JSON POST body and returns base64-encoded MP4 bytes.
No Modal API key is required in the calling code — the endpoint URL is the only
credential needed.
"""

import io
import base64
import modal

app = modal.App("moralverse-ltx-video")

MODEL_ID = "Lightricks/LTX-Video"

ltx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        "diffusers>=0.33.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "sentencepiece",
        "imageio[ffmpeg]",
        "Pillow",
        "fastapi[standard]",
    )
)

with ltx_image.imports():
    import torch
    from diffusers import LTXImageToVideoPipeline
    from PIL import Image
    import imageio
    import numpy as np


@app.cls(
    gpu="A10G",
    image=ltx_image,
    timeout=600,
    scaledown_window=300,
)
class LTXModel:
    @modal.enter()
    def load(self):
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

    @modal.method()
    def generate(
        self,
        image_b64: str,
        prompt: str = "gentle cinematic motion, smooth camera, vivid colors",
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
        num_frames: int = 97,   # ~4 seconds at 24 fps
        fps: int = 24,
        height: int = 480,
        width: int = 704,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        seed: int = 42,
    ) -> bytes:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        frames = output.frames[0]
        buf = io.BytesIO()
        writer = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264",
                                    output_params=["-crf", "23"])
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()
        return buf.getvalue()


@app.function(image=ltx_image)
@modal.fastapi_endpoint(method="POST")
def generate_i2v(item: dict) -> dict:
    """
    POST body (JSON):
        image           - base64-encoded PNG/JPG (required)
        prompt          - scene motion description (optional)
        negative_prompt - what to avoid (optional)
        num_frames      - default 97 (~4s at 24fps)
        seed            - default 42

    Response (JSON):
        video           - base64-encoded mp4 bytes
    """
    video_bytes = LTXModel().generate.remote(
        image_b64=item["image"],
        prompt=item.get(
            "prompt",
            "gentle cinematic motion, smooth camera, vivid colors"
        ),
        negative_prompt=item.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted",
        ),
        num_frames=item.get("num_frames", 97),
        fps=item.get("fps", 24),
        seed=item.get("seed", 42),
    )
    return {"video": base64.b64encode(video_bytes).decode()}