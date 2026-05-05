"""
Runtime helpers for hardware-aware execution.

These utilities keep GPU detection in one place so modules can
opt into acceleration while still falling back safely to CPU.
"""

from __future__ import annotations

import os
from typing import TypedDict


class GPUProfile(TypedDict):
    available: bool
    name: str
    vram_gb: float


_gpu_profile_cache: GPUProfile | None = None


def get_gpu_profile() -> GPUProfile:
    """
    Return a lightweight GPU profile.

    Supports force flags:
    - FORCE_CPU=true  -> always report CPU-only
    - FORCE_GPU=true  -> mark GPU available (for testing config paths)
    """
    global _gpu_profile_cache
    if _gpu_profile_cache is not None:
        return _gpu_profile_cache

    force_cpu = (os.getenv("FORCE_CPU") or "").lower() in {"1", "true", "yes"}
    if force_cpu:
        _gpu_profile_cache = {"available": False, "name": "CPU (forced)", "vram_gb": 0.0}
        return _gpu_profile_cache

    force_gpu = (os.getenv("FORCE_GPU") or "").lower() in {"1", "true", "yes"}

    try:
        import torch

        if torch.cuda.is_available() or force_gpu:
            device_name = "CUDA GPU"
            vram_gb = 0.0
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                vram_gb = round(props.total_memory / (1024**3), 2)
            _gpu_profile_cache = {"available": True, "name": device_name, "vram_gb": vram_gb}
            return _gpu_profile_cache
    except Exception:
        pass

    _gpu_profile_cache = {"available": force_gpu, "name": "CPU", "vram_gb": 0.0}
    return _gpu_profile_cache


def get_render_preset() -> dict[str, object]:
    """
    Return render defaults tuned for available hardware.
    """
    profile = get_gpu_profile()
    if profile["available"] and profile["vram_gb"] >= 5.0:
        return {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "video_codec": "h264_nvenc",
            "preset": "fast",
        }

    return {
        "width": 1280,
        "height": 720,
        "fps": 24,
        "video_codec": "libx264",
        "preset": "ultrafast",
    }

