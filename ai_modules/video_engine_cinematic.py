"""
Cinematic video assembly engine.

Each scene uses an AI-generated animated video clip (from img2video.py)
instead of a static image. Clips are concatenated with crossfade transitions,
narration audio is laid over each clip, and background music is ducked during
speech.
"""

from __future__ import annotations

import os
import logging
import threading
from pathlib import Path
from typing import List, Optional

from django.conf import settings

logger = logging.getLogger(__name__)


def assemble_cinematic(
    story_id: int,
    scene_images: List[str],
    scene_audio: List[str],
    scene_clip_paths: List[Optional[str]],
    output_path: str,
    background_music: Optional[str] = None,
    progress_callback=None,
) -> str:
    """
    Assemble a cinematic video from AI-generated per-scene video clips.

    Args:
        story_id: Used for logging.
        scene_images: Relative media paths to scene images (fallback if clip missing).
        scene_audio: Relative media paths to TTS narration audio.
        scene_clip_paths: Relative media paths to AI video clips (may be None per scene).
        output_path: Relative media path for the final output MP4.
        background_music: Optional path to background music file.
        progress_callback: Optional callable(pct: int) called as assembly progresses.

    Returns:
        output_path on success.
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.VideoClip import ImageClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.audio.AudioClip import CompositeAudioClip
        from moviepy import concatenate_videoclips, concatenate_audioclips
        import moviepy.video.fx as vfx
    except ImportError as exc:
        raise ImportError("MoviePy is required: pip install moviepy>=2.0.0") from exc

    media_root = Path(settings.MEDIA_ROOT)
    full_output = media_root / output_path
    full_output.parent.mkdir(parents=True, exist_ok=True)

    num_scenes = min(len(scene_images), len(scene_audio))
    if num_scenes == 0:
        raise ValueError("No scenes to assemble")

    def _progress(pct: int):
        if progress_callback:
            progress_callback(pct)

    _progress(5)
    clips = []

    for i in range(num_scenes):
        scene_num = i + 1
        logger.info("[cinematic] story=%s scene=%d/%d assembling", story_id, scene_num, num_scenes)

        # --- Load narration audio ---
        audio_path = media_root / scene_audio[i]
        if audio_path.exists():
            try:
                narration = AudioFileClip(str(audio_path))
                duration = narration.duration
            except Exception as exc:
                logger.warning("Audio load failed for scene %d: %s", scene_num, exc)
                narration = None
                duration = 5.0
        else:
            narration = None
            duration = 5.0

        # Pad duration slightly so the clip doesn't cut off right at the end of narration
        clip_duration = duration + 0.5

        # --- Load video clip or fall back to static image ---
        clip_rel = scene_clip_paths[i] if scene_clip_paths else None
        clip_full = (media_root / clip_rel) if clip_rel else None
        video_clip = None

        if clip_full and clip_full.exists():
            try:
                vc = VideoFileClip(str(clip_full))
                # Loop the clip if narration is longer than clip duration
                if vc.duration < clip_duration:
                    loops_needed = int(clip_duration / vc.duration) + 1
                    vc = concatenate_videoclips([vc] * loops_needed)
                video_clip = vc.subclipped(0, clip_duration).resized((1280, 720))
                logger.info("[cinematic] scene %d: using AI video clip", scene_num)
            except Exception as exc:
                logger.warning("VideoFileClip load failed scene %d: %s", scene_num, exc)

        if video_clip is None:
            # Fallback: animated static image (same as slideshow pan/zoom)
            img_path = media_root / scene_images[i]
            if not img_path.exists():
                logger.warning("[cinematic] scene %d image missing, skipping", scene_num)
                continue
            try:
                video_clip = (
                    ImageClip(str(img_path))
                    .with_duration(clip_duration)
                    .resized((1280, 720))
                )
                logger.info("[cinematic] scene %d: fallback to static image", scene_num)
            except Exception as exc:
                logger.warning("ImageClip failed scene %d: %s", scene_num, exc)
                continue

        # --- Attach narration audio ---
        if narration is not None:
            try:
                video_clip = video_clip.with_audio(narration)
            except Exception as exc:
                logger.warning("Audio attach failed scene %d: %s", scene_num, exc)

        clips.append(video_clip)
        _progress(5 + int(60 * scene_num / num_scenes))

    if not clips:
        raise RuntimeError("No scene clips assembled — check images and audio paths")

    # --- Concatenate with crossfade transitions ---
    logger.info("[cinematic] story=%s concatenating %d clips", story_id, len(clips))
    try:
        transition_duration = 0.8
        final_clips = []
        for idx, clip in enumerate(clips):
            if idx == 0:
                final_clips.append(clip)
            else:
                # Crossfade: fade out previous end, fade in this clip start
                faded = clip.with_effects([vfx.CrossFadeIn(transition_duration)])
                final_clips.append(faded)
        final_video = concatenate_videoclips(final_clips, method="compose", padding=-transition_duration)
    except Exception as exc:
        logger.warning("[cinematic] Crossfade failed (%s) — using plain concatenation", exc)
        final_video = concatenate_videoclips(clips, method="compose")

    _progress(70)

    # --- Background music with ducking ---
    if background_music and os.path.exists(background_music):
        try:
            bg = AudioFileClip(background_music)
            # Loop music to match video length
            loops = int(final_video.duration / bg.duration) + 1
            bg_looped = concatenate_audioclips([bg] * loops).subclipped(0, final_video.duration)
            # Duck to 12% — quieter than slideshow (30%) so speech stays very clear
            bg_ducked = bg_looped.with_multiply_volume(0.12)

            if final_video.audio:
                final_audio = CompositeAudioClip([final_video.audio, bg_ducked])
                final_video = final_video.with_audio(final_audio)
            else:
                final_video = final_video.with_audio(bg_ducked)
        except Exception as exc:
            logger.warning("[cinematic] Background music failed: %s", exc)

    _progress(80)

    # --- Write output ---
    logger.info("[cinematic] writing output: %s", full_output)
    try:
        final_video.write_videofile(
            str(full_output),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            logger=None,
        )
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        try:
            final_video.close()
        except Exception:
            pass

    _progress(100)
    logger.info("[cinematic] story=%s video complete: %s", story_id, full_output)
    return output_path


def generate_scene_clips_parallel(
    story_id: int,
    scene_images: List[str],
    clip_output_dir: str,
    progress_callback=None,
    max_workers: int = 2,
) -> List[Optional[str]]:
    """
    Generate AI video clips for all scenes in parallel (max 2 at a time).

    Returns a list of relative media paths (or None where generation failed).
    """
    from ai_modules.img2video import generate_clip

    media_root = Path(settings.MEDIA_ROOT)
    clip_dir = media_root / clip_output_dir
    clip_dir.mkdir(parents=True, exist_ok=True)

    results: List[Optional[str]] = [None] * len(scene_images)
    lock = threading.Lock()
    semaphore = threading.Semaphore(max_workers)
    completed = [0]

    def _generate_one(index: int, image_rel: str):
        with semaphore:
            image_full = str(media_root / image_rel)
            clip_rel = f"{clip_output_dir}/scene_{index + 1}.mp4"
            clip_full = str(media_root / clip_rel)

            if os.path.exists(clip_full):
                logger.info("[cinematic] scene %d clip already cached", index + 1)
                with lock:
                    results[index] = clip_rel
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(int(completed[0] / len(scene_images) * 60))
                return

            try:
                generate_clip(image_full, clip_full, duration=4)
                with lock:
                    results[index] = clip_rel
                    logger.info("[cinematic] scene %d clip generated", index + 1)
            except Exception as exc:
                logger.error("[cinematic] scene %d clip generation failed: %s", index + 1, exc)
                with lock:
                    results[index] = None

            with lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(int(completed[0] / len(scene_images) * 60))

    threads = []
    for i, img in enumerate(scene_images):
        t = threading.Thread(target=_generate_one, args=(i, img), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results
