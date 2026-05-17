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
    lipsync_clip_paths: Optional[List[Optional[str]]] = None,
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
        lipsync_clip_paths: Optional per-scene talking-head lipsync clips (SadTalker output).
            When present and a clip exists for a scene, the face is composited at the
            bottom-left corner over the main Kling scene clip.
        progress_callback: Optional callable(pct: int) called as assembly progresses.

    Returns:
        output_path on success.
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.VideoClip import ImageClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.audio.AudioClip import CompositeAudioClip
        from moviepy import concatenate_videoclips, concatenate_audioclips, CompositeVideoClip
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

        # --- Composite lipsync face if available ---
        lip_rel = lipsync_clip_paths[i] if lipsync_clip_paths and i < len(lipsync_clip_paths) else None
        lip_full = (media_root / lip_rel) if lip_rel else None
        if lip_full and lip_full.exists():
            try:
                lip_vc = VideoFileClip(str(lip_full)).without_audio()
                # Loop lipsync clip if it's shorter than scene clip
                if lip_vc.duration < clip_duration:
                    loops_needed = int(clip_duration / lip_vc.duration) + 1
                    lip_vc = concatenate_videoclips([lip_vc] * loops_needed)
                lip_vc = lip_vc.subclipped(0, clip_duration).resized((220, 220))
                # Position: bottom-left corner with 16px margin
                lip_positioned = lip_vc.with_position((16, video_clip.h - 236))
                video_clip = CompositeVideoClip([video_clip, lip_positioned])
                logger.info("[cinematic] scene %d: lipsync face composited", scene_num)
            except Exception as exc:
                logger.warning("[cinematic] scene %d lipsync composite failed: %s", scene_num, exc)

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
    scene_prompts: Optional[List[str]] = None,
    scene_durations: Optional[List[float]] = None,
    progress_callback=None,
    max_workers: int = 2,
) -> List[Optional[str]]:
    """
    Generate AI video clips for all scenes in parallel (max 2 at a time).

    Args:
        scene_prompts: Optional per-scene text prompts for guided animation.
        scene_durations: Optional per-scene narration audio lengths in seconds.
            When provided, Kling selects 10s clip for scenes >= 6s, else 5s.
        progress_callback: Called with 0-100 proportional completion.

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

    def _generate_one(index: int, image_rel: str, prompt: str, audio_dur: Optional[float]):
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
                        progress_callback(int(completed[0] / len(scene_images) * 100))
                return

            try:
                generate_clip(image_full, clip_full, duration=5, prompt=prompt, audio_duration=audio_dur)
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
                    progress_callback(int(completed[0] / len(scene_images) * 100))

    threads = []
    for i, img in enumerate(scene_images):
        prompt = (scene_prompts[i] if scene_prompts and i < len(scene_prompts) else "") or ""
        audio_dur = (scene_durations[i] if scene_durations and i < len(scene_durations) else None)
        t = threading.Thread(target=_generate_one, args=(i, img, prompt, audio_dur), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    successful = sum(1 for r in results if r is not None)
    total = len(scene_images)
    if successful == 0:
        logger.error("[cinematic] story=%s: 0/%d Kling clips generated — all scenes will use static image fallback", story_id, total)
    elif successful < total:
        logger.warning("[cinematic] story=%s: %d/%d Kling clips generated — %d scene(s) will use static image fallback", story_id, successful, total, total - successful)
    else:
        logger.info("[cinematic] story=%s: all %d/%d Kling clips generated successfully", story_id, successful, total)

    return results


def generate_lipsync_clips_parallel(
    story_id: int,
    avatar_path: str,
    scene_audio: List[str],
    lipsync_output_dir: str,
    progress_callback=None,
    max_workers: int = 2,
) -> List[Optional[str]]:
    """
    Generate SadTalker lipsync clips for every scene in parallel.

    Args:
        avatar_path: Absolute local path to the avatar portrait image.
        scene_audio: Relative media paths to per-scene narration audio.
        lipsync_output_dir: Relative media dir where clips are saved.
        progress_callback: Called with 0-100 proportional completion.

    Returns a list of relative media paths (or None where generation failed).
    """
    from ai_modules.lipsync import generate_lipsync_clip

    media_root = Path(settings.MEDIA_ROOT)
    lip_dir = media_root / lipsync_output_dir
    lip_dir.mkdir(parents=True, exist_ok=True)

    results: List[Optional[str]] = [None] * len(scene_audio)
    lock = threading.Lock()
    semaphore = threading.Semaphore(max_workers)
    completed = [0]

    def _generate_one(index: int, audio_rel: str):
        with semaphore:
            audio_full = str(media_root / audio_rel)
            lip_rel = f"{lipsync_output_dir}/scene_{index + 1}.mp4"
            lip_full = str(media_root / lip_rel)

            if os.path.exists(lip_full):
                logger.info("[lipsync] scene %d already cached", index + 1)
                with lock:
                    results[index] = lip_rel
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(int(completed[0] / len(scene_audio) * 100))
                return

            try:
                generate_lipsync_clip(avatar_path, audio_full, lip_full)
                with lock:
                    results[index] = lip_rel
                    logger.info("[lipsync] scene %d generated", index + 1)
            except Exception as exc:
                logger.error("[lipsync] scene %d failed: %s", index + 1, exc)
                with lock:
                    results[index] = None

            with lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(int(completed[0] / len(scene_audio) * 100))

    threads = []
    for i, audio in enumerate(scene_audio):
        t = threading.Thread(target=_generate_one, args=(i, audio), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results
