"""
Video assembly engine using MoviePy.

This module assembles scene images, audio, and subtitles into a final MP4 video.
"""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional

from django.conf import settings
from .runtime import get_render_preset, get_gpu_profile

logger = logging.getLogger(__name__)

# Set ImageMagick binary path at the module level to ensure TextClip works correctly.
_known_magick_paths = [
    r"E:\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
    r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
    "/usr/bin/magick",
    "/usr/bin/convert",
    "/usr/local/bin/magick",
]
magick_path = next((p for p in _known_magick_paths if os.path.exists(p)), None)
if not magick_path:
    magick_path = shutil.which("magick") or shutil.which("convert")
if magick_path:
    os.environ["IMAGEMAGICK_BINARY"] = magick_path
    logger.info(f"ImageMagick detected at: {magick_path}")


def _find_subtitle_font() -> str | None:
    """Find a usable TTF font path for Pillow-based TextClip subtitle rendering."""
    env_font = os.getenv("SUBTITLE_FONT")
    if env_font and os.path.exists(env_font):
        return env_font

    # Urdu-capable fonts first (support Arabic/Naskh script for Urdu subtitles)
    urdu_candidates = [
        r"C:\Windows\Fonts\NotoNaskhArabic-Regular.ttf",
        r"C:\Windows\Fonts\times.ttf",           # Times New Roman has partial Arabic
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.otf",
    ]
    # Check project assets/fonts/ bundle (ship Noto with the app for server deployments)
    import pathlib as _pathlib
    _assets_dir = _pathlib.Path(__file__).parent.parent / "assets" / "fonts"
    for _fname in ("NotoNaskhArabic-Regular.ttf", "NotoSansArabic-Regular.ttf",
                   "NotoNastaliqUrdu-Regular.ttf"):
        _bundled = str(_assets_dir / _fname)
        urdu_candidates.insert(0, _bundled)

    for path in urdu_candidates:
        if os.path.exists(path):
            logger.info("Subtitle font (Urdu-capable): %s", path)
            return path

    # Latin fallback fonts (used for English subtitles only)
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def assemble_video(
    scene_images: List[str],
    scene_audio: List[str],
    subtitles_srt: Optional[str],
    output_path: str,
    background_music: Optional[str] = None,
) -> str:
    """
    Assemble a video from scene images, audio, and subtitles.
    
    Args:
        scene_images: List of relative paths to scene images (e.g., ["scenes/scene1.png", ...])
        scene_audio: List of relative paths to audio files (e.g., ["audio/scene1.wav", ...])
        subtitles_srt: Relative path to SRT subtitle file (optional)
        output_path: Relative path for output video (e.g., "videos/story_123.mp4")
        background_music: Optional path to background music file
        
    Returns:
        Relative path to generated video file
        
    Raises:
        Exception: If video assembly fails
    """
    try:
        # moviepy>=2 removed the `moviepy.editor` convenience module.
        # Import directly from submodules to stay compatible.
        from moviepy.video.VideoClip import ImageClip, TextClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
        from moviepy import concatenate_videoclips, concatenate_audioclips
        from moviepy.audio.AudioClip import CompositeAudioClip
        import moviepy.video.fx as vfx
    except ImportError:
        logger.error("MoviePy not installed. Install with: pip install moviepy")
        raise ImportError("MoviePy is required for video assembly. Install with: pip install moviepy")
    
    media_root = Path(settings.MEDIA_ROOT)
    output_full_path = media_root / output_path
    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure we have matching number of images and audio files
    num_scenes = min(len(scene_images), len(scene_audio))
    if num_scenes == 0:
        raise ValueError("At least one scene image and audio file required")
    
    render_preset = get_render_preset()
    clips = []
    total_duration = 0.0
    
    import random
    for i in range(num_scenes):
        # Load image
        image_path = media_root / scene_images[i]
        if not image_path.exists():
            logger.warning(f"Scene image not found: {image_path}, skipping")
            continue
        
        # Load audio to get duration
        audio_path = media_root / scene_audio[i]
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}, using default duration")
            duration = 5.0
        else:
            try:
                audio_clip = AudioFileClip(str(audio_path))
                duration = audio_clip.duration
                audio_clip.close()
            except Exception as e:
                logger.warning(f"Failed to load audio {audio_path}: {e}")
                duration = 5.0
        
        # No upper cap — let narration finish naturally.
        # 8-second cap was cutting TTS mid-sentence (10-15s per scene is normal).
        duration = max(3.0, duration)
        
        # Create image clip with slight padding for movement
        img_clip = ImageClip(str(image_path)).with_duration(duration)
        
        # Hardware-aware output resolution.
        target_width = int(render_preset["width"])
        target_height = int(render_preset["height"])
        
        # Add Randomized "Cinematic Camera" Paths (Animation Feel)
        try:
            import numpy as np
            from PIL import Image as _PIL_Image

            oversize_factor = 1.2
            base_w = int(target_width * oversize_factor)
            base_h = int(target_height * oversize_factor)
            img_clip = img_clip.resized(new_size=(base_w, base_h))

            movement_type = random.choice(["zoom_in", "zoom_out", "pan_right", "pan_left", "slide_up"])

            def get_animated_frame(get_frame, t):
                progress = min(t / max(duration, 0.001), 1.0)
                frame = get_frame(t)  # numpy (H, W, 3)
                h, w = frame.shape[:2]

                if movement_type in ("zoom_in", "zoom_out"):
                    scale = (1.0 + 0.15 * progress) if movement_type == "zoom_in" else (1.15 - 0.15 * progress)
                    src_w = max(1, int(w / scale))
                    src_h = max(1, int(h / scale))
                    x1 = (w - src_w) // 2
                    y1 = (h - src_h) // 2
                    cropped = frame[y1:y1 + src_h, x1:x1 + src_w]
                    return np.array(_PIL_Image.fromarray(cropped).resize((target_width, target_height), _PIL_Image.BILINEAR))

                if movement_type == "pan_right":
                    x_off = int((w - target_width) * progress)
                elif movement_type == "pan_left":
                    x_off = int((w - target_width) * (1 - progress))
                else:
                    x_off = (w - target_width) // 2

                y_off = int((h - target_height) * (1 - progress)) if movement_type == "slide_up" else (h - target_height) // 2

                x_off = max(0, min(x_off, max(0, w - target_width)))
                y_off = max(0, min(y_off, max(0, h - target_height)))
                return frame[y_off:y_off + target_height, x_off:x_off + target_width]

            img_clip = img_clip.fl(get_animated_frame)
            # Update size metadata to match what get_animated_frame actually returns
            img_clip.w = target_width
            img_clip.h = target_height

            img_clip = img_clip.with_effects([
                vfx.FadeIn(0.5),
                vfx.FadeOut(0.5)
            ])

        except Exception as e:
            logger.warning(f"Animation effect failed for scene {i}: {e}")
            img_clip = img_clip.resized(new_size=(target_width, target_height))

        # Set audio
        if audio_path.exists():
            try:
                audio_clip = AudioFileClip(str(audio_path))
                if audio_clip.duration > duration:
                    audio_clip = audio_clip.subclipped(0, duration)
                img_clip = img_clip.with_audio(audio_clip)
            except Exception as e:
                logger.warning(f"Failed to set audio for scene {i}: {e}")
        
        clips.append(img_clip)
        total_duration += duration
    
    if not clips:
        raise ValueError("No valid clips created")

    # Crossfade transition between scenes instead of hard cuts.
    # Each clip (except the first) gets a 0.5s CrossFadeIn; clips overlap by 0.5s via
    # padding=-0.5 and method="compose" so MoviePy blends the outgoing/incoming frames.
    _XFADE = 0.5
    try:
        if len(clips) > 1:
            faded = []
            for _j, _c in enumerate(clips):
                if _j > 0:
                    _c = _c.with_effects([vfx.CrossFadeIn(_XFADE)])
                faded.append(_c)
            final_video = concatenate_videoclips(faded, padding=-_XFADE, method="compose")
        else:
            final_video = concatenate_videoclips(clips, method="chain")
    except Exception as _xfade_err:
        logger.warning("CrossFade failed (%s), falling back to chain concat", _xfade_err)
        final_video = concatenate_videoclips(clips, method="chain")
    
    # Add background music if provided
    if background_music:
        bg_music_path = Path(background_music)
        if bg_music_path.exists():
            try:
                bg_audio = AudioFileClip(str(bg_music_path))
                # Loop background music to match video duration
                if bg_audio.duration < final_video.duration:
                    loops = min(50, int(final_video.duration / bg_audio.duration) + 1)
                    bg_clips = [bg_audio] * loops
                    bg_audio = concatenate_audioclips(bg_clips).subclipped(0, final_video.duration)
                else:
                    bg_audio = bg_audio.subclipped(0, final_video.duration)
                
                # Lower volume of background music (30% of original)
                bg_audio = bg_audio.with_multiply_volume(0.3)
                
                # Mix with existing audio
                if final_video.audio:
                    final_audio = CompositeAudioClip([final_video.audio, bg_audio])
                else:
                    final_audio = bg_audio
                
                final_video = final_video.with_audio(final_audio)
            except Exception as e:
                logger.warning(f"Failed to add background music: {e}")
    
    # Add subtitles if provided
    if subtitles_srt:
        srt_path = media_root / subtitles_srt
        if srt_path.exists():
            try:
                subtitle_font = _find_subtitle_font()
                logger.info(f"Subtitle font resolved to: {subtitle_font}")

                from moviepy.video.VideoClip import TextClip

                def make_textclip(txt):
                    """Create a text clip for subtitles using Pillow (MoviePy 2.x)."""
                    kwargs = dict(
                        text=txt,
                        font_size=24,
                        color='white',
                        stroke_color='black',
                        stroke_width=2,
                        method='caption',
                        size=(target_width - 100, None),
                        text_align='center',
                    )
                    if subtitle_font:
                        kwargs['font'] = subtitle_font
                    return TextClip(**kwargs)
                
                # Parse SRT and create subtitle clips
                subtitle_clips = []
                with open(srt_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()

                # Normalize line endings (Windows \r\n → \n, old Mac \r → \n)
                srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')

                # Block-based SRT parser: split on blank lines (robust, no regex fragility)
                import re
                blocks = re.split(r'\n{2,}', srt_content.strip())
                for block in blocks:
                    try:
                        lines = block.strip().split('\n')
                        if len(lines) < 3:
                            continue
                        # lines[0] = index number, lines[1] = timestamps, lines[2+] = text
                        ts_line = lines[1]
                        if '-->' not in ts_line:
                            continue
                        start_str, end_str = ts_line.split('-->')
                        start_str = start_str.strip().replace(',', '.')
                        end_str = end_str.strip().replace(',', '.')

                        def _ts_to_sec(ts):
                            parts = ts.split(':')
                            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

                        start_sec = _ts_to_sec(start_str)
                        end_sec = _ts_to_sec(end_str)
                        # Clamp to video duration — a subtitle past the end extends the
                        # CompositeVideoClip, producing a frozen last frame.
                        end_sec = min(end_sec, final_video.duration)
                        if end_sec <= start_sec:
                            continue
                        sub_duration = max(0.1, end_sec - start_sec)
                        text = ' '.join(lines[2:]).strip()
                        if not text:
                            continue

                        txt_clip = make_textclip(text)
                        txt_clip = txt_clip.with_start(start_sec).with_duration(sub_duration).with_position(('center', 'bottom'))
                        subtitle_clips.append(txt_clip)
                    except Exception as sub_err:
                        logger.warning(f"Skipping malformed subtitle entry: {sub_err}")

                if subtitle_clips:
                    logger.info(f"Compositing {len(subtitle_clips)} subtitle clip(s) onto video")
                    final_video = CompositeVideoClip([final_video] + subtitle_clips)
                else:
                    logger.warning("SRT parsed but produced 0 subtitle clips — check SRT file format")
            except Exception as e:
                logger.warning(f"Failed to add subtitles: {e}")
    
    # Write final video
    try:
        codec = str(render_preset["video_codec"])
        fps = int(render_preset["fps"])
        preset = str(render_preset["preset"])
        gpu_profile = get_gpu_profile()
        logger.info(
            "Video render preset: %sx%s @ %sfps | codec=%s | gpu=%s (%.2fGB)",
            target_width,
            target_height,
            fps,
            codec,
            gpu_profile["name"],
            float(gpu_profile["vram_gb"]),
        )

        try:
            final_video.write_videofile(
                str(output_full_path),
                fps=fps,
                codec=codec,
                audio_codec='aac',
                temp_audiofile=str(media_root / "temp_audio.m4a"),
                remove_temp=True,
                preset=preset,
            )
        except Exception as e:
            if codec != "libx264":
                logger.warning("GPU codec failed (%s), retrying with libx264: %s", codec, e)
                final_video.write_videofile(
                    str(output_full_path),
                    fps=fps,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=str(media_root / "temp_audio.m4a"),
                    remove_temp=True,
                    preset='medium',
                )
            else:
                raise
        logger.info(f"Video assembled successfully: {output_full_path}")
    except Exception as e:
        logger.error(f"Failed to write video: {e}")
        raise
    
    finally:
        # Clean up clips
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        try:
            final_video.close()
        except:
            pass
    
    return output_path

