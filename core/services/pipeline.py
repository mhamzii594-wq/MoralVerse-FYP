"""
Pipeline orchestrator for the complete story → images → audio → video generation.

This module coordinates:
1. Avatar extraction
2. LLM story generation
3. Image generation per scene (with avatar blending)
4. TTS per scene
5. SRT subtitle creation
6. Final MP4 video rendering
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from ai_modules import llm_engine, avatar_processor
from ai_modules.image_engine import generate_scene_image
from ai_modules.tts_engine import generate_audio
from ai_modules.subtitle_engine import generate_subtitles
from ai_modules.video_engine import assemble_video

logger = logging.getLogger(__name__)


def generate_story_video(user_input_id: int) -> str:
    """
    Complete pipeline to generate a story video from user input.
    
    Args:
        user_input_id: ID of StoryRequest model instance
        
    Returns:
        Relative path to generated video file (e.g., "videos/story_123.mp4")
        
    Raises:
        Exception: If any step of the pipeline fails
    """
    from core.models import StoryRequest, StoryScene
    
    try:
        # Load story request
        story_request = StoryRequest.objects.get(id=user_input_id)
        story_request.status = 'generating'
        story_request.save()
        
        logger.info(f"Starting video generation for story request {user_input_id}")
        
        # Step 1: Generate story JSON (if not already generated)
        if not story_request.story_json or not story_request.story_json.get('scenes'):
            logger.info("Generating story with LLM...")

            # Describe the avatar BEFORE story generation so the LLM gets real traits
            # (hair texture, eye colour, outfit) instead of the generic fallback anchor.
            # describe_avatar uses the _orig.jpg (real photo) — vision models read real
            # photos accurately, not anime drawings.
            avatar_description = "a friendly child"
            if story_request.avatar_path:
                try:
                    avatar_description = avatar_processor.describe_avatar(
                        story_request.avatar_path
                    )
                    logger.info("Avatar described: %s", avatar_description[:80])
                except Exception as _desc_err:
                    logger.warning("describe_avatar failed: %s", _desc_err)

            input_data = {
                "prompt": story_request.prompt,
                "child_name": story_request.child_name,
                "child_age": story_request.child_age,
                "moral_theme": story_request.moral_theme,
                "avatar_path": story_request.avatar_path,
                "avatar_description": avatar_description,
                "llm_provider": story_request.llm_provider,
                "image_provider": story_request.image_provider,
                "preferred_language": story_request.preferred_language,
            }

            story_json = llm_engine.generate_story(input_data)
            story_request.story_json = story_json
            story_request.title = story_json.get('title', f"Story for {story_request.child_name}")
            story_request.save()
            
            # Create scene records
            for scene_data in story_json.get('scenes', []):
                StoryScene.objects.create(
                    story_request=story_request,
                    scene_id=scene_data.get('id', 0),
                    text=scene_data.get('text', ''),
                    image_prompt=scene_data.get('image_prompt', ''),
                    decision=scene_data.get('decision'),
                )
        else:
            story_json = story_request.story_json
        
        scenes = story_json.get('scenes', [])
        if not scenes:
            raise ValueError("No scenes in story JSON")
        scene_ids = [scene_data.get('id', i + 1) for i, scene_data in enumerate(scenes)]
        scene_records = {
            s.scene_id: s
            for s in StoryScene.objects.filter(story_request=story_request, scene_id__in=scene_ids)
        }
        
        # Step 2: Generate images for each scene (with consistency via img2img)
        logger.info("Generating scene images...")
        # Always use the Ghibli avatar as reference for ALL scenes —
        # switching to the first scene image after scene 1 causes character drift.
        avatar_path = story_request.avatar_path if story_request.avatar_path else None
        character_anchor = story_json.get('character_anchor', '')

        # Pre-scan: split scenes into cached vs needs-generation BEFORE launching threads
        # so we never make an API call for a scene whose image was already saved.
        scene_images = [None] * len(scenes)
        image_tasks = []  # (list_index, scene_data, scene_record, resolved_prompt)
        for i, scene_data in enumerate(scenes):
            scene_id = scene_data.get('id', i + 1)
            scene = scene_records.get(scene_id)
            if scene is None:
                scene = StoryScene.objects.create(
                    story_request=story_request,
                    scene_id=scene_id,
                    text=scene_data.get('text', ''),
                    image_prompt=scene_data.get('image_prompt', ''),
                    decision=scene_data.get('decision'),
                )
                scene_records[scene_id] = scene

            image_prompt = scene_data.get('image_prompt', '') or f"Children's story illustration: {scene_data.get('text', '')[:100]}"
            if character_anchor and character_anchor not in image_prompt:
                image_prompt = f"{image_prompt}, {character_anchor}"

            if scene.image_path:
                _img_full = Path(settings.MEDIA_ROOT) / scene.image_path
                if _img_full.exists() and _img_full.stat().st_size > 10_000:
                    logger.info(f"Reusing existing image for scene {scene_id}: {scene.image_path}")
                    scene_images[i] = scene.image_path
                else:
                    logger.warning(f"Scene {scene_id} image in DB but missing/small on disk — regenerating")
                    image_tasks.append((i, scene_data, scene, image_prompt))
            else:
                image_tasks.append((i, scene_data, scene, image_prompt))

        if image_tasks:
            _ip = story_request.image_provider
            _av = avatar_path
            # Fixed seed shared by all 6 scene calls — ensures consistent character
            # colour palette and style across scenes generated in the same story.
            _seed = random.randint(10000, 999999)
            _anchor = character_anchor

            def _gen_img(task):
                idx, sd, sc, prompt = task
                path = generate_scene_image(
                    prompt, _av, image_provider=_ip,
                    seed=_seed, character_anchor=_anchor,
                )
                # Save immediately so a preview-page refresh sees this scene as done.
                StoryScene.objects.filter(pk=sc.pk).update(image_path=path)
                return idx, path, sd.get('id')

            with ThreadPoolExecutor(max_workers=6) as _pool:
                futures = {_pool.submit(_gen_img, t): t[0] for t in image_tasks}
                errors = []
                for fut in as_completed(futures):
                    try:
                        idx, path, sid = fut.result()
                        scene_images[idx] = path
                        logger.info(f"Generated image for scene {sid}: {path}")
                    except Exception as e:
                        errors.append(str(e))
                        logger.error(f"Image generation failed: {e}")
                if errors:
                    raise RuntimeError(f"Image generation failed for {len(errors)} scene(s): {errors[0]}")

        _missing_imgs = [i for i, p in enumerate(scene_images) if not p]
        if _missing_imgs:
            logger.error("Pipeline bug: scene images not populated for indices %s — assembly may fail", _missing_imgs)

        # Step 3: Generate audio for each scene
        logger.info("Generating audio for scenes...")
        scene_audio = [None] * len(scenes)
        audio_tasks = []  # (list_index, scene_data, scene_record, scene_text)
        for i, scene_data in enumerate(scenes):
            scene_id = scene_data.get('id', i + 1)
            scene = scene_records.get(scene_id)
            if scene is None:
                scene = StoryScene.objects.create(
                    story_request=story_request,
                    scene_id=scene_id,
                    text=scene_data.get('text', ''),
                    image_prompt=scene_data.get('image_prompt', ''),
                    decision=scene_data.get('decision'),
                )
                scene_records[scene_id] = scene

            scene_text = scene_data.get('text', '') or f"Scene {scene_data.get('id')}"

            if scene.audio_path:
                _aud_full = Path(settings.MEDIA_ROOT) / scene.audio_path
                if _aud_full.exists() and _aud_full.stat().st_size > 30_000:
                    logger.info(f"Reusing existing audio for scene {scene_id}: {scene.audio_path}")
                    scene_audio[i] = scene.audio_path
                else:
                    logger.warning(f"Scene {scene_id} audio in DB but missing/small on disk — regenerating")
                    audio_tasks.append((i, scene_data, scene, scene_text))
            else:
                audio_tasks.append((i, scene_data, scene, scene_text))

        if audio_tasks:
            _lang = story_request.preferred_language
            _tts = story_request.tts_provider

            def _gen_aud(task):
                idx, sd, sc, text = task
                path = generate_audio(
                    text,
                    voice="child_friendly",
                    language=_lang,
                    provider_override=_tts,
                    tts_prompt=sd.get("tts_prompt"),
                )
                StoryScene.objects.filter(pk=sc.pk).update(audio_path=path)
                return idx, path, sd.get('id')

            with ThreadPoolExecutor(max_workers=6) as _pool:
                futures = {_pool.submit(_gen_aud, t): t[0] for t in audio_tasks}
                errors = []
                for fut in as_completed(futures):
                    try:
                        idx, path, sid = fut.result()
                        scene_audio[idx] = path
                        logger.info(f"Generated audio for scene {sid}: {path}")
                    except Exception as e:
                        errors.append(str(e))
                        logger.error(f"Audio generation failed: {e}")
                if errors:
                    raise RuntimeError(f"Audio generation failed for {len(errors)} scene(s): {errors[0]}")

        _missing_aud = [i for i, p in enumerate(scene_audio) if not p]
        if _missing_aud:
            logger.error("Pipeline bug: scene audio not populated for indices %s — assembly may fail", _missing_aud)

        # Step 4: Generate subtitles (Urdu + English) with perfect sync
        logger.info("Generating subtitles...")
        scene_texts = [scene_data.get('text', '') for scene_data in scenes]
        
        # Calculate actual audio durations for perfect sync
        scene_durations = []
        try:
            from mutagen.mp3 import MP3
            from mutagen.wave import WAVE
            for audio_rel_path in scene_audio:
                audio_full_path = Path(settings.MEDIA_ROOT) / audio_rel_path
                if audio_full_path.suffix.lower() == '.mp3':
                    scene_durations.append(MP3(str(audio_full_path)).info.length)
                else:
                    scene_durations.append(WAVE(str(audio_full_path)).info.length)
        except Exception as e:
            logger.warning(f"Could not calculate exact audio durations: {e}")
            scene_durations = None

        subtitle_path = generate_subtitles(
            scene_texts,
            source_language=story_request.preferred_language,
            durations=scene_durations
        )
        story_request.subtitle_path = subtitle_path
        story_request.save()
        
        logger.info(f"Generated subtitles: {subtitle_path}")
        
        # Step 5: Assemble final video
        logger.info("Assembling video...")
        timestamp = int(timezone.now().timestamp() * 1000)
        video_filename = f"story_{user_input_id}_{timestamp}.mp4"
        video_path = f"videos/{video_filename}"
        
        # Optional: Add background music if available
        background_music = None
        music_path = Path(settings.MEDIA_ROOT) / "background_music.mp3"
        if music_path.exists():
            background_music = str(music_path)
        
        final_video_path = assemble_video(
            scene_images=scene_images,
            scene_audio=scene_audio,
            subtitles_srt=subtitle_path,
            output_path=video_path,
            background_music=background_music,
        )
        
        # Update story request
        story_request.video_path = final_video_path
        story_request.mark_completed()
        
        logger.info(f"Video generation completed: {final_video_path}")
        return final_video_path
        
    except StoryRequest.DoesNotExist:
        error_msg = f"Story request {user_input_id} not found"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        logger.exception(f"Pipeline failed for story request {user_input_id}: {e}")
        try:
            story_request = StoryRequest.objects.get(id=user_input_id)
            story_request.mark_failed(str(e))
        except:
            pass
        raise


def generate_story_video_async(user_input_id: int) -> None:
    """
    Async wrapper for generate_story_video (for background tasks).

    This function can be called by Django-Q or Celery workers.
    """
    try:
        generate_story_video(user_input_id)
    except Exception as e:
        logger.exception(f"Async video generation failed for request {user_input_id}: {e}")


def generate_story_video_cinematic(user_input_id: int) -> str:
    """
    Cinematic pipeline: generate AI animated clips per scene then assemble.

    Progress is written to StoryRequest.video_progress (0-100) so the
    frontend can show a real progress bar while the job runs in a thread.
    """
    from core.models import StoryRequest, StoryScene
    from ai_modules.video_engine_cinematic import (
        generate_scene_clips_parallel,
        generate_lipsync_clips_parallel,
        assemble_cinematic,
    )

    story_request = StoryRequest.objects.get(id=user_input_id)

    def _set_progress(pct: int):
        StoryRequest.objects.filter(pk=user_input_id).update(video_progress=pct)

    try:
        _set_progress(0)
        scenes = list(story_request.scenes.order_by('scene_id'))
        if not scenes:
            raise ValueError("No scenes found for story")

        scene_images = [s.image_path for s in scenes if s.image_path]
        scene_audio  = [s.audio_path  for s in scenes if s.audio_path]

        if len(scene_images) != len(scene_audio):
            raise ValueError(
                f"Scene count mismatch: {len(scene_images)} images vs {len(scene_audio)} audio"
            )

        # Build per-scene video prompts for Kling v2.1.
        # Priority: LLM-generated video_prompt stored in story_json (Pass 3) → rule-based fallback.
        from ai_modules.prompt_builder import build_video_prompt

        story_scenes_data = {}
        character_anchor = ''
        if story_request.story_json:
            for sd in story_request.story_json.get('scenes', []):
                story_scenes_data[sd.get('id', 0)] = sd
            character_anchor = story_request.story_json.get('character_anchor', '')

        scenes_with_images = [s for s in scenes if s.image_path]
        total_scenes_count = len(scenes_with_images)
        scene_prompts = []
        position = 0
        for s in scenes_with_images:
            position += 1
            sd = story_scenes_data.get(s.scene_id, {})

            # Use LLM-generated video_prompt if available (stored during story generation)
            llm_video_prompt = (sd.get('video_prompt') or '').strip()
            if llm_video_prompt:
                prompt = llm_video_prompt
                logger.info('[cinematic] scene %d/%d using LLM video_prompt', position, total_scenes_count)
            else:
                # Fallback: rule-based prompt builder
                text = (sd.get('text') or '').strip()
                prompt = build_video_prompt(
                    text=text,
                    character_anchor=character_anchor,
                    position=position,
                    total_scenes=total_scenes_count,
                    title=story_request.title or '',
                )
                logger.info('[cinematic] scene %d/%d using rule-based prompt (no LLM prompt stored)', position, total_scenes_count)

            logger.info('[cinematic] scene %d prompt: %s', position, prompt[:200])
            scene_prompts.append(prompt)

        # --- Measure per-scene audio durations for duration-matched Kling clips ---
        scene_durations: list = []
        try:
            from mutagen.mp3 import MP3
            from mutagen.wave import WAVE
            for audio_rel in scene_audio:
                audio_full = Path(settings.MEDIA_ROOT) / audio_rel
                try:
                    if audio_full.suffix.lower() == '.mp3':
                        scene_durations.append(MP3(str(audio_full)).info.length)
                    else:
                        scene_durations.append(WAVE(str(audio_full)).info.length)
                except Exception:
                    scene_durations.append(None)
            logger.info('[cinematic] audio durations: %s', [round(d, 2) if d else None for d in scene_durations])
        except ImportError:
            logger.warning('[cinematic] mutagen not installed — using default 5s clip duration')
            scene_durations = [None] * len(scene_audio)

        # Step 1a: Generate subtitles (same as slideshow pipeline)
        from ai_modules.subtitle_engine import generate_subtitles
        scene_texts_for_subs = [
            (story_scenes_data.get(s.scene_id, {}).get('text') or '')
            for s in scenes_with_images
        ]
        try:
            subtitle_path = generate_subtitles(
                scene_texts_for_subs,
                source_language=story_request.preferred_language,
                durations=scene_durations if scene_durations else None,
            )
            story_request.subtitle_path = subtitle_path
            story_request.save()
            logger.info('[cinematic] subtitles generated: %s', subtitle_path)
        except Exception as _sub_exc:
            subtitle_path = None
            logger.warning('[cinematic] subtitle generation failed: %s', _sub_exc)

        # Step 1b: Generate AI video clips for each scene (0–50%)
        clip_dir = f"clips/story_{user_input_id}"

        def _clip_progress(pct: int):
            # pct is 0-100 proportional; map to 0-50%
            _set_progress(int(pct * 0.50))

        clip_paths = generate_scene_clips_parallel(
            story_id=user_input_id,
            scene_images=scene_images,
            clip_output_dir=clip_dir,
            scene_prompts=scene_prompts,
            scene_durations=scene_durations if scene_durations else None,
            progress_callback=_clip_progress,
            max_workers=2,
        )

        # Cache clip paths on scenes for re-use
        for scene, clip_rel in zip(scenes, clip_paths):
            if clip_rel:
                StoryScene.objects.filter(pk=scene.pk).update(video_clip_path=clip_rel)

        _set_progress(50)

        # Step 2: Generate lipsync clips if an avatar is available (50–80%)
        lipsync_paths: list = [None] * len(scene_images)
        avatar_abs = None
        if story_request.avatar_path:
            _candidate = Path(settings.MEDIA_ROOT) / story_request.avatar_path
            if _candidate.exists():
                avatar_abs = str(_candidate)
        if not avatar_abs and story_request.ghibli_avatar_path:
            _candidate = Path(settings.MEDIA_ROOT) / story_request.ghibli_avatar_path
            if _candidate.exists():
                avatar_abs = str(_candidate)

        if avatar_abs:
            logger.info('[cinematic] Generating lipsync clips from avatar: %s', avatar_abs)
            lipsync_dir = f"lipsync/story_{user_input_id}"

            def _lipsync_progress(pct: int):
                # pct is 0-100 proportional; map to 50-80%
                _set_progress(50 + int(pct * 0.30))

            lipsync_paths = generate_lipsync_clips_parallel(
                story_id=user_input_id,
                avatar_path=avatar_abs,
                scene_audio=scene_audio,
                lipsync_output_dir=lipsync_dir,
                progress_callback=_lipsync_progress,
                max_workers=2,
            )
            success_count = sum(1 for p in lipsync_paths if p)
            logger.info('[cinematic] Lipsync: %d/%d clips generated', success_count, len(lipsync_paths))
        else:
            logger.info('[cinematic] No avatar uploaded — skipping lipsync')

        _set_progress(80)

        # Step 3: Assemble final video (80–100%)
        timestamp = int(timezone.now().timestamp() * 1000)
        video_path = f"videos/story_{user_input_id}_cinematic_{timestamp}.mp4"

        background_music = None
        music_path = Path(settings.MEDIA_ROOT) / "background_music.mp3"
        if music_path.exists():
            background_music = str(music_path)

        def _assemble_progress(pct: int):
            _set_progress(80 + int(pct * 0.20))

        assemble_cinematic(
            story_id=user_input_id,
            scene_images=scene_images,
            scene_audio=scene_audio,
            scene_clip_paths=clip_paths,
            output_path=video_path,
            background_music=background_music,
            lipsync_clip_paths=lipsync_paths,
            subtitles_srt=subtitle_path,
            progress_callback=_assemble_progress,
        )

        story_request.video_path = video_path
        story_request.video_progress = 100
        story_request.mark_completed()
        logger.info("Cinematic video complete: %s", video_path)
        return video_path

    except Exception as exc:
        logger.exception("Cinematic pipeline failed for story %s: %s", user_input_id, exc)
        try:
            StoryRequest.objects.filter(pk=user_input_id).update(
                status='failed', error_message=str(exc)
            )
        except Exception:
            pass
        raise

