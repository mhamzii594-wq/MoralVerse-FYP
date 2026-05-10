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
                logger.info(f"Reusing existing image for scene {scene_id}: {scene.image_path}")
                scene_images[i] = scene.image_path
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

        scene_images = [p for p in scene_images if p]
        
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
                logger.info(f"Reusing existing audio for scene {scene_id}: {scene.audio_path}")
                scene_audio[i] = scene.audio_path
            else:
                audio_tasks.append((i, scene_data, scene, scene_text))

        if audio_tasks:
            _lang = story_request.preferred_language
            _tts = story_request.tts_provider

            def _gen_aud(task):
                idx, sd, sc, text = task
                path = generate_audio(text, voice="child_friendly", language=_lang, provider_override=_tts)
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

        scene_audio = [p for p in scene_audio if p]
        
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

