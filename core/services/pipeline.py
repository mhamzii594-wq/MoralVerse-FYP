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
            input_data = {
                "prompt": story_request.prompt,
                "child_name": story_request.child_name,
                "child_age": story_request.child_age,
                "moral_theme": story_request.moral_theme,
                "avatar_path": story_request.avatar_path,
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
        scene_images = []
        avatar_path = story_request.avatar_path if story_request.avatar_path else None
        reference_image_path = avatar_path  # Start with avatar for consistency
        
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
            
            image_prompt = scene_data.get('image_prompt', '')
            if not image_prompt:
                # Generate a default prompt from scene text
                image_prompt = f"Children's story illustration: {scene_data.get('text', '')[:100]}"

            # Reuse existing image if already generated (e.g. from preview page)
            if scene.image_path:
                logger.info(f"Reusing existing image for scene {scene_data.get('id')}: {scene.image_path}")
                scene_images.append(scene.image_path)
                if i == 0:
                    reference_image_path = scene.image_path
                continue

            # Use reference image for consistency (avatar for first scene, then previous scene)
            image_path = generate_scene_image(
                image_prompt,
                reference_image_path,
                image_provider=story_request.image_provider,
            )
            scene.image_path = image_path
            scene.save()
            scene_images.append(image_path)

            # Update reference to this scene's image for next scene's consistency
            if i == 0:  # After first scene, use its image as reference for flow
                reference_image_path = image_path

            logger.info(f"Generated image for scene {scene_data.get('id')}: {image_path}")
        
        # Step 3: Generate audio for each scene
        logger.info("Generating audio for scenes...")
        scene_audio = []
        
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
            
            scene_text = scene_data.get('text', '')
            if not scene_text:
                scene_text = f"Scene {scene_data.get('id')}"

            # Reuse existing audio if already generated (e.g. from preview page)
            if scene.audio_path:
                logger.info(f"Reusing existing audio for scene {scene_data.get('id')}: {scene.audio_path}")
                scene_audio.append(scene.audio_path)
                continue

            # Generate audio
            audio_path = generate_audio(
                scene_text,
                voice="child_friendly",
                language=story_request.preferred_language,
                provider_override=story_request.tts_provider,
            )
            scene.audio_path = audio_path
            scene.save()
            scene_audio.append(audio_path)

            logger.info(f"Generated audio for scene {scene_data.get('id')}: {audio_path}")
        
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

