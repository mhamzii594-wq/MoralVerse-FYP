import json
import os
import logging
from pathlib import Path
from typing import Any, Dict

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from ai_modules.decision_engine import apply_decision
from ai_modules.llm_engine import generate_story
from ai_modules.avatar_processor import process_webcam_avatar, describe_avatar
from ai_modules.image_engine import generate_scene_image
from ai_modules.tts_engine import generate_audio
from core.models import StoryRequest, StoryScene, UserDecision
from core.services.pipeline import generate_story_video, generate_story_video_async

logger = logging.getLogger(__name__)
User = get_user_model()


def landing(request: HttpRequest) -> HttpResponse:
    """Landing page with signup/login options."""
    if request.user.is_authenticated:
        return redirect("home")
    return render(request, "landing.html")


def home(request: HttpRequest) -> HttpResponse:
    """
    Main home page - user can create stories (login optional for creation, required for download).
    """
    context = {
        "user": request.user,
    }
    return render(request, "home.html", context)


def story_api(request: HttpRequest) -> JsonResponse:
    """POST endpoint to generate a story."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        payload = {}

    story = generate_story(payload or {})
    return JsonResponse(story)


def decision_api(request: HttpRequest) -> JsonResponse:
    """POST endpoint that applies the decision engine to an existing story JSON."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    story = payload.get("story")
    option = payload.get("option")
    if not story or option not in {"A", "B"}:
        return JsonResponse({"error": "story and option (A/B) are required"}, status=400)

    updated_story = apply_decision(story, option)
    return JsonResponse(updated_story)


def create_story_request(request: HttpRequest) -> HttpResponse:
    """Create a new story request and redirect to preview page."""
    if request.method != "POST":
        return redirect("home")

    user_prompt = request.POST.get("prompt", "") or "A short moral story for children."
    child_name = (request.POST.get("child_name", "") or "").strip()
    moral_theme = (request.POST.get("moral_theme", "") or "").strip()
    llm_provider = ((request.POST.get("llm_provider") or "").strip() or (os.getenv("LLM_PROVIDER") or "gemini")).lower()
    image_provider = ((request.POST.get("image_provider") or "").strip() or (os.getenv("IMAGE_PROVIDER") or "gemini")).lower()
    tts_provider = ((request.POST.get("tts_provider") or "").strip() or (os.getenv("TTS_PROVIDER") or "gemini")).lower()

    if llm_provider not in {"openai", "gemini", "groq", "stub"}:
        llm_provider = (os.getenv("LLM_PROVIDER") or "gemini").lower()
    if image_provider not in {"openai", "gemini", "replicate", "local_sd", "stability", "flux", "stub"}:
        image_provider = (os.getenv("IMAGE_PROVIDER") or "gemini").lower()
    if tts_provider not in {"gemini", "openai", "elevenlabs", "coqui", "stub"}:
        tts_provider = (os.getenv("TTS_PROVIDER") or "gemini").lower()

    webcam_avatar_path = (request.POST.get("webcam_avatar_path", "") or "").strip()
    avatar_file = request.FILES.get("avatar")

    # SRDS FR-1/FR-3: server-side validation.
    try:
        child_age = int((request.POST.get("child_age") or "").strip())
    except Exception:
        child_age = -1

    # Rate Limiting (Credit Protection)
    from django.utils import timezone
    from datetime import timedelta
    
    limit = int(os.getenv("DAILY_STORY_LIMIT", 3))
    if request.user.is_authenticated:
        recent_stories = StoryRequest.objects.filter(
            user=request.user, 
            created_at__gte=timezone.now() - timedelta(days=1)
        ).count()
    else:
        # Simple session-based limit for guest users
        recent_stories = request.session.get('daily_story_count', 0)
        last_story_time = request.session.get('last_story_date')
        today = timezone.now().date().isoformat()
        
        if last_story_time != today:
            recent_stories = 0
            request.session['daily_story_count'] = 0
            request.session['last_story_date'] = today

    if recent_stories >= limit:
        logger.warning("create_story: daily limit hit (recent=%s, limit=%s)", recent_stories, limit)
        messages.error(request, f"You have reached your limit of {limit} stories per day to protect API credits.")
        return redirect("home")

    # Increment count
    if not request.user.is_authenticated:
        request.session['daily_story_count'] = recent_stories + 1

    if not child_name:
        logger.warning("create_story: empty child_name")
        messages.error(request, "Child name cannot be empty.")
        return redirect("home")
    if child_age < 3 or child_age > 12:
        logger.warning("create_story: invalid child_age=%s", child_age)
        messages.error(request, "Child age must be between 3 and 12.")
        return redirect("home")
    if not moral_theme:
        logger.warning("create_story: empty moral_theme")
        messages.error(request, "Moral theme cannot be empty.")
        return redirect("home")

    # SRDS: language selection.
    preferred_language = (request.POST.get("language") or "en").strip().lower()
    if preferred_language not in {"en", "ur"}:
        preferred_language = "en"

    # SRDS FR-8: default generic avatar when none is uploaded.
    from ai_modules.avatar_processor import create_default_avatar, process_avatar

    avatar_path = create_default_avatar(user_identifier=child_name)
    if webcam_avatar_path:
        avatar_path = webcam_avatar_path
    elif avatar_file:
        try:
            allowed_types = {"image/jpeg", "image/png"}
            if avatar_file.content_type not in allowed_types:
                messages.error(request, "Avatar must be a JPG or PNG image.")
                return redirect("home")
            if avatar_file.size > 5 * 1024 * 1024:
                messages.error(request, "Avatar image is too large (max 5MB).")
                return redirect("home")

            # Always attempt Ghibli-style conversion with fallbacks (local API, Stability, etc.)
            convert_ghibli = True
            avatar_path = process_avatar(
                avatar_file,
                user_identifier=child_name,
                convert_ghibli=convert_ghibli,
            )
        except Exception as e:
            logger.error(f"Avatar processing failed: {e}")
            messages.error(request, "Failed to process avatar image.")

    # Create story request (associate with user if logged in)
    story_request = StoryRequest.objects.create(
        user=request.user if request.user.is_authenticated else None,
        child_name=child_name,
        child_age=child_age,
        moral_theme=moral_theme,
        prompt=user_prompt,
        avatar_path=avatar_path,
        llm_provider=llm_provider,
        image_provider=image_provider,
        tts_provider=tts_provider,
        preferred_language=preferred_language,
        status="pending",
    )
    
    return redirect("story_preview", story_id=story_request.id)


def story_preview(request: HttpRequest, story_id: int) -> HttpResponse:
    """Preview page showing story JSON, generated images, and audio playback."""
    story_request = get_object_or_404(StoryRequest, id=story_id)
    
    # If status is failed, reset to pending to allow retry
    if story_request.status == 'failed':
        story_request.status = 'pending'
        story_request.save()

    # Generate story if not already generated
    if not story_request.story_json or not story_request.story_json.get('scenes'):
        try:
            from ai_modules.avatar_processor import describe_avatar
            avatar_desc = describe_avatar(story_request.avatar_path) if story_request.avatar_path else "a friendly child"
            
            input_data = {
                "prompt": story_request.prompt,
                "child_name": story_request.child_name,
                "child_age": story_request.child_age,
                "moral_theme": story_request.moral_theme,
                "avatar_path": story_request.avatar_path,
                "avatar_description": avatar_desc,
                "llm_provider": story_request.llm_provider,
                "image_provider": story_request.image_provider,
                "preferred_language": story_request.preferred_language,
            }
            story_json = generate_story(input_data)
            story_request.story_json = story_json
            story_request.title = story_json.get('title', f"Story for {story_request.child_name}")
            story_request.save()
            
            # Create scene records
            for scene_data in story_json.get('scenes', []):
                StoryScene.objects.get_or_create(
                    story_request=story_request,
                    scene_id=scene_data.get('id', 0),
                    defaults={
                        'text': scene_data.get('text', ''),
                        'image_prompt': scene_data.get('image_prompt', ''),
                        'decision': scene_data.get('decision'),
                    }
                )
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            messages.error(request, f"Failed to generate story: {e}")
    
    # Generate missing image/audio for preview so users can validate API pipeline
    # without waiting for full video generation.
    try:
        for scene in story_request.scenes.all().order_by('scene_id'):
            # Cache Guard: Only generate if file doesn't exist on disk
            image_exists = False
            if scene.image_path:
                full_image_path = Path(settings.MEDIA_ROOT) / scene.image_path
                # Real images are usually > 50KB. Stubs are ~6KB.
                if full_image_path.exists() and full_image_path.stat().size > 10000:
                    image_exists = True

            if not image_exists:
                image_prompt = scene.image_prompt or f"Children's story illustration: {scene.text[:100]}"
                scene.image_path = generate_scene_image(
                    image_prompt,
                    story_request.avatar_path if story_request.avatar_path else None,
                    image_provider=story_request.image_provider,
                )
            
            # Cache Guard: Only generate audio if file doesn't exist
            audio_exists = False
            if scene.audio_path:
                full_audio_path = Path(settings.MEDIA_ROOT) / scene.audio_path
                # Real audio is usually > 10KB. Stubs are very small.
                if full_audio_path.exists() and full_audio_path.stat().size > 5000:
                    audio_exists = True

            if not audio_exists:
                scene.audio_path = generate_audio(
                    scene.text or f"Scene {scene.scene_id}",
                    voice="child_friendly",
                    language=story_request.preferred_language,
                    provider_override=story_request.tts_provider,
                )
            scene.save(update_fields=["image_path", "audio_path"])
    except Exception as e:
        logger.warning("Scene media generation failed in preview: %s", e)

    # Load scenes from database
    scenes = story_request.scenes.all().order_by('scene_id')
    
    context = {
        "story_request": story_request,
        "story_json": json.dumps(story_request.story_json, indent=2, ensure_ascii=False),
        "scenes": scenes,
        "has_video": bool(story_request.video_path),
        "user": request.user,
    }
    
    return render(request, "story_preview.html", context)


def generate_video_api(request: HttpRequest, story_id: int) -> JsonResponse:
    """API endpoint to trigger video generation for a story."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)
    
    story_request = get_object_or_404(StoryRequest, id=story_id)
    
    if story_request.status == 'generating':
        return JsonResponse({"status": "generating", "message": "Video generation in progress"})
    
    if story_request.video_path:
        return JsonResponse({
            "status": "completed",
            "video_path": story_request.video_path,
            "message": "Video already generated"
        })
    
    try:
        # Start video generation (synchronous for now, can be made async)
        story_request.status = 'generating'
        story_request.save()
        
        # Try async if Django-Q is available
        try:
            from django_q.tasks import async_task
            async_task(generate_story_video_async, story_id)
            return JsonResponse({
                "status": "generating",
                "message": "Video generation started in background"
            })
        except ImportError:
            # Fallback to synchronous
            video_path = generate_story_video(story_id)
            return JsonResponse({
                "status": "completed",
                "video_path": video_path,
                "message": "Video generated successfully"
            })
    except Exception as e:
        logger.exception(f"Video generation failed: {e}")
        story_request.mark_failed(str(e))
        return JsonResponse({
            "status": "failed",
            "error": str(e)
        }, status=500)


@login_required
def video_ready(request: HttpRequest, story_id: int) -> HttpResponse:
    """
    Page showing the completed video with player and download option.
    Requires login to download videos.
    """
    story_request = get_object_or_404(StoryRequest, id=story_id)
    
    if not story_request.video_path:
        messages.warning(request, "Video not yet generated. Please wait for generation to complete.")
        return redirect("story_preview", story_id=story_id)
    
    context = {
        "story_request": story_request,
        "video_path": story_request.video_path,
        "subtitle_path": story_request.subtitle_path,
        "user": request.user,
    }
    
    return render(request, "video_ready.html", context)


def decision_interactive_api(request: HttpRequest, story_id: int) -> JsonResponse:
    """Interactive decision API that updates the story and regenerates subsequent scenes."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)
    
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    story_request = get_object_or_404(StoryRequest, id=story_id)
    option = payload.get("option")
    scene_id = payload.get("scene_id")
    
    if option not in {"A", "B"}:
        return JsonResponse({"error": "option must be 'A' or 'B'"}, status=400)
    
    if not scene_id:
        return JsonResponse({"error": "scene_id is required"}, status=400)
    
    try:
        scene = StoryScene.objects.get(story_request=story_request, scene_id=scene_id)
        
        # Record decision
        UserDecision.objects.create(
            story_request=story_request,
            scene=scene,
            choice=option
        )
        
        # Apply decision to story JSON
        story_json = story_request.story_json
        updated_story = apply_decision(story_json, option)

        # SRDS FR-15: regenerate subsequent story parts after a decision.
        try:
            from ai_modules.llm_engine import regenerate_story_after_decision

            updated_story = regenerate_story_after_decision(
                existing_story=updated_story,
                option=option,
                decision_scene_id=scene.scene_id,
                input_data={
                    "prompt": story_request.prompt,
                    "child_name": story_request.child_name,
                    "child_age": story_request.child_age,
                    "moral_theme": story_request.moral_theme,
                    "avatar_path": story_request.avatar_path,
                    "avatar_description": describe_avatar(story_request.avatar_path) if story_request.avatar_path else "a friendly child",
                    "llm_provider": story_request.llm_provider,
                    "image_provider": story_request.image_provider,
                    "preferred_language": story_request.preferred_language,
                },
            )
        except Exception:
            # Keep deterministic update if regeneration is unavailable.
            pass

        story_request.story_json = updated_story
        story_request.save()
        
        # Update scene
        # Keep DB scene text consistent with the updated story JSON (important for UI reloads).
        try:
            for sc in (updated_story.get("scenes") or []):
                if not isinstance(sc, dict):
                    continue
                sc_id = sc.get("id")
                if not sc_id:
                    continue
                # Update decision scene and all subsequent scenes.
                if int(sc_id) >= int(scene.scene_id):
                    # 1. Get the new image_path from the regenerated JSON
                    new_image_path = sc.get("image_path")
                    
                    # 2. Generate audio for the new text
                    new_audio_path = generate_audio(
                        sc.get("text", "") or f"Scene {sc_id}",
                        voice="child_friendly",
                        language=story_request.preferred_language,
                        provider_override=story_request.tts_provider,
                    )
                    
                    # 3. Update the database record
                    StoryScene.objects.filter(
                        story_request=story_request,
                        scene_id=sc_id,
                    ).update(
                        text=sc.get("text", "") or "",
                        image_prompt=sc.get("image_prompt", "") or "",
                        decision=sc.get("decision"),
                        image_path=new_image_path,
                        audio_path=new_audio_path,
                    )
        except Exception:
            pass

        scene.applied_decision = option
        scene.save()
        
        return JsonResponse(updated_story)
    except StoryScene.DoesNotExist:
        return JsonResponse({"error": f"Scene {scene_id} not found"}, status=404)
    except Exception as e:
        logger.exception(f"Decision application failed: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def admin_dashboard(request: HttpRequest) -> HttpResponse:
    """Admin dashboard to view all story requests, statistics, etc."""
    from django.db.models import Count, Q, Sum
    
    # Get statistics
    total_stories = StoryRequest.objects.count()
    completed_stories = StoryRequest.objects.filter(status='completed').count()
    pending_stories = StoryRequest.objects.filter(status='pending').count()
    generating_stories = StoryRequest.objects.filter(status='generating').count()
    failed_stories = StoryRequest.objects.filter(status='failed').count()
    
    # Get recent stories
    recent_stories = StoryRequest.objects.all()[:50]
    
    # Get stories with videos
    stories_with_videos = StoryRequest.objects.exclude(video_path='').count()
    
    # API call statistics
    total_api_calls = StoryRequest.objects.aggregate(
        total=Sum('api_calls_count')
    )['total'] or 0
    
    context = {
        "total_stories": total_stories,
        "completed_stories": completed_stories,
        "pending_stories": pending_stories,
        "generating_stories": generating_stories,
        "failed_stories": failed_stories,
        "stories_with_videos": stories_with_videos,
        "total_api_calls": total_api_calls,
        "recent_stories": recent_stories,
    }
    
    return render(request, "admin_dashboard.html", context)


@login_required
def user_dashboard(request: HttpRequest) -> HttpResponse:
    """User dashboard showing their stories."""
    user_stories = StoryRequest.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        "user": request.user,
        "stories": user_stories,
        "total_stories": user_stories.count(),
        "completed_stories": user_stories.filter(status='completed').count(),
        "videos_count": user_stories.exclude(video_path='').count(),
    }
    
    return render(request, "user_dashboard.html", context)


@login_required
@require_POST
def delete_story(request: HttpRequest, story_id: int) -> JsonResponse:
    """Delete a story owned by the current user."""
    story = get_object_or_404(StoryRequest, id=story_id, user=request.user)
    story.delete()
    return JsonResponse({"success": True})


@csrf_exempt
def process_webcam_avatar_api(request: HttpRequest) -> JsonResponse:
    """
    API endpoint to process webcam-captured image and convert to Ghibli style.
    
    Expected payload: {
        "image_data": "data:image/jpeg;base64,...",
        "user_identifier": "ali",
        "convert_ghibli": true
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)
    
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    image_data = payload.get("image_data", "")
    user_identifier = payload.get("user_identifier", "user")
    convert_ghibli = payload.get("convert_ghibli", True)
    
    if not image_data:
        return JsonResponse({"error": "image_data is required"}, status=400)
    
    s_key = os.getenv("STABILITY_API_KEY", "")
    logger.debug("Stability key ends in: ...%s", s_key[-4:] if s_key else "NONE")
    logger.debug("Webcam image_data length: %d", len(image_data))
    logger.debug("Webcam image_data prefix: %s", image_data[:50])
    
    try:
        avatar_path = process_webcam_avatar(
            image_data=image_data,
            user_identifier=user_identifier,
            convert_ghibli=convert_ghibli
        )
        
        return JsonResponse({
            "success": True,
            "avatar_path": avatar_path,
            "message": "Avatar processed successfully"
        })
    except Exception as e:
        logger.exception(f"Webcam avatar processing failed: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)
