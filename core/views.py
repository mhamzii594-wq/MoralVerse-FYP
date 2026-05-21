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
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
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


def _provider_catalog() -> Dict[str, list]:
    """Build the AI-provider option lists, flagging which have an API key configured.

    `key=None` means the provider needs no key (free). Each option gets `available`.
    """
    def has(env_key: str) -> bool:
        return bool(os.getenv(env_key, "").strip())

    llm = [
        {"value": "groq",      "label": "Groq — Llama 3.3 70B",     "key": "GROQ_API_KEY"},
        {"value": "gemini",    "label": "Gemini — 2.0 Flash",       "key": "GEMINI_API_KEY"},
        {"value": "openai",    "label": "OpenAI — GPT-4o Mini",     "key": "OPENAI_API_KEY"},
        {"value": "modelslab", "label": "ModelsLab — Llama 3.1 70B","key": "MODELSLAB_API_KEY"},
    ]
    image = [
        {"value": "modelslab",        "label": "ModelsLab — FLUX Dev (Scene Images)", "key": "MODELSLAB_API_KEY"},
        {"value": "modelslab_ghibli", "label": "ModelsLab — Anime / Ghibli Style",    "key": "MODELSLAB_API_KEY"},
        {"value": "gemini",           "label": "Gemini — Flash Image",                "key": "GEMINI_API_KEY"},
        {"value": "stability",        "label": "Stability AI — Core",                 "key": "STABILITY_API_KEY"},
        {"value": "openai",           "label": "OpenAI — DALL·E 3",                   "key": "OPENAI_API_KEY"},
        {"value": "pollinations",     "label": "Pollinations — FLUX (Free)",          "key": None},
    ]
    tts = [
        {"value": "modelslab",  "label": "ModelsLab — Madison Voice",     "key": "MODELSLAB_API_KEY"},
        {"value": "gemini",     "label": "Gemini — 2.0 Flash TTS",        "key": "GEMINI_API_KEY"},
        {"value": "openai",     "label": "OpenAI — TTS-1 Nova",           "key": "OPENAI_API_KEY"},
        {"value": "elevenlabs", "label": "ElevenLabs — Multilingual v2",  "key": "ELEVENLABS_API_KEY"},
    ]
    for group in (llm, image, tts):
        for opt in group:
            opt["available"] = opt["key"] is None or has(opt["key"])
    return {"llm_providers": llm, "image_providers": image, "tts_providers": tts}


def home(request: HttpRequest) -> HttpResponse:
    """
    Main home page - user can create stories (login optional for creation, required for download).
    """
    context = {
        "user": request.user,
        **_provider_catalog(),
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

    if llm_provider not in {"openai", "gemini", "groq", "modelslab", "stub"}:
        llm_provider = (os.getenv("LLM_PROVIDER") or "modelslab").lower()
    if image_provider not in {"openai", "gemini", "replicate", "local_sd", "stability", "flux", "modelslab", "modelslab_ghibli", "pollinations", "stub"}:
        image_provider = (os.getenv("IMAGE_PROVIDER") or "modelslab").lower()
    if tts_provider not in {"gemini", "openai", "elevenlabs", "coqui", "modelslab", "stub"}:
        tts_provider = (os.getenv("TTS_PROVIDER") or "modelslab").lower()

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
    ghibli_avatar_path = None
    if webcam_avatar_path:
        ghibli_avatar_path = webcam_avatar_path
        # Use the original photo for describe_avatar — vision models extract real traits
        # (hair colour, outfit) from photos, not from anime drawings.
        # Falls back to the ghibli path when the client has not sent the orig field.
        _webcam_orig = (request.POST.get("webcam_orig_path", "") or "").strip()
        avatar_path = _webcam_orig or webcam_avatar_path
    elif avatar_file:
        try:
            allowed_types = {"image/jpeg", "image/png"}
            if avatar_file.content_type not in allowed_types:
                messages.error(request, "Avatar must be a JPG or PNG image.")
                return redirect("home")
            if avatar_file.size > 5 * 1024 * 1024:
                messages.error(request, "Avatar image is too large (max 5MB).")
                return redirect("home")

            # process_avatar returns (orig_path, ghibli_path):
            #   orig_path  — original photo → used by describe_avatar for real trait extraction
            #   ghibli_path — anime-converted image → used for FLUX img2img scene consistency
            avatar_path, ghibli_avatar_path = process_avatar(
                avatar_file,
                user_identifier=child_name,
                convert_ghibli=True,
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
        ghibli_avatar_path=ghibli_avatar_path,
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
    
    # Reset stuck/terminal states so the page isn't permanently broken
    if story_request.status == 'failed':
        story_request.status = 'pending'
        story_request.save()
    elif story_request.status == 'generating':
        # If stuck in 'generating' for > 10 min (e.g. worker was killed), self-heal
        from django.utils import timezone
        from datetime import timedelta
        if story_request.updated_at and (timezone.now() - story_request.updated_at) > timedelta(minutes=10):
            StoryRequest.objects.filter(pk=story_request.pk).update(status='pending')
            story_request.status = 'pending'

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
    
    # Generate missing image/audio for preview — all scenes in parallel to cut wait time.
    try:
        from concurrent.futures import ThreadPoolExecutor
        from django.utils import timezone
        from datetime import timedelta

        _character_anchor = (story_request.story_json or {}).get('character_anchor', '')
        _ref_avatar = story_request.ghibli_avatar_path or story_request.avatar_path

        # Anti-wastage guard: if another request is already generating media for this story
        # (e.g. a concurrent page refresh), skip — the in-flight request will save results
        # to DB shortly. Allow retry after 8 minutes in case of a crash.
        _fresh = StoryRequest.objects.values('status', 'updated_at').get(pk=story_request.pk)
        _skip_generation = (
            _fresh['status'] == 'generating' and
            _fresh['updated_at'] is not None and
            (timezone.now() - _fresh['updated_at']) < timedelta(minutes=8)
        )

        if not _skip_generation:
            # Pre-scan ALL scenes before launching any threads so we only bill for what is
            # genuinely missing right now (not for work another thread just completed).
            needs_image, needs_audio = [], []
            for _scene in story_request.scenes.all().order_by('scene_id'):
                if _scene.image_path:
                    _ip = Path(settings.MEDIA_ROOT) / _scene.image_path
                    if not (_ip.exists() and _ip.stat().st_size > 10000):
                        needs_image.append(_scene)
                else:
                    needs_image.append(_scene)

                if _scene.audio_path:
                    _ap = Path(settings.MEDIA_ROOT) / _scene.audio_path
                    if not (_ap.exists() and _ap.stat().st_size > 5000):
                        needs_audio.append(_scene)
                else:
                    needs_audio.append(_scene)

            if needs_image or needs_audio:
                # Mark as generating so a concurrent refresh finds _skip_generation=True.
                StoryRequest.objects.filter(pk=story_request.pk).update(status='generating')

                _img_provider = story_request.image_provider
                _tts_provider = story_request.tts_provider
                _lang = story_request.preferred_language
                _anchor = _character_anchor
                _avatar = _ref_avatar

                def _gen_image(scene):
                    try:
                        prompt = scene.image_prompt or f"Children's story illustration: {scene.text[:100]}"
                        if _anchor and _anchor not in prompt:
                            prompt = f"{prompt}, {_anchor}"
                        path = generate_scene_image(prompt, _avatar, image_provider=_img_provider, scene_text=scene.text)
                        # Save immediately so a page refresh sees this scene as done.
                        StoryScene.objects.filter(pk=scene.pk).update(image_path=path)
                        logger.info("Preview: scene %s image saved: %s", scene.scene_id, path)
                    except Exception as _e:
                        logger.warning("Preview: scene %s image failed: %s", scene.scene_id, _e)

                def _gen_audio(scene):
                    try:
                        path = generate_audio(
                            scene.text or f"Scene {scene.scene_id}",
                            voice="child_friendly",
                            language=_lang,
                            provider_override=_tts_provider,
                        )
                        StoryScene.objects.filter(pk=scene.pk).update(audio_path=path)
                        logger.info("Preview: scene %s audio saved: %s", scene.scene_id, path)
                    except Exception as _e:
                        logger.warning("Preview: scene %s audio failed: %s", scene.scene_id, _e)

                with ThreadPoolExecutor(max_workers=6) as _pool:
                    _futs = (
                        [_pool.submit(_gen_image, s) for s in needs_image] +
                        [_pool.submit(_gen_audio, s) for s in needs_audio]
                    )
                    for _f in _futs:
                        try:
                            _f.result()
                        except Exception as _e:
                            logger.warning("Preview generation thread raised: %s", _e)

                # Reset status — video not yet assembled so don't mark completed.
                # Only reset if we set it (don't overwrite 'completed' from a pipeline run).
                StoryRequest.objects.filter(pk=story_request.pk, status='generating').update(status='pending')

    except Exception as e:
        logger.warning("Scene media generation failed in preview: %s", e)


    # Load scenes from database; annotate each with a JSON-safe decision string
    scenes = list(story_request.scenes.all().order_by('scene_id'))
    for _s in scenes:
        _s.decision_json = json.dumps(_s.decision, ensure_ascii=False) if _s.decision else 'null'

    context = {
        "story_request": story_request,
        "story_json": json.dumps(story_request.story_json, indent=2, ensure_ascii=False),
        "scenes": scenes,
        "has_video": bool(story_request.video_path),
        "user": request.user,
    }
    
    return render(request, "story_preview.html", context)



def generate_video_api(request: HttpRequest, story_id: int) -> JsonResponse:
    """API endpoint to trigger video generation or poll its current status."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    story_request = get_object_or_404(StoryRequest, id=story_id)

    # ── Poll mode: just return current status, no side effects ──────────────
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        payload = {}

    if payload.get("poll"):
        if story_request.status == 'completed' and story_request.video_path:
            return JsonResponse({"status": "completed", "video_path": story_request.video_path})
        if story_request.status == 'failed':
            return JsonResponse({"status": "failed", "error": story_request.error_message})
        return JsonResponse({
            "status": story_request.status,
            "progress": story_request.video_progress,
            "video_type": story_request.video_type,
        })

    # ── Trigger mode ─────────────────────────────────────────────────────────
    if story_request.status == 'generating':
        return JsonResponse({"status": "generating", "message": "Video generation already in progress"})

    if story_request.status == 'completed' and story_request.video_path:
        return JsonResponse({
            "status": "completed",
            "video_path": story_request.video_path,
            "message": "Video already generated"
        })

    video_type = payload.get("video_type", "slideshow")
    if video_type not in ("slideshow", "cinematic"):
        video_type = "slideshow"

    try:
        story_request.status = 'generating'
        story_request.error_message = ''
        story_request.video_type = video_type
        story_request.video_progress = 0
        story_request.video_path = ''
        story_request.save()

        import threading
        from core.services.pipeline import generate_story_video, generate_story_video_cinematic

        if video_type == "cinematic":
            def _bg_cinematic(sid):
                try:
                    generate_story_video_cinematic(sid)
                except Exception as exc:
                    logger.exception("Cinematic video generation failed: %s", exc)
                    try:
                        from core.models import StoryRequest as SR
                        SR.objects.filter(id=sid).update(status='failed', error_message=str(exc))
                    except Exception:
                        pass
            threading.Thread(target=_bg_cinematic, args=(story_id,), daemon=True).start()
            return JsonResponse({
                "status": "generating",
                "video_type": "cinematic",
                "message": "Cinematic video generation started — this takes 3–5 minutes",
            })
        else:
            try:
                from django_q.tasks import async_task
                from core.services.pipeline import generate_story_video_async
                async_task(generate_story_video_async, story_id)
            except ImportError:
                def _bg_slideshow(sid):
                    try:
                        generate_story_video(sid)
                    except Exception as exc:
                        logger.exception("Slideshow video generation failed: %s", exc)
                        try:
                            from core.models import StoryRequest as SR
                            SR.objects.filter(id=sid).update(status='failed', error_message=str(exc))
                        except Exception:
                            pass
                threading.Thread(target=_bg_slideshow, args=(story_id,), daemon=True).start()
            return JsonResponse({
                "status": "generating",
                "video_type": "slideshow",
                "message": "Slideshow video generation started",
            })

    except Exception as e:
        logger.exception(f"Video generation failed: {e}")
        story_request.mark_failed(str(e))
        return JsonResponse({"status": "failed", "error": str(e)}, status=500)


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
        # Sync DB StoryScene records with updated story JSON.
        # New scenes from regeneration (IDs >= decision_scene_id) need to be created, not just updated.
        # Pre-decision scenes are left untouched (images/audio already exist).
        try:
            for sc in (updated_story.get("scenes") or []):
                if not isinstance(sc, dict):
                    continue
                sc_id = sc.get("id")
                if not sc_id:
                    continue
                if int(sc_id) >= int(scene.scene_id):
                    new_audio_path = generate_audio(
                        sc.get("text", "") or f"Scene {sc_id}",
                        voice="child_friendly",
                        language=story_request.preferred_language,
                        provider_override=story_request.tts_provider,
                    )
                    new_image_path = sc.get("image_path")
                    db_scene, _created = StoryScene.objects.get_or_create(
                        story_request=story_request,
                        scene_id=sc_id,
                        defaults={
                            "text": sc.get("text", "") or "",
                            "image_prompt": sc.get("image_prompt", "") or "",
                            "decision": sc.get("decision"),
                            "image_path": new_image_path,
                            "audio_path": new_audio_path,
                        },
                    )
                    if not _created:
                        db_scene.text = sc.get("text", "") or ""
                        db_scene.image_prompt = sc.get("image_prompt", "") or ""
                        db_scene.decision = sc.get("decision")
                        db_scene.image_path = new_image_path
                        db_scene.audio_path = new_audio_path
                        db_scene.save()
                    # Also persist the audio path back into story_json so the pipeline reuses it.
                    sc["audio_path"] = new_audio_path
            story_request.story_json = updated_story
            story_request.save()
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
    agg = StoryRequest.objects.aggregate(
        total_stories=Count('id'),
        completed_stories=Count('id', filter=Q(status='completed')),
        pending_stories=Count('id', filter=Q(status='pending')),
        generating_stories=Count('id', filter=Q(status='generating')),
        failed_stories=Count('id', filter=Q(status='failed')),
        stories_with_videos=Count('id', filter=~Q(video_path='')),
        total_api_calls=Sum('api_calls_count'),
    )
    context = {
        **agg,
        "total_api_calls": agg['total_api_calls'] or 0,
        "recent_stories": list(StoryRequest.objects.select_related('user').order_by('-created_at')[:50]),
    }
    return render(request, "admin_dashboard.html", context)


@login_required
@ensure_csrf_cookie
def user_dashboard(request: HttpRequest) -> HttpResponse:
    """User dashboard showing their stories."""
    from django.db.models import Count, Q, Prefetch
    from core.models import StoryScene
    user_stories = list(
        StoryRequest.objects.filter(user=request.user)
        .prefetch_related(Prefetch('scenes', queryset=StoryScene.objects.order_by('scene_id')))
        .order_by('-created_at')
    )
    agg = StoryRequest.objects.filter(user=request.user).aggregate(
        total=Count('id'),
        completed=Count('id', filter=Q(status='completed')),
        with_video=Count('id', filter=~Q(video_path='')),
    )
    context = {
        "user": request.user,
        "stories": user_stories,
        "total_stories": agg['total'],
        "completed_stories": agg['completed'],
        "videos_count": agg['with_video'],
    }
    return render(request, "user_dashboard.html", context)


@require_POST
def delete_story(request: HttpRequest, story_id: int) -> JsonResponse:
    """Delete a story owned by the current user, including all media files."""
    # Manual auth check so AJAX gets JSON 403 instead of a 302 redirect
    if not request.user.is_authenticated:
        return JsonResponse({"success": False, "error": "Login required"}, status=403)

    story = get_object_or_404(StoryRequest, id=story_id, user=request.user)

    # --- Clean up all media files from disk ---
    media_root = Path(settings.MEDIA_ROOT)
    files_to_delete: list[Path] = []

    # Video & subtitle files
    if story.video_path:
        files_to_delete.append(media_root / story.video_path)
    if story.subtitle_path:
        files_to_delete.append(media_root / story.subtitle_path)
    # Avatar file
    if story.avatar_path:
        files_to_delete.append(media_root / story.avatar_path)

    # Scene images & audio
    for scene in story.scenes.all():
        if scene.image_path:
            files_to_delete.append(media_root / scene.image_path)
        if scene.audio_path:
            files_to_delete.append(media_root / scene.audio_path)

    for fpath in files_to_delete:
        try:
            if fpath.exists() and fpath.is_file():
                fpath.unlink()
                logger.info("Deleted media file: %s", fpath)
        except Exception as exc:
            logger.warning("Could not delete %s: %s", fpath, exc)

    story.delete()
    logger.info("Deleted story %s for user %s", story_id, request.user.username)
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
        orig_path, ghibli_path = process_webcam_avatar(
            image_data=image_data,
            user_identifier=user_identifier,
            convert_ghibli=convert_ghibli,
        )
        return JsonResponse({
            "success": True,
            "avatar_path": ghibli_path,   # backward compat — JS uses this for preview
            "ghibli_path": ghibli_path,
            "orig_path": orig_path,        # original photo for describe_avatar
            "message": "Avatar processed successfully",
        })
    except Exception as e:
        logger.exception(f"Webcam avatar processing failed: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)
