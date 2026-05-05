"""
Django admin configuration for MoralVerse.AI models.
"""

from django.contrib import admin
from .models import StoryRequest, StoryScene, UserDecision


@admin.register(StoryRequest)
class StoryRequestAdmin(admin.ModelAdmin):
    list_display = ['id', 'child_name', 'moral_theme', 'status', 'created_at', 'has_video']
    list_filter = ['status', 'moral_theme', 'created_at']
    search_fields = ['child_name', 'moral_theme', 'title']
    readonly_fields = ['created_at', 'updated_at', 'completed_at']
    fieldsets = (
        ('User Input', {
            'fields': ('child_name', 'moral_theme', 'prompt', 'avatar_path')
        }),
        ('Providers', {
            'fields': ('llm_provider', 'image_provider', 'tts_provider')
        }),
        ('Generated Content', {
            'fields': ('title', 'story_json', 'video_path', 'subtitle_path')
        }),
        ('Status & Metadata', {
            'fields': ('status', 'api_calls_count', 'error_message', 'created_at', 'updated_at', 'completed_at')
        }),
    )
    
    def has_video(self, obj):
        return bool(obj.video_path)
    has_video.boolean = True
    has_video.short_description = 'Has Video'


@admin.register(StoryScene)
class StorySceneAdmin(admin.ModelAdmin):
    list_display = ['id', 'story_request', 'scene_id', 'has_image', 'has_audio', 'has_decision']
    list_filter = ['story_request', 'scene_id']
    search_fields = ['story_request__child_name', 'text']
    
    def has_image(self, obj):
        return bool(obj.image_path)
    has_image.boolean = True
    has_image.short_description = 'Has Image'
    
    def has_audio(self, obj):
        return bool(obj.audio_path)
    has_audio.boolean = True
    has_audio.short_description = 'Has Audio'
    
    def has_decision(self, obj):
        return bool(obj.decision)
    has_decision.boolean = True
    has_decision.short_description = 'Has Decision'


@admin.register(UserDecision)
class UserDecisionAdmin(admin.ModelAdmin):
    list_display = ['id', 'story_request', 'scene', 'choice', 'created_at']
    list_filter = ['choice', 'created_at']
    search_fields = ['story_request__child_name']
    readonly_fields = ['created_at']

