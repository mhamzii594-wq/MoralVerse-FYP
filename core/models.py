"""
Database models for MoralVerse.AI story persistence.
"""

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
import json


class StoryRequest(models.Model):
    """Main story generation request."""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('generating', 'Generating'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    # User association
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='stories', null=True, blank=True)
    
    # User input
    child_name = models.CharField(max_length=100)
    # SRDS requires age (3-12) influencing complexity and narration speed.
    child_age = models.PositiveIntegerField(default=7)
    moral_theme = models.CharField(max_length=100)
    prompt = models.TextField(blank=True)
    avatar_path = models.CharField(max_length=500, blank=True, null=True)

    # SRDS requires language selection (Urdu/English).
    # Stored as 'en' or 'ur'.
    preferred_language = models.CharField(
        max_length=2,
        default="en",
        choices=[("en", "English"), ("ur", "Urdu")],
    )
    
    # Provider settings
    llm_provider = models.CharField(max_length=50, default='openai')
    image_provider = models.CharField(max_length=50, default='stub')
    tts_provider = models.CharField(max_length=50, default='stub')
    
    # Generated content
    story_json = models.JSONField(default=dict, blank=True)
    title = models.CharField(max_length=200, blank=True)
    
    # Video output
    video_path = models.CharField(max_length=500, blank=True, null=True)
    subtitle_path = models.CharField(max_length=500, blank=True, null=True)
    
    # Status and metadata
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # API call tracking
    api_calls_count = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Story Request'
        verbose_name_plural = 'Story Requests'
    
    def __str__(self):
        return f"Story for {self.child_name} - {self.status} ({self.created_at})"
    
    def mark_completed(self):
        """Mark story as completed."""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save()
    
    def mark_failed(self, error_message: str = ""):
        """Mark story as failed."""
        self.status = 'failed'
        self.error_message = error_message
        self.save()


class StoryScene(models.Model):
    """Individual scene in a story."""
    
    story_request = models.ForeignKey(StoryRequest, on_delete=models.CASCADE, related_name='scenes')
    scene_id = models.IntegerField()
    text = models.TextField()
    image_prompt = models.TextField(blank=True)
    image_path = models.CharField(max_length=500, blank=True, null=True)
    audio_path = models.CharField(max_length=500, blank=True, null=True)
    decision = models.JSONField(default=dict, blank=True, null=True)  # {"A": "...", "B": "..."}
    applied_decision = models.CharField(max_length=1, blank=True, null=True)  # 'A' or 'B'
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['story_request', 'scene_id']
        unique_together = ['story_request', 'scene_id']
    
    def __str__(self):
        return f"Scene {self.scene_id} of {self.story_request}"


class UserDecision(models.Model):
    """Track user decisions made during story interaction."""
    
    story_request = models.ForeignKey(StoryRequest, on_delete=models.CASCADE, related_name='decisions')
    scene = models.ForeignKey(StoryScene, on_delete=models.CASCADE, related_name='user_decisions')
    choice = models.CharField(max_length=1)  # 'A' or 'B'
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['story_request', 'created_at']
    
    def __str__(self):
        return f"Decision {self.choice} for Scene {self.scene.scene_id}"

