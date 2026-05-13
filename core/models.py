"""
Database models for MoralVerse.AI story persistence.
"""

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
import json
from datetime import timedelta
import secrets


class StoryRequest(models.Model):
    """Main story generation request."""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('generating', 'Generating'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    VIDEO_TYPE_CHOICES = [
        ('slideshow', 'Slideshow'),
        ('cinematic', 'Cinematic'),
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
    # Ghibli-converted version of the avatar (used for scene img2img consistency)
    ghibli_avatar_path = models.CharField(max_length=500, blank=True, null=True)

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
    video_type = models.CharField(max_length=20, choices=VIDEO_TYPE_CHOICES, default='slideshow')
    video_progress = models.IntegerField(default=0)
    
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
    video_clip_path = models.CharField(max_length=500, blank=True, null=True)
    decision = models.JSONField(default=dict, blank=True, null=True)  # {"A": "...", "B": "..."}
    applied_decision = models.CharField(max_length=1, blank=True, null=True)  # 'A' or 'B'
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['story_request', 'scene_id']
        unique_together = ['story_request', 'scene_id']


class EmailVerification(models.Model):
    """Email verification OTP for user signup."""

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    otp = models.CharField(max_length=6)  # 6-digit OTP
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_verified = models.BooleanField(default=False)

    class Meta:
        verbose_name = 'Email Verification'
        verbose_name_plural = 'Email Verifications'

    def __str__(self):
        return f"Email verification for {self.user.username}"

    def is_expired(self):
        """Check if OTP has expired."""
        return timezone.now() > self.expires_at

    def generate_otp(self):
        """Generate 6-digit OTP and set expiry."""
        import random
        self.otp = str(random.randint(100000, 999999))
        self.expires_at = timezone.now() + timedelta(minutes=10)  # 10 minute expiry
        self.save()

    def verify_otp(self, entered_otp):
        """Verify entered OTP."""
        if self.is_expired():
            return False, "OTP has expired"
        if self.otp != entered_otp:
            return False, "Invalid OTP"
        self.is_verified = True
        self.save()
        return True, "OTP verified successfully"

    @classmethod
    def create_verification(cls, user):
        """Create and return a new verification instance."""
        verification = cls(user=user)
        verification.generate_otp()
        verification.save()
        return verification


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

