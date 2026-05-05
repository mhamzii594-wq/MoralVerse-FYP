import os
import django
import sys

# Set up Django environment
sys.path.append('c:/Users/user/Desktop/fyp.031.full_pipeline_working/MoralVerse_FYP')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'moralverse.settings')
django.setup()

from core.models import StoryRequest

print("Latest Story Requests:")
for sr in StoryRequest.objects.order_by('-created_at')[:5]:
    print(f"ID: {sr.id} | Status: {sr.status} | Created: {sr.created_at}")
    if sr.status == 'failed':
        print(f"  Error: {sr.error_message}")
