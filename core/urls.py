from django.urls import path

from . import views
from . import auth_views

urlpatterns = [
    # Landing and Auth
    path("", views.landing, name="landing"),
    path("home/", views.home, name="home"),
    path("signup/", auth_views.signup_view, name="signup"),
    path("login/", auth_views.login_view, name="login"),
    path("logout/", auth_views.logout_view, name="logout"),
    path("admin-login/", auth_views.admin_login_view, name="admin_login"),
    
    # Story Management
    path("api/story/", views.story_api, name="api_story"),
    path("api/decision/", views.decision_api, name="api_decision"),
    path("create-story/", views.create_story_request, name="create_story"),
    path("story/<int:story_id>/", views.story_preview, name="story_preview"),
    path("story/<int:story_id>/generate-video/", views.generate_video_api, name="generate_video"),
    path("story/<int:story_id>/video/", views.video_ready, name="video_ready"),
    path("story/<int:story_id>/decision/", views.decision_interactive_api, name="decision_interactive"),
    path("story/<int:story_id>/delete/", views.delete_story, name="delete_story"),
    
    # Dashboards
    path("dashboard/", views.user_dashboard, name="user_dashboard"),
    path("admin/dashboard/", views.admin_dashboard, name="admin_dashboard"),
    
    # Webcam Avatar
    path("api/process-webcam-avatar/", views.process_webcam_avatar_api, name="process_webcam_avatar"),
]


