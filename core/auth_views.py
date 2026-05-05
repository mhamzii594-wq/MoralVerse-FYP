"""
Authentication views for user signup, login, and logout.
"""

from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.http import HttpRequest, HttpResponse


@require_http_methods(["GET", "POST"])
def signup_view(request: HttpRequest) -> HttpResponse:
    """User signup view."""
    if request.user.is_authenticated:
        return redirect("home")
    
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        password_confirm = request.POST.get("password_confirm", "")
        
        # Validation
        if not username or not email or not password:
            messages.error(request, "All fields are required.")
            return render(request, "auth/signup.html")
        
        if password != password_confirm:
            messages.error(request, "Passwords do not match.")
            return render(request, "auth/signup.html")
        
        if len(password) < 6:
            messages.error(request, "Password must be at least 6 characters.")
            return render(request, "auth/signup.html")
        
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return render(request, "auth/signup.html")
        
        if User.objects.filter(email__iexact=email).exists():
            messages.error(request, "Email already in use. Please use another email.")
            return render(request, "auth/signup.html")
        
        # Create user
        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )
            messages.success(request, "Account created successfully! Please login.")
            return redirect("login")
        except Exception as e:
            messages.error(request, f"Error creating account: {str(e)}")
            return render(request, "auth/signup.html")
    
    return render(request, "auth/signup.html")


@require_http_methods(["GET", "POST"])
def login_view(request: HttpRequest) -> HttpResponse:
    """User login view."""
    if request.user.is_authenticated:
        return redirect("home")
    
    if request.method == "POST":
        username_or_email = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        
        if not username_or_email or not password:
            messages.error(request, "Username and password are required.")
            return render(request, "auth/login.html")

        # Resolve account first so we can return a precise error message.
        user_obj = None
        try:
            if "@" in username_or_email:
                user_obj = User.objects.filter(email__iexact=username_or_email).first()
            else:
                user_obj = User.objects.filter(username__iexact=username_or_email).first()
        except Exception:
            user_obj = None

        if not user_obj:
            messages.error(request, "No account found with this username/email.")
            return render(request, "auth/login.html")

        user = authenticate(request, username=user_obj.username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            next_url = request.GET.get("next", "home")
            return redirect(next_url)
        else:
            messages.error(request, "Wrong password. Please try again.")
            return render(request, "auth/login.html")
    
    return render(request, "auth/login.html")


def logout_view(request: HttpRequest) -> HttpResponse:
    """User logout view."""
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("landing")


def admin_login_view(request: HttpRequest) -> HttpResponse:
    """Admin login view (redirects to Django admin)."""
    if request.user.is_authenticated and request.user.is_staff:
        return redirect("/admin/")
    
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        
        if not username or not password:
            messages.error(request, "Username and password are required.")
            return render(request, "auth/admin_login.html")
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_staff:
            login(request, user)
            messages.success(request, "Admin login successful!")
            return redirect("/admin/")
        else:
            messages.error(request, "Invalid admin credentials or insufficient permissions.")
            return render(request, "auth/admin_login.html")
    
    return render(request, "auth/admin_login.html")

