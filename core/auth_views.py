"""
Authentication views for user signup, login, and logout.
"""

from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate, update_session_auth_hash
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.http import require_http_methods
import json

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.utils.http import url_has_allowed_host_and_scheme
from django.conf import settings

from .models import EmailVerification


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
        
        # Create inactive user and send verification OTP
        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                is_active=False  # User cannot login until verified
            )
            
            # Create verification OTP
            verification = EmailVerification.create_verification(user)
            
            # Send verification OTP email — clean up user if send fails
            try:
                send_verification_email(user, verification.otp)
            except Exception as email_err:
                import logging; logging.getLogger(__name__).warning("Verification email failed: %s", email_err)
                user.delete()
                messages.error(request, "Verification email could not be sent. Please try again.")
                return render(request, "auth/signup.html")

            messages.success(request,
                "Account created! Please check your email for the verification code and enter it below.")
            return redirect("verify_otp")
        except Exception as e:
            messages.error(request, f"Error creating account: {str(e)}")
            return render(request, "auth/signup.html")
    
    return render(request, "auth/signup.html")


def send_verification_email(user, otp):
    """Send OTP verification email to user."""
    
    context = {
        'user': user,
        'otp': otp,
        'site_name': 'MoralVerse.AI'
    }
    
    html_message = render_to_string('auth/email_verification.html', context)
    plain_message = strip_tags(html_message)
    
    send_mail(
        subject='Your MoralVerse.AI Verification Code',
        message=plain_message,
        html_message=html_message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
        fail_silently=False,
    )


@require_http_methods(["POST"])
def resend_otp_view(request: HttpRequest) -> JsonResponse:
    """Resend OTP to user email."""
    try:
        data = json.loads(request.body)
        email = data.get("email", "").strip().lower()
        
        if not email:
            return JsonResponse({"success": False, "message": "Email is required"})
        
        user = User.objects.get(email__iexact=email, is_active=False)
        verification = EmailVerification.objects.get(user=user)
        
        # Generate new OTP
        verification.generate_otp()
        
        # Send new OTP email
        send_verification_email(user, verification.otp)
        
        return JsonResponse({"success": True, "message": "New verification code sent"})
        
    except User.DoesNotExist:
        return JsonResponse({"success": False, "message": "No unverified account found with this email"})
    except EmailVerification.DoesNotExist:
        return JsonResponse({"success": False, "message": "No verification request found"})
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Error: {str(e)}"})


@require_http_methods(["GET", "POST"])
def verify_otp_view(request: HttpRequest) -> HttpResponse:
    """OTP verification view."""
    if request.user.is_authenticated:
        return redirect("home")
    
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        otp = request.POST.get("otp", "").strip()
        
        if not email or not otp:
            messages.error(request, "Email and OTP are required.")
            return render(request, "auth/verify_otp.html")
        
        try:
            user = User.objects.get(email__iexact=email, is_active=False)
            verification = EmailVerification.objects.get(user=user)
            
            success, message = verification.verify_otp(otp)
            
            if success:
                user.is_active = True
                user.save()
                messages.success(request, "Email verified successfully! You can now login.")
                return redirect("login")
            else:
                messages.error(request, message)
                return render(request, "auth/verify_otp.html")
                
        except User.DoesNotExist:
            messages.error(request, "No unverified account found with this email.")
            return render(request, "auth/verify_otp.html")
        except EmailVerification.DoesNotExist:
            messages.error(request, "No verification request found for this email.")
            return render(request, "auth/verify_otp.html")
        except Exception as e:
            messages.error(request, f"Error verifying OTP: {str(e)}")
            return render(request, "auth/verify_otp.html")
    
    return render(request, "auth/verify_otp.html")


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

        # Check if user is active
        if not user_obj.is_active:
            messages.warning(request, "Your account is not verified yet. Please check your email for the verification code.")
            return redirect("verify_otp")

        user = authenticate(request, username=user_obj.username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            next_url = request.GET.get("next") or ""
            if not url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
                next_url = "home"
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


@require_http_methods(["GET"])
def account_settings_view(request: HttpRequest) -> HttpResponse:
    """Account settings page — profile, change password, delete account."""
    if not request.user.is_authenticated:
        messages.error(request, "Please login to access account settings.")
        return redirect("login")
    from .models import StoryRequest
    story_count = StoryRequest.objects.filter(user=request.user).count()
    return render(request, "user_settings.html", {"story_count": story_count})


@require_http_methods(["POST"])
def change_password_view(request: HttpRequest) -> HttpResponse:
    """Handle password change form submission."""
    if not request.user.is_authenticated:
        return redirect("login")

    current_password = request.POST.get("current_password", "")
    new_password = request.POST.get("new_password", "")
    confirm_password = request.POST.get("confirm_password", "")

    if not request.user.check_password(current_password):
        messages.error(request, "Current password is incorrect.")
        return redirect("account_settings")

    if len(new_password) < 6:
        messages.error(request, "New password must be at least 6 characters.")
        return redirect("account_settings")

    if new_password != confirm_password:
        messages.error(request, "New passwords do not match.")
        return redirect("account_settings")

    request.user.set_password(new_password)
    request.user.save()
    update_session_auth_hash(request, request.user)
    messages.success(request, "Password changed successfully.")
    return redirect("account_settings")


@require_http_methods(["POST"])
def delete_account_view(request: HttpRequest) -> HttpResponse:
    """Permanently delete the authenticated user's account."""
    import logging
    logger = logging.getLogger(__name__)

    if not request.user.is_authenticated:
        return redirect("login")

    confirm_password = request.POST.get("confirm_password", "")

    if not confirm_password:
        messages.error(request, "Please enter your password to confirm account deletion.")
        return redirect("account_settings")

    if not request.user.check_password(confirm_password):
        messages.error(request, "Incorrect password. Account was not deleted.")
        return redirect("account_settings")

    user_id = request.user.pk
    try:
        User.objects.filter(pk=user_id).delete()
    except Exception as e:
        logger.exception("Failed to delete account pk=%s: %s", user_id, e)
        messages.error(request, "An error occurred while deleting your account. Please try again.")
        return redirect("account_settings")

    logout(request)
    messages.success(request, "Your account has been permanently deleted.")
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

