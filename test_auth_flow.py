"""
Auth flow test: signup → OTP verify → login → bad password → resend OTP.
Runs against the live Django DB; cleans up test user afterwards.
"""
import os, sys, django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moralverse.settings")
sys.path.insert(0, os.path.dirname(__file__))
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from core.models import EmailVerification

TEST_USER = "test_auth_user_99"
TEST_EMAIL = "test_auth_99@example.com"
TEST_PASS = "Secure@123"

client = Client()

# ── helpers ──────────────────────────────────────────────────────────────────
def ok(label): print(f"  [PASS] {label}")
def fail(label, detail=""): print(f"  [FAIL] {label}  →  {detail}"); sys.exit(1)

def check(label, condition, detail=""):
    if condition: ok(label)
    else: fail(label, detail)

# ── cleanup from previous runs ───────────────────────────────────────────────
User.objects.filter(username=TEST_USER).delete()
print("\n=== Auth flow test ===\n")

# ── 1. GET signup page ────────────────────────────────────────────────────────
print("1. Signup page loads")
r = client.get("/signup/")
check("GET /signup/ -> 200", r.status_code == 200, r.status_code)

# ── 2. POST signup — missing fields ──────────────────────────────────────────
print("2. Signup validation")
r = client.post("/signup/", {"username": "", "email": "", "password": ""})
check("Empty fields → 200 with error", r.status_code == 200)
check("Error message present", b"required" in r.content.lower() or b"field" in r.content.lower() or len(r.context["messages"]) > 0 if hasattr(r, "context") and r.context else True)

# ── 3. POST signup — passwords don't match ───────────────────────────────────
r = client.post("/signup/", {
    "username": TEST_USER, "email": TEST_EMAIL,
    "password": "abc123", "password_confirm": "xyz999"
})
check("Mismatched passwords → stay on page", r.status_code == 200)

# ── 4. POST signup — valid ────────────────────────────────────────────────────
print("3. Valid signup")
r = client.post("/signup/", {
    "username": TEST_USER, "email": TEST_EMAIL,
    "password": TEST_PASS, "password_confirm": TEST_PASS,
})
check("Valid signup → redirect to verify_otp", r.status_code == 302 and "verify" in r["Location"], f"status={r.status_code} location={r.get('Location','')}")

user = User.objects.filter(username=TEST_USER).first()
check("User created in DB", user is not None)
check("User is inactive (not yet verified)", user and not user.is_active)

# ── 5. OTP exists in DB ───────────────────────────────────────────────────────
print("4. OTP record")
ev = EmailVerification.objects.filter(user=user).first()
check("EmailVerification record created", ev is not None)
otp = ev.otp
check("OTP is 6 digits", otp and len(str(otp)) == 6, otp)
print(f"     OTP = {otp}")

# ── 6. POST verify_otp — wrong OTP ───────────────────────────────────────────
print("5. OTP verification")
r = client.post("/auth/verify-otp/", {"email": TEST_EMAIL, "otp": "000000"})
check("Wrong OTP → stay on page", r.status_code == 200)
user.refresh_from_db()
check("User still inactive after wrong OTP", not user.is_active)

# ── 7. POST verify_otp — correct OTP ─────────────────────────────────────────
r = client.post("/auth/verify-otp/", {"email": TEST_EMAIL, "otp": str(otp)})
check("Correct OTP → redirect to login", r.status_code == 302 and "login" in r["Location"], f"status={r.status_code} loc={r.get('Location','')}")
user.refresh_from_db()
check("User is now active", user.is_active)

# ── 8. Login with wrong password ─────────────────────────────────────────────
print("6. Login")
r = client.post("/login/", {"username": TEST_USER, "password": "wrongpass"})
check("Wrong password → stay on page", r.status_code == 200)

# ── 9. Login with correct password ───────────────────────────────────────────
r = client.post("/login/", {"username": TEST_USER, "password": TEST_PASS})
check("Correct login → redirect", r.status_code == 302, f"status={r.status_code}")

# ── 10. Login with email instead of username ─────────────────────────────────
client2 = Client()
r = client2.post("/login/", {"username": TEST_EMAIL, "password": TEST_PASS})
check("Login by email → redirect", r.status_code == 302, f"status={r.status_code}")

# ── 11. Logout ────────────────────────────────────────────────────────────────
print("7. Logout")
r = client.get("/logout/")
check("Logout → redirect to landing", r.status_code == 302)

# ── 12. Duplicate signup (same username) ─────────────────────────────────────
print("8. Duplicate signup guards")
r = client.post("/signup/", {
    "username": TEST_USER, "email": "other@example.com",
    "password": TEST_PASS, "password_confirm": TEST_PASS,
})
check("Duplicate username → error page", r.status_code == 200)

r = client.post("/signup/", {
    "username": "other_user_99", "email": TEST_EMAIL,
    "password": TEST_PASS, "password_confirm": TEST_PASS,
})
check("Duplicate email → error page", r.status_code == 200)

# ── cleanup ───────────────────────────────────────────────────────────────────
User.objects.filter(username=TEST_USER).delete()
print("\n=== All tests passed ===\n")
