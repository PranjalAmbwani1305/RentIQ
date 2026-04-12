"""
core/security.py — RentIQ Phase 1
────────────────────────────────────────────────────────────
Security stack:
  • BCrypt (cost=12)  — password hashing, brute-force resistant
  • HMAC-SHA256 JWT   — stateless session tokens
  • RBAC bitfields    — per-role permission enforcement
  • Token bucket      — IP-level rate limiting on login
  • Account lockout   — 5 failed attempts → 15-min freeze

Role hierarchy:
  SUPER_ADMIN  → every permission bit set
  ADMIN        → manage users, view all data, export, admin dashboard
  ANALYST      → view all data, run models, export
  AGENT        → predictions, compare, create leads
  TENANT       → predictions only

Admin isolation rule:
  ADMIN + SUPER_ADMIN see ONLY:  Admin panel, Models registry
  All other roles see ONLY:      Predictor, Similar, Analytics (by perm)
────────────────────────────────────────────────────────────
"""

import os, time, uuid, hmac, hashlib, logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
import jwt
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("RentIQ.Security")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ══════════════════════════════════════════════════════════
#  1.  Permission Bitfield
# ══════════════════════════════════════════════════════════
class Perm:
    VIEW_PREDICTIONS = 1 << 0   #   1  — run rent predictor
    COMPARE_LISTINGS = 1 << 1   #   2  — vector similarity search
    VIEW_ALL_DATA    = 1 << 2   #   4  — analytics dashboard
    RUN_MODELS       = 1 << 3   #   8  — model registry page
    MANAGE_USERS     = 1 << 4   #  16  — user management tab
    EXPORT_DATA      = 1 << 5   #  32  — CSV download button
    ADMIN_DASHBOARD  = 1 << 7   # 128  — admin panel access
    CREATE_LEADS     = 1 << 8   # 256  — agent lead creation


ROLE_PERMS = {
    "SUPER_ADMIN": sum(1 << i for i in range(10)),   # all bits
    "ADMIN": (
        Perm.VIEW_ALL_DATA | Perm.RUN_MODELS | Perm.MANAGE_USERS |
        Perm.EXPORT_DATA   | Perm.ADMIN_DASHBOARD
    ),
    "ANALYST": (
        Perm.VIEW_ALL_DATA | Perm.RUN_MODELS | Perm.EXPORT_DATA |
        Perm.VIEW_PREDICTIONS | Perm.COMPARE_LISTINGS
    ),
    "AGENT": (
        Perm.VIEW_PREDICTIONS | Perm.COMPARE_LISTINGS | Perm.CREATE_LEADS
    ),
    "TENANT": Perm.VIEW_PREDICTIONS,
}

# Roles whose nav is LOCKED to admin-only pages
ADMIN_ROLES = {"SUPER_ADMIN", "ADMIN"}


# ══════════════════════════════════════════════════════════
#  2.  JWT Manager
# ══════════════════════════════════════════════════════════
class JWTManager:
    ALGO = "HS256"

    def __init__(self):
        secret = os.getenv("JWT_SECRET", "")
        if not secret or len(secret) < 32:
            import secrets
            secret = secrets.token_hex(32)
            log.warning("JWT_SECRET not set — using ephemeral key (dev only)")
        self._secret = secret
        self._exp_h  = int(os.getenv("JWT_EXPIRY_HOURS", "8"))
        self._ref_d  = int(os.getenv("JWT_REFRESH_DAYS", "7"))

    def create_access_token(self, user_id: str, role: str, perms: int) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub":   user_id,
            "role":  role,
            "perms": perms,
            "iat":   now,
            "exp":   now + timedelta(hours=self._exp_h),
            "jti":   str(uuid.uuid4()),
            "type":  "access",
        }
        return jwt.encode(payload, self._secret, algorithm=self.ALGO)

    def create_refresh_token(self, user_id: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub":  user_id,
            "iat":  now,
            "exp":  now + timedelta(days=self._ref_d),
            "jti":  str(uuid.uuid4()),
            "type": "refresh",
        }
        return jwt.encode(payload, self._secret, algorithm=self.ALGO)

    def decode(self, token: str) -> Optional[dict]:
        try:
            return jwt.decode(token, self._secret, algorithms=[self.ALGO])
        except jwt.ExpiredSignatureError:
            log.debug("Token expired")
        except jwt.InvalidTokenError as e:
            log.debug(f"Invalid token: {e}")
        return None


jwt_manager = JWTManager()


# ══════════════════════════════════════════════════════════
#  3.  Password Manager
# ══════════════════════════════════════════════════════════
class PasswordManager:
    COST = 12

    def hash(self, plaintext: str) -> str:
        return bcrypt.hashpw(
            plaintext.encode("utf-8"),
            bcrypt.gensalt(rounds=self.COST)
        ).decode("utf-8")

    def verify(self, plaintext: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(plaintext.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False


pwd_manager = PasswordManager()


# ══════════════════════════════════════════════════════════
#  4.  Token Bucket Rate Limiter (per IP)
# ══════════════════════════════════════════════════════════
class TokenBucketLimiter:
    def __init__(self, capacity=10, rate=1.0, cost=2):
        self._buckets: dict = {}
        self.capacity = capacity
        self.rate     = rate   # tokens/second
        self.cost     = cost

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        b   = self._buckets.setdefault(key, {"tokens": self.capacity, "last": now})
        elapsed      = now - b["last"]
        b["tokens"]  = min(self.capacity, b["tokens"] + elapsed * self.rate)
        b["last"]    = now
        if b["tokens"] >= self.cost:
            b["tokens"] -= self.cost
            return True, 0.0
        retry = (self.cost - b["tokens"]) / self.rate
        return False, retry


login_limiter = TokenBucketLimiter(capacity=10, rate=0.5, cost=2)


# ══════════════════════════════════════════════════════════
#  5.  In-memory User Store  (replace with MongoDB in Phase 2)
# ══════════════════════════════════════════════════════════
_USERS: dict = {}

SEED_USERS = [
    {"username": "superadmin",  "password": "Admin@RentIQ#2024", "role": "SUPER_ADMIN",
     "full_name": "Platform Administrator", "email": "admin@rentiq.ai"},
    {"username": "admin1",      "password": "Admin@456!",        "role": "ADMIN",
     "full_name": "Neha Kapoor",            "email": "neha@rentiq.ai"},
    {"username": "analyst1",    "password": "Analyst@123!",      "role": "ANALYST",
     "full_name": "Priya Sharma",           "email": "priya@rentiq.ai"},
    {"username": "agent1",      "password": "Agent@456!",        "role": "AGENT",
     "full_name": "Rahul Mehta",            "email": "rahul@rentiq.ai"},
    {"username": "tenant1",     "password": "Tenant@789!",       "role": "TENANT",
     "full_name": "Arjun Kumar",            "email": "arjun@rentiq.ai"},
]


def seed_users():
    """Hash passwords and populate the in-memory user store."""
    for u in SEED_USERS:
        _USERS[u["username"]] = {
            "id":               str(uuid.uuid4()),
            "username":         u["username"],
            "password_hash":    pwd_manager.hash(u["password"]),
            "role":             u["role"],
            "permissions":      ROLE_PERMS[u["role"]],
            "full_name":        u["full_name"],
            "email":            u["email"],
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "failed_attempts":  0,
            "locked_until":     None,
            "last_login":       None,
        }
    log.info(f"Seeded {len(_USERS)} users into auth store")


# ══════════════════════════════════════════════════════════
#  6.  Auth Service
# ══════════════════════════════════════════════════════════
def login(username: str, password: str, ip: str = "web") -> Optional[dict]:
    """
    Full login flow:
      1. IP rate limit check
      2. User lookup
      3. Account lockout check
      4. BCrypt verify
      5. Issue JWT tokens
    Returns dict with tokens + role info, or dict with "error" key.
    """
    allowed, retry = login_limiter.allow(ip)
    if not allowed:
        return {"error": f"Too many attempts. Retry in {retry:.0f}s.", "code": 429}

    user = _USERS.get(username.strip().lower()) or _USERS.get(username.strip())
    if not user:
        log.info(f"LOGIN_FAIL unknown_user={username} ip={ip}")
        return {"error": "Invalid username or password.", "code": 401}

    lock = user.get("locked_until")
    if lock and time.time() < lock:
        return {"error": f"Account locked. Retry in {lock - time.time():.0f}s.", "code": 403}

    if not pwd_manager.verify(password, user["password_hash"]):
        user["failed_attempts"] += 1
        max_att = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
        if user["failed_attempts"] >= max_att:
            user["locked_until"] = time.time() + int(os.getenv("LOCK_DURATION_SECONDS", "900"))
            log.warning(f"ACCOUNT_LOCKED user={username}")
        log.info(f"LOGIN_FAIL bad_password user={username} attempts={user['failed_attempts']}")
        return {"error": "Invalid username or password.", "code": 401}

    user["failed_attempts"] = 0
    user["locked_until"]    = None
    user["last_login"]      = datetime.now(timezone.utc).isoformat()

    access  = jwt_manager.create_access_token(user["id"], user["role"], user["permissions"])
    refresh = jwt_manager.create_refresh_token(user["id"])

    log.info(f"LOGIN_OK user={username} role={user['role']} ip={ip}")
    return {
        "access_token":  access,
        "refresh_token": refresh,
        "role":          user["role"],
        "username":      username,
        "display_name":  user["full_name"],
        "permissions":   user["permissions"],
        "is_admin_role": user["role"] in ADMIN_ROLES,
    }


def verify_token(token: str) -> Optional[dict]:
    """Decode and validate an access token. Returns payload or None."""
    payload = jwt_manager.decode(token)
    if payload and payload.get("type") == "access":
        return payload
    return None


# ══════════════════════════════════════════════════════════
#  7.  Registration Service
# ══════════════════════════════════════════════════════════
import re as _re

_EMAIL_RE = _re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PWD_RE   = _re.compile(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$")

# Roles that self-registration is allowed to request
SELF_REGISTER_ROLES = {"TENANT", "AGENT"}


def register_user(
    username: str,
    password: str,
    full_name: str,
    email: str,
    role: str = "TENANT",
) -> dict:
    """
    Self-registration flow:
      1. Input validation (username, email, password strength)
      2. Duplicate check
      3. Role whitelist (only TENANT / AGENT allowed via self-reg)
      4. BCrypt hash + insert into in-memory store
    Returns dict with success flag, or dict with \"error\" key.
    """
    username  = username.strip()
    full_name = full_name.strip()
    email     = email.strip().lower()
    role      = role.upper()

    # — Validation ——————————————————————————————————————
    if not username or len(username) < 3:
        return {"error": "Username must be at least 3 characters."}
    if not _re.match(r"^\w+$", username):
        return {"error": "Username may only contain letters, digits and underscores."}
    if not full_name:
        return {"error": "Full name is required."}
    if not _EMAIL_RE.match(email):
        return {"error": "Please enter a valid email address."}
    if not _PWD_RE.match(password):
        return {
            "error": (
                "Password must be at least 8 characters and contain "
                "an uppercase letter, a lowercase letter, a digit, "
                "and a special character (!@#$%^&*)."
            )
        }
    if role not in SELF_REGISTER_ROLES:
        return {"error": f"Role '{role}' is not available for self-registration."}

    # — Duplicate check ————————————————————————————————
    if username.lower() in {u.lower() for u in _USERS}:
        return {"error": "Username already taken. Please choose another."}
    if email in {u.get("email", "").lower() for u in _USERS.values()}:
        return {"error": "An account with this email already exists."}

    # — Create user ————————————————————————————————————
    _USERS[username] = {
        "id":              str(uuid.uuid4()),
        "username":        username,
        "password_hash":   pwd_manager.hash(password),
        "role":            role,
        "permissions":     ROLE_PERMS[role],
        "full_name":       full_name,
        "email":           email,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "failed_attempts": 0,
        "locked_until":    None,
        "last_login":      None,
    }
    log.info(f"REGISTER_OK user={username} role={role} email={email}")
    return {"success": True, "username": username, "role": role}

