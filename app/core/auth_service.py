"""Authentication and Role-Based Access Control (RBAC) Service for NODE SENTINEL.

Provides secure password hashing (PBKDF2-HMAC-SHA256), signed tamper-proof bearer tokens,
role/permission verification, user repository management, and FastAPI security dependencies.
Zero third-party crypto dependencies required (pure standard library: hashlib, hmac, secrets).
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import logging
from pathlib import Path
import secrets
import threading
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.core.audit_logger import audit_logger
from app.models.audit_models import AuditAction
from app.models.auth_models import (
    CreateUserRequest,
    LoginRequest,
    LoginResponse,
    Permission,
    ROLE_PERMISSIONS,
    Role,
    TokenPayload,
    UpdateUserRequest,
    User,
    UserResponse,
)

logger = logging.getLogger(__name__)

# FastAPI HTTPBearer scheme with auto_error=False to allow optional token handling
bearer_scheme = HTTPBearer(auto_error=False)


# ==============================================================================
# 1. CRYPTO: PASSWORD HASHER (PBKDF2-HMAC-SHA256)
# ==============================================================================

class PasswordHasher:
    """Standard-library PBKDF2 password hasher with random salt."""

    ITERATIONS = 100_000

    @classmethod
    def hash_password(cls, plain_password: str) -> str:
        """Hash password using PBKDF2-HMAC-SHA256 with a unique 16-byte salt."""
        salt = secrets.token_bytes(16)
        key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, cls.ITERATIONS)
        salt_hex = salt.hex()
        key_hex = key.hex()
        return f"pbkdf2_sha256${cls.ITERATIONS}${salt_hex}${key_hex}"

    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        """Verify plain password against hashed string using constant-time comparison."""
        try:
            parts = hashed_password.split("$")
            if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
                return False
            iterations = int(parts[1])
            salt = bytes.fromhex(parts[2])
            expected_key = bytes.fromhex(parts[3])
            derived_key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(derived_key, expected_key)
        except Exception:
            return False


# ==============================================================================
# 2. CRYPTO: TOKEN MANAGER (SIGNED COMPACT BEARER TOKENS)
# ==============================================================================

class TokenManager:
    """HMAC-SHA256 signed bearer token generator and validator."""

    def __init__(self, secret_key: str, expire_minutes: int):
        self.secret_key = secret_key.encode("utf-8")
        self.expire_minutes = expire_minutes
        self._revoked_tokens: Set[str] = set()
        self._lock = threading.RLock()

    def _b64encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    def _b64decode(self, data: str) -> bytes:
        padding = 4 - (len(data) % 4)
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Generate a signed bearer token containing sub, role, and expiry."""
        now = datetime.now(timezone.utc)
        delta = expires_delta or timedelta(minutes=self.expire_minutes)
        expire = now + delta

        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": uuid.uuid4().hex,
        }

        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        payload_b64 = self._b64encode(payload_bytes)

        sig = hmac.new(self.secret_key, payload_b64.encode("utf-8"), hashlib.sha256).digest()
        sig_b64 = self._b64encode(sig)

        return f"{payload_b64}.{sig_b64}"

    def decode_access_token(self, token: str) -> Optional[TokenPayload]:
        """Validate signature and expiration of bearer token."""
        if not token or "." not in token:
            return None

        with self._lock:
            if token in self._revoked_tokens:
                return None

        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None
            payload_b64, sig_b64 = parts

            # Verify signature
            expected_sig = hmac.new(self.secret_key, payload_b64.encode("utf-8"), hashlib.sha256).digest()
            actual_sig = self._b64decode(sig_b64)

            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            payload_bytes = self._b64decode(payload_b64)
            payload = json.loads(payload_bytes.decode("utf-8"))

            exp = payload.get("exp", 0)
            now_ts = int(datetime.now(timezone.utc).timestamp())
            if now_ts > exp:
                return None

            return TokenPayload(
                sub=payload["sub"],
                username=payload["username"],
                role=Role(payload["role"]),
                exp=exp,
                iat=payload.get("iat", now_ts),
                jti=payload.get("jti"),
            )
        except Exception as e:
            logger.debug(f"Token decode error: {e}")
            return None

    def revoke_token(self, token: str) -> None:
        """Add token to revocation blacklist."""
        with self._lock:
            self._revoked_tokens.add(token)


token_manager = TokenManager(
    secret_key=settings.SECRET_KEY,
    expire_minutes=settings.AUTH_TOKEN_EXPIRE_MINUTES,
)


# ==============================================================================
# 3. USER REPOSITORY (PERSISTENT JSON STORAGE)
# ==============================================================================

class UserRepository:
    """Thread-safe persistent storage for system users."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = Path(file_path or settings.USERS_FILE)
        self._lock = threading.RLock()
        self._ensure_seed_users()

    def _ensure_seed_users(self) -> None:
        """Seed default users if users file does not exist or is empty."""
        with self._lock:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists() or self.file_path.stat().st_size == 0:
                if not settings.DEMO_MODE:
                    raise RuntimeError(
                        "No user store is configured. Set USERS_FILE to a persistent, access-controlled user store "
                        "before starting a non-demo deployment."
                    )
                logger.info("Initializing users database with seed accounts...")
                now_str = datetime.now(timezone.utc).isoformat()
                seed_users = [
                    User(
                        user_id="usr_admin",
                        username="admin",
                        full_name="System Administrator",
                        role=Role.ADMIN,
                        password_hash=PasswordHasher.hash_password("AdminPassword123!"),
                        is_active=True,
                        created_at=now_str,
                    ),
                    User(
                        user_id="usr_investigator",
                        username="investigator",
                        full_name="Lead Investigator",
                        role=Role.INVESTIGATOR,
                        password_hash=PasswordHasher.hash_password("Investigator123!"),
                        is_active=True,
                        created_at=now_str,
                    ),
                    User(
                        user_id="usr_analyst",
                        username="analyst",
                        full_name="Intelligence Analyst",
                        role=Role.ANALYST,
                        password_hash=PasswordHasher.hash_password("Analyst123!"),
                        is_active=True,
                        created_at=now_str,
                    ),
                    User(
                        user_id="usr_viewer",
                        username="viewer",
                        full_name="Auditor / Viewer",
                        role=Role.VIEWER,
                        password_hash=PasswordHasher.hash_password("Viewer123!"),
                        is_active=True,
                        created_at=now_str,
                    ),
                ]
                self._save_users_locked({u.user_id: u for u in seed_users})

    def _load_users_locked(self) -> Dict[str, User]:
        """Load users mapping from disk. Caller MUST hold self._lock."""
        if not self.file_path.exists():
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                data = json.loads(content)
                return {item["user_id"]: User(**item) for item in data}
        except Exception as e:
            logger.error(f"Failed to load users from {self.file_path}: {e}")
            return {}

    def _save_users_locked(self, users: Dict[str, User]) -> None:
        """Save users mapping atomically to disk. Caller MUST hold self._lock."""
        tmp_path = self.file_path.with_suffix(".tmp")
        try:
            user_list = [u.model_dump() for u in users.values()]
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(user_list, f, indent=2, default=str)
            # Resilient atomic replacement for Windows and POSIX
            for attempt in range(5):
                try:
                    tmp_path.replace(self.file_path)
                    break
                except (PermissionError, OSError):
                    if attempt < 4:
                        import time
                        time.sleep(0.02 * (attempt + 1))
                    else:
                        import shutil
                        shutil.copyfile(str(tmp_path), str(self.file_path))
                        tmp_path.unlink(missing_ok=True)
                        break
        except Exception as e:
            logger.error(f"Failed to save users atomically to {self.file_path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def get_by_id(self, user_id: str) -> Optional[User]:
        with self._lock:
            users = self._load_users_locked()
            return users.get(user_id)

    def get_by_username(self, username: str) -> Optional[User]:
        with self._lock:
            users = self._load_users_locked()
            uname_lower = username.lower()
            for u in users.values():
                if u.username.lower() == uname_lower:
                    return u
            return None

    def list_users(self) -> List[User]:
        with self._lock:
            users = self._load_users_locked()
            return sorted(users.values(), key=lambda u: u.created_at)

    def create_user(self, req: CreateUserRequest) -> User:
        with self._lock:
            users = self._load_users_locked()
            if any(u.username.lower() == req.username.lower() for u in users.values()):
                raise ValueError(f"Username '{req.username}' already exists")

            user_id = f"usr_{uuid.uuid4().hex[:8]}"
            user = User(
                user_id=user_id,
                username=req.username.strip(),
                full_name=req.full_name.strip(),
                role=req.role,
                password_hash=PasswordHasher.hash_password(req.password),
                is_active=req.is_active,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            users[user_id] = user
            self._save_users_locked(users)
            return user

    def update_user(self, user_id: str, req: UpdateUserRequest) -> Optional[User]:
        with self._lock:
            users = self._load_users_locked()
            user = users.get(user_id)
            if not user:
                return None

            if req.full_name is not None:
                user.full_name = req.full_name.strip()
            if req.role is not None:
                user.role = req.role
            if req.is_active is not None:
                user.is_active = req.is_active
            if req.password:
                user.password_hash = PasswordHasher.hash_password(req.password)

            users[user_id] = user
            self._save_users_locked(users)
            return user

    def update_last_login(self, user_id: str) -> None:
        with self._lock:
            users = self._load_users_locked()
            user = users.get(user_id)
            if user:
                user.last_login = datetime.now(timezone.utc).isoformat()
                users[user_id] = user
                self._save_users_locked(users)

    def delete_user(self, user_id: str) -> bool:
        with self._lock:
            users = self._load_users_locked()
            if user_id not in users:
                return False
            del users[user_id]
            self._save_users_locked(users)
            return True


user_repo = UserRepository()


# ==============================================================================
# 4. AUTH SERVICE (COORDINATOR)
# ==============================================================================

class AuthService:
    """Coordinates authentication flows, session creation, and audit logging."""

    def __init__(self, repo: UserRepository = user_repo, tokens: TokenManager = token_manager):
        self.repo = repo
        self.tokens = tokens

    def authenticate_user(self, username: str, plain_password: str) -> Optional[User]:
        user = self.repo.get_by_username(username)
        if not user or not user.is_active:
            return None
        if not PasswordHasher.verify_password(plain_password, user.password_hash):
            return None
        return user

    def login(self, request: LoginRequest, ip: Optional[str] = None, user_agent: Optional[str] = None) -> LoginResponse:
        user = self.authenticate_user(request.username, request.password)
        if not user:
            # Audit failed login attempt
            audit_logger.log(
                action=AuditAction.AUTH_LOGIN_FAILED,
                username=request.username,
                ip_address=ip,
                user_agent=user_agent,
                status="FAILED",
                details={"reason": "Invalid username or password"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = self.tokens.create_access_token(user)
        self.repo.update_last_login(user.user_id)

        # Audit successful login
        audit_logger.log(
            action=AuditAction.AUTH_LOGIN,
            user_id=user.user_id,
            username=user.username,
            role=user.role,
            ip_address=ip,
            user_agent=user_agent,
            status="SUCCESS",
        )

        permissions = [p.value for p in ROLE_PERMISSIONS.get(user.role, set())]
        user_res = UserResponse.from_user(user, permissions=permissions)

        return LoginResponse(
            access_token=token,
            token_type="bearer",
            expires_in=settings.AUTH_TOKEN_EXPIRE_MINUTES * 60,
            user=user_res,
        )

    def logout(self, token: str, user: Optional[User] = None, ip: Optional[str] = None, user_agent: Optional[str] = None) -> None:
        self.tokens.revoke_token(token)
        if user:
            audit_logger.log(
                action=AuditAction.AUTH_LOGOUT,
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                ip_address=ip,
                user_agent=user_agent,
                status="SUCCESS",
            )


auth_service = AuthService()


# ==============================================================================
# 5. FASTAPI DEPENDENCIES & AUTHORIZATION ENFORCEMENT
# ==============================================================================

def get_token_from_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """Extract token from Authorization header or cookie."""
    if credentials and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    # Fallback to cookie if present
    cookie_token = request.cookies.get("nodesentinel_token")
    if cookie_token:
        return cookie_token
    return None


def get_current_user_optional(
    request: Request,
    token: Optional[str] = Depends(get_token_from_request),
) -> Optional[User]:
    """Return authenticated User if token is valid, else None (without raising 401)."""
    if not token:
        return None
    payload = token_manager.decode_access_token(token)
    if not payload:
        return None
    user = user_repo.get_by_id(payload.sub)
    if not user or not user.is_active:
        return None
    return user


def get_current_user(
    request: Request,
    token: Optional[str] = Depends(get_token_from_request),
) -> User:
    """Strictly require authenticated User. Raises HTTP 401 if missing or invalid."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = token_manager.decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = user_repo.get_by_id(payload.sub)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )
    return user


def require_permission(permission: Permission) -> Callable[..., User]:
    """Dependency factory checking that the user possesses a specific Permission."""
    def permission_checker(
        request: Request,
        user: User = Depends(get_current_user),
    ) -> User:
        allowed_perms = ROLE_PERMISSIONS.get(user.role, set())
        if permission not in allowed_perms:
            ip = request.client.host if request.client else None
            audit_logger.log(
                action=AuditAction.RBAC_ACCESS_DENIED,
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                resource_type="permission",
                resource_id=permission.value,
                ip_address=ip,
                status="DENIED",
                details={"required_permission": permission.value, "user_role": user.role.value},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: role '{user.role.value}' lacks permission '{permission.value}'",
            )
        return user

    return permission_checker


def enforce_permission(permission: Permission) -> Callable[..., Optional[User]]:
    """Enforce a permission whenever authentication is enabled.

    Local demo mode intentionally remains open for presentations and the legacy
    offline test fixtures.  Deployments set REQUIRE_AUTH=true and are then
    protected server-side, rather than relying on the browser to hide controls.
    """
    def permission_checker(
        request: Request,
        user: Optional[User] = Depends(get_current_user_optional),
    ) -> Optional[User]:
        if not settings.REQUIRE_AUTH:
            return user
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        allowed_perms = ROLE_PERMISSIONS.get(user.role, set())
        if permission not in allowed_perms:
            ip = request.client.host if request.client else None
            audit_logger.log(
                action=AuditAction.RBAC_ACCESS_DENIED,
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                resource_type="permission",
                resource_id=permission.value,
                ip_address=ip,
                status="DENIED",
                details={"required_permission": permission.value, "user_role": user.role.value},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: role '{user.role.value}' lacks permission '{permission.value}'",
            )
        return user

    return permission_checker


def require_role(*allowed_roles: Role) -> Callable[..., User]:
    """Dependency factory checking that user role is among the allowed roles."""
    def role_checker(
        request: Request,
        user: User = Depends(get_current_user),
    ) -> User:
        if user.role not in allowed_roles:
            ip = request.client.host if request.client else None
            audit_logger.log(
                action=AuditAction.RBAC_ACCESS_DENIED,
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                resource_type="role",
                resource_id=",".join(r.value for r in allowed_roles),
                ip_address=ip,
                status="DENIED",
                details={"allowed_roles": [r.value for r in allowed_roles], "user_role": user.role.value},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: role '{user.role.value}' is not authorized for this operation",
            )
        return user

    return role_checker
