# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Authentication & RBAC Data Models (STEP 15)
Defines roles, granular permissions, user entities, login contracts, and token structures.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    ADMIN = "ADMIN"
    INVESTIGATOR = "INVESTIGATOR"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"


class Permission(str, Enum):
    VIEW_GRAPH = "VIEW_GRAPH"
    UNIVERSAL_SEARCH = "UNIVERSAL_SEARCH"
    FACE_SEARCH = "FACE_SEARCH"
    CDR_ANALYSIS = "CDR_ANALYSIS"
    FINANCIAL_ANALYSIS = "FINANCIAL_ANALYSIS"
    TIMELINE = "TIMELINE"
    RISK_ANALYSIS = "RISK_ANALYSIS"
    AI_ASSISTANT = "AI_ASSISTANT"
    GENERATE_REPORT = "GENERATE_REPORT"
    INGEST_DATA = "INGEST_DATA"
    MANAGE_USERS = "MANAGE_USERS"
    VIEW_AUDIT_LOG = "VIEW_AUDIT_LOG"
    VIEW_SENSITIVE_DATA = "VIEW_SENSITIVE_DATA"

    # Convenient Aliases
    USER_MANAGEMENT = "MANAGE_USERS"
    AUDIT_VIEW = "VIEW_AUDIT_LOG"
    DATA_INGEST = "INGEST_DATA"
    GRAPH_VIEW = "VIEW_GRAPH"
    REPORT_GENERATE = "GENERATE_REPORT"
    FINANCIAL_ANALYZE = "FINANCIAL_ANALYSIS"
    CDR_ANALYZE = "CDR_ANALYSIS"


# Centralized Role-Permission Mapping Matrix
ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
    Role.ADMIN: [
        Permission.VIEW_GRAPH,
        Permission.UNIVERSAL_SEARCH,
        Permission.FACE_SEARCH,
        Permission.CDR_ANALYSIS,
        Permission.FINANCIAL_ANALYSIS,
        Permission.TIMELINE,
        Permission.RISK_ANALYSIS,
        Permission.AI_ASSISTANT,
        Permission.GENERATE_REPORT,
        Permission.INGEST_DATA,
        Permission.MANAGE_USERS,
        Permission.VIEW_AUDIT_LOG,
        Permission.VIEW_SENSITIVE_DATA,
    ],
    Role.INVESTIGATOR: [
        Permission.VIEW_GRAPH,
        Permission.UNIVERSAL_SEARCH,
        Permission.FACE_SEARCH,
        Permission.CDR_ANALYSIS,
        Permission.FINANCIAL_ANALYSIS,
        Permission.TIMELINE,
        Permission.RISK_ANALYSIS,
        Permission.AI_ASSISTANT,
        Permission.GENERATE_REPORT,
        Permission.INGEST_DATA,
        Permission.VIEW_SENSITIVE_DATA,
    ],
    Role.ANALYST: [
        Permission.VIEW_GRAPH,
        Permission.UNIVERSAL_SEARCH,
        Permission.CDR_ANALYSIS,
        Permission.FINANCIAL_ANALYSIS,
        Permission.TIMELINE,
        Permission.RISK_ANALYSIS,
        Permission.AI_ASSISTANT,
        Permission.GENERATE_REPORT,
    ],
    Role.VIEWER: [
        Permission.VIEW_GRAPH,
        Permission.UNIVERSAL_SEARCH,
        Permission.TIMELINE,
        Permission.VIEW_AUDIT_LOG,
    ],
}


class User(BaseModel):
    """Internal user record stored in repository."""
    user_id: str = Field(..., description="Unique user identifier, e.g. USR-ADMIN-01")
    username: str = Field(..., description="Unique login username")
    full_name: Optional[str] = Field(None, description="User's full display name")
    password_hash: str = Field(..., description="Salted PBKDF2-HMAC-SHA256 password hash")
    role: Role = Field(Role.VIEWER, description="Assigned role")
    is_active: bool = Field(True, description="Whether user account is active/enabled")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_login: Optional[str] = Field(None, description="Timestamp of most recent successful login")


class UserResponse(BaseModel):
    """Safe user profile representation returned in public APIs (never reveals password_hash)."""
    user_id: str
    username: str
    full_name: Optional[str] = None
    role: Role
    is_active: bool
    created_at: str
    updated_at: Optional[str] = None
    last_login: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)

    @classmethod
    def from_user(cls, user: User, permissions: Optional[List[str]] = None) -> UserResponse:
        perms = permissions or [p.value for p in ROLE_PERMISSIONS.get(user.role, [])]
        return cls(
            user_id=user.user_id,
            username=user.username,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login=user.last_login,
            permissions=perms,
        )


class LoginRequest(BaseModel):
    """Credentials payload submitted at login."""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")


class LoginResponse(BaseModel):
    """Authentication token response returned upon successful authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(28800, description="Token validity in seconds (default 8 hours)")
    user: UserResponse
    permissions: List[str] = Field(default_factory=list)


class CreateUserRequest(BaseModel):
    """Admin request to create a new user account."""
    username: str = Field(..., min_length=3, max_length=50, description="Desired username")
    password: str = Field(..., min_length=6, description="Initial password")
    full_name: Optional[str] = Field(None, description="User full display name")
    role: Role = Field(Role.VIEWER, description="Initial role assignment")
    is_active: bool = Field(True, description="Initial active state")


class UpdateUserRequest(BaseModel):
    """Admin request to update user role, enabled status, full name, or reset password."""
    full_name: Optional[str] = None
    role: Optional[Role] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=6, description="New password if resetting")


class TokenPayload(BaseModel):
    """Decoded bearer token claims."""
    sub: Optional[str] = None
    user_id: Optional[str] = None
    username: str
    role: Role
    exp: int
    iat: int
    jti: Optional[str] = None

