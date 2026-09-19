"""Authentication API endpoints for NODE SENTINEL."""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status

from app.config import settings
from app.core.audit_logger import audit_logger
from app.core.auth_service import (
    auth_service,
    get_current_user,
    get_token_from_request,
    require_permission,
)
from app.core.graph_engine import get_graph_engine
from app.models.audit_models import AuditAction
from app.models.auth_models import (
    LoginRequest,
    LoginResponse,
    Permission,
    ROLE_PERMISSIONS,
    User,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class RevealSensitiveRequest(BaseModel):
    entity_id: Optional[str] = None
    identifier_value: Optional[str] = None
    identifier_type: Optional[str] = None
    entity_type: Optional[str] = None


class RevealSensitiveResponse(BaseModel):
    entity_id: Optional[str] = None
    unmasked_value: str
    authorized_by: Optional[str] = None
    role: Optional[str] = None
    audit_logged: bool = True
    authorized: bool = True


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    response: Response,
    payload: LoginRequest,
) -> LoginResponse:
    """Authenticate user with username and password, returning a signed bearer token."""
    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    login_res = auth_service.login(payload, ip=ip, user_agent=user_agent)

    # Set secure HTTP-only cookie for optional browser session persistence
    response.set_cookie(
        key="nodesentinel_token",
        value=login_res.access_token,
        max_age=login_res.expires_in,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="strict",
        path="/",
    )

    return login_res


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    token: Optional[str] = Depends(get_token_from_request),
) -> dict:
    """Revoke current access token and clear session."""
    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    if token:
        auth_service.logout(token, user=current_user, ip=ip, user_agent=user_agent)

    response.delete_cookie(key="nodesentinel_token")
    return {"status": "success", "message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Retrieve profile and assigned permissions for the currently authenticated user."""
    permissions = [p.value for p in ROLE_PERMISSIONS.get(current_user.role, set())]
    return UserResponse.from_user(current_user, permissions=permissions)


@router.post("/reveal-sensitive", response_model=RevealSensitiveResponse)
async def reveal_sensitive_identifier(
    request: Request,
    payload: RevealSensitiveRequest,
    current_user: User = Depends(require_permission(Permission.VIEW_SENSITIVE_DATA)),
) -> RevealSensitiveResponse:
    """
    Authorized endpoint to resolve and reveal an unmasked sensitive identifier (Phone / Bank Account).
    Enforces strict server-side RBAC (ADMIN / INVESTIGATOR only).
    Audits the reveal access without recording plain sensitive values.
    """
    clean_id = (payload.entity_id or payload.identifier_value or "").strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID or identifier value is required.")

    # Retrieve real value from Knowledge Graph
    engine = get_graph_engine()
    node = engine.get_node(clean_id)

    unmasked = clean_id
    if node:
        props = node.properties or {}
        if node.label == "Phone" or props.get("phone_number"):
            unmasked = props.get("phone_number") or node.name or clean_id
        elif node.label == "BankAccount" or props.get("account_number"):
            unmasked = props.get("account_number") or node.name or clean_id
        elif node.name:
            unmasked = node.name
    elif clean_id.startswith("PHONE_"):
        unmasked = "+" + clean_id.replace("PHONE_", "").lstrip("+")
    elif clean_id.startswith("ACC_"):
        unmasked = clean_id.replace("ACC_", "")

    # Sanitize identifier for audit log (only record suffix to protect privacy)
    suffix = clean_id[-4:] if len(clean_id) >= 4 else clean_id
    ip = request.client.host if request.client else None

    id_type = payload.entity_type or payload.identifier_type or (node.label if node else "UNKNOWN")

    audit_logger.log(
        action=AuditAction.REVEAL_SENSITIVE_DATA,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="SENSITIVE_IDENTIFIER",
        resource_id=f"ID_ENDING_IN_{suffix}",
        ip_address=ip,
        status="SUCCESS",
        details={
            "entity_type": id_type,
            "target_suffix": suffix,
        },
    )

    return RevealSensitiveResponse(
        entity_id=clean_id,
        unmasked_value=unmasked,
        authorized_by=current_user.username,
        role=current_user.role,
        audit_logged=True,
        authorized=True,
    )

