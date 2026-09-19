# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Audit Logging Data Models (STEP 15)
Defines audit actions, structured event records, and query filters.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AuditAction(str, Enum):
    # Authentication & Access
    LOGIN = "LOGIN"
    AUTH_LOGIN = "LOGIN"
    LOGIN_FAILED = "LOGIN_FAILED"
    AUTH_LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    AUTH_LOGOUT = "LOGOUT"
    RBAC_ACCESS_DENIED = "RBAC_ACCESS_DENIED"

    # Investigation & Queries
    SEARCH = "SEARCH"
    VIEW_PROFILE = "VIEW_PROFILE"
    VIEW_NETWORK = "VIEW_NETWORK"
    FACE_SEARCH = "FACE_SEARCH"
    CDR_ANALYSIS = "CDR_ANALYSIS"
    FINANCIAL_ANALYSIS = "FINANCIAL_ANALYSIS"
    VIEW_TIMELINE = "VIEW_TIMELINE"
    VIEW_RISK = "VIEW_RISK"
    AI_QUERY = "AI_QUERY"
    GENERATE_REPORT = "GENERATE_REPORT"
    REVEAL_SENSITIVE_DATA = "REVEAL_SENSITIVE_DATA"

    # Data Ingestion
    INGEST_FIR = "INGEST_FIR"
    INGEST_CDR = "INGEST_CDR"
    INGEST_FINANCIAL = "INGEST_FINANCIAL"
    DATA_EXPORT = "DATA_EXPORT"

    # User Administration
    USER_CREATED = "USER_CREATED"
    USER_CREATE = "USER_CREATED"
    USER_UPDATED = "USER_UPDATED"
    USER_UPDATE = "USER_UPDATED"
    USER_ROLE_CHANGED = "USER_ROLE_CHANGED"
    USER_ROLE_CHANGE = "USER_ROLE_CHANGED"
    USER_DISABLED = "USER_DISABLED"
    USER_ENABLED = "USER_ENABLED"
    USER_DELETED = "USER_DELETED"
    USER_DELETE = "USER_DELETED"
    PASSWORD_RESET = "PASSWORD_RESET"

    # System Lifespan
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"


class AuditLogEntry(BaseModel):
    """Structured audit record representing a security or investigative operation."""
    log_id: Optional[str] = None
    event_id: Optional[str] = None
    timestamp: str = Field(..., description="ISO UTC timestamp of action")
    user_id: Optional[str] = Field("ANONYMOUS", description="Acting user identifier or 'ANONYMOUS'")
    username: Optional[str] = Field("anonymous", description="Acting username or 'anonymous'")
    role: Optional[str] = Field("VIEWER", description="Role of the actor at time of action")
    action: str = Field(..., description="Standard audit action verb")
    resource_type: Optional[str] = Field(None, description="Target resource category (e.g. AUTH, GRAPH, CDR, REPORT, USER)")
    resource: Optional[str] = None
    resource_id: Optional[str] = Field(None, description="Specific entity ID, case ID, or record ID accessed")
    status: str = Field("SUCCESS", description="Operation outcome: SUCCESS, FAILED, or DENIED")
    result: Optional[str] = None
    ip_address: Optional[str] = Field(None, description="Client IP address if available")
    user_agent: Optional[str] = Field(None, description="Client user agent if available")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Sanitized metadata (passwords & tokens omitted)")

    def model_post_init(self, __context: Any) -> None:
        if not self.log_id and self.event_id:
            self.log_id = self.event_id
        elif not self.event_id and self.log_id:
            self.event_id = self.log_id

        if not self.resource and self.resource_type:
            self.resource = self.resource_type
        elif not self.resource_type and self.resource:
            self.resource_type = self.resource

        if not self.result and self.status:
            self.result = self.status
        elif not self.status and self.result:
            self.status = self.result


class AuditQueryFilter(BaseModel):
    """Query parameters for filtering audit trail."""
    action: Optional[str] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    user: Optional[str] = None
    role: Optional[str] = None
    resource_type: Optional[str] = None
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    status: Optional[str] = None
    result: Optional[str] = None
    start_time: Optional[str] = None
    date_from: Optional[str] = None
    end_time: Optional[str] = None
    date_to: Optional[str] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=500)
    limit: Optional[int] = None
    offset: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.username and self.user:
            self.username = self.user
        if not self.status and self.result:
            self.status = self.result
        if not self.resource_type and self.resource:
            self.resource_type = self.resource
        if not self.start_time and self.date_from:
            self.start_time = self.date_from
        if not self.end_time and self.date_to:
            self.end_time = self.date_to
        if self.limit and self.page_size == 50:
            self.page_size = self.limit


class AuditLogResponse(BaseModel):
    """Paginated response containing matching audit entries."""
    total: int = 0
    total_records: Optional[int] = None
    page: int = 1
    page_size: int = 50
    limit: Optional[int] = None
    offset: Optional[int] = None
    items: List[AuditLogEntry] = Field(default_factory=list)
    records: Optional[List[AuditLogEntry]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.total_records is None:
            self.total_records = self.total
        elif self.total == 0 and self.total_records:
            self.total = self.total_records

        if self.records is None:
            self.records = self.items
        elif not self.items and self.records:
            self.items = self.records

        if self.limit is None:
            self.limit = self.page_size
        if self.offset is None:
            self.offset = (self.page - 1) * self.page_size

