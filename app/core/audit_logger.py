"""Audit Logging Engine for NODE SENTINEL.

Provides immutable-style audit trail recording for security-sensitive actions,
investigation queries, data exports, user administration, and system events.
Enforces strict sanitization to never log passwords, tokens, or sensitive credentials.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from app.config import settings
from app.models.audit_models import (
    AuditAction,
    AuditLogEntry,
    AuditLogResponse,
    AuditQueryFilter,
)
from app.models.auth_models import Role

logger = logging.getLogger(__name__)

# Keys that MUST be sanitized if found in details/metadata
SENSITIVE_KEYS = {
    "password",
    "password_hash",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "authorization",
    "credential",
    "credentials",
    "api_key",
    "apikey",
    "private_key",
}


def sanitize_dict(data: Any) -> Any:
    """Recursively scrub passwords, tokens, and sensitive keys from log details."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(sens in k_lower for sens in SENSITIVE_KEYS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_dict(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_dict(item) for item in data]
    return data


class AuditLogger:
    """Thread-safe Audit Logger that records operational and security events to disk."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path or settings.AUDIT_FILE)
        self._lock = threading.RLock()
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Ensure parent directories and the initial JSON file exist."""
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                try:
                    with open(self.log_path, "w", encoding="utf-8") as f:
                        json.dump([], f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to initialize audit log at {self.log_path}: {e}")

    def _read_entries_raw(self) -> List[Dict[str, Any]]:
        """Read raw entries from file."""
        if not self.log_path.exists():
            return []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error reading audit log file {self.log_path}: {e}")
            return []

    def _write_entries_raw(self, entries: List[Dict[str, Any]]) -> None:
        """Write raw entries atomically to disk."""
        tmp_path = self.log_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, default=str)
            for attempt in range(5):
                try:
                    tmp_path.replace(self.log_path)
                    break
                except (PermissionError, OSError):
                    if attempt < 4:
                        import time
                        time.sleep(0.02 * (attempt + 1))
                    else:
                        import shutil
                        shutil.copyfile(str(tmp_path), str(self.log_path))
                        tmp_path.unlink(missing_ok=True)
                        break
        except Exception as e:
            logger.error(f"Error writing audit log atomically to {self.log_path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def log(
        self,
        action: AuditAction | str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        role: Optional[Role | str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        """Record an audit event with sanitized details and thread safety."""
        action_val = action.value if isinstance(action, AuditAction) else str(action)
        role_val = role.value if isinstance(role, Role) else (str(role) if role else None)

        clean_details = sanitize_dict(details) if details else None

        entry = AuditLogEntry(
            log_id=f"audit_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action_val,
            user_id=user_id,
            username=username,
            role=role_val,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            details=clean_details,
        )

        with self._lock:
            entries = self._read_entries_raw()
            entries.append(entry.model_dump())
            # Maintain reasonable bounds (e.g. keep latest 10,000 entries)
            if len(entries) > 10000:
                entries = entries[-10000:]
            self._write_entries_raw(entries)

        logger.info(f"AUDIT [{status}] {action_val} by {username or 'anonymous'} on {resource_type or 'system'}:{resource_id or ''}")
        return entry

    def query_logs(self, query: AuditQueryFilter) -> AuditLogResponse:
        """Filter, sort, and paginate audit logs."""
        with self._lock:
            raw_entries = self._read_entries_raw()

        filtered: List[AuditLogEntry] = []
        for r in reversed(raw_entries):  # Newest first
            # Filter action
            if query.action and r.get("action") != query.action:
                continue
            # Filter user_id
            if query.user_id and r.get("user_id") != query.user_id:
                continue
            # Filter username
            if query.username and (r.get("username") or "").lower() != query.username.lower():
                continue
            # Filter role
            if query.role and r.get("role") != query.role:
                continue
            # Filter resource_type
            if query.resource_type and r.get("resource_type") != query.resource_type:
                continue
            # Filter resource_id
            if query.resource_id and r.get("resource_id") != query.resource_id:
                continue
            # Filter status
            if query.status and r.get("status") != query.status:
                continue
            # Filter timestamp range
            ts = r.get("timestamp", "")
            if query.start_time and ts < query.start_time:
                continue
            if query.end_time and ts > query.end_time:
                continue

            filtered.append(AuditLogEntry(**r))

        total = len(filtered)
        start_idx = (query.page - 1) * query.page_size
        end_idx = start_idx + query.page_size
        paged_items = filtered[start_idx:end_idx]

        return AuditLogResponse(
            total=total,
            page=query.page,
            page_size=query.page_size,
            items=paged_items,
        )

    def get_recent_logs(self, limit: int = 50) -> List[AuditLogEntry]:
        """Convenience helper to retrieve most recent logs."""
        query = AuditQueryFilter(page=1, page_size=limit)
        return self.query_logs(query).items


# Singleton instance
audit_logger = AuditLogger()
