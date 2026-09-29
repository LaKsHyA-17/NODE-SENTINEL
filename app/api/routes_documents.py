# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Document Management REST API (SIH26190 Phase 2B & Phase 3).

Provides secure endpoints for:
- POST /api/v1/documents/upload
- GET  /api/v1/documents/{document_id}
- GET  /api/v1/documents/{document_id}/download
- GET  /api/v1/documents/case/{case_id}
- POST /api/v1/documents/{document_id}/versions
- GET  /api/v1/documents/{document_id}/versions
- GET  /api/v1/documents/{document_id}/versions/{version_id}
- GET  /api/v1/documents/{document_id}/versions/{version_id}/download
- POST /api/v1/documents/{document_id}/verify-integrity
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.core.audit_logger import audit_logger
from app.core.auth_service import (
    get_current_user,
    enforce_permission,
)
from app.core.document_service import (
    DigitalSignatureService,
    DocumentArchivedError,
    DocumentNotFoundError,
    DocumentSealedError,
    DocumentStorageService,
    FileSizeLimitExceededError,
    IntegrityAnchorService,
    InvalidExtensionError,
    InvalidMimeTypeError,
    PathTraversalError,
    get_document_registry,
    get_document_storage_service,
    process_document_intelligence,
    sync_document_to_knowledge_graph,
)
from app.models.audit_models import AuditAction
from app.models.auth_models import (
    Permission,
    ROLE_PERMISSIONS,
    Role,
    User,
)
from app.models.document_models import (
    ConfidentialityLevel,
    DocumentEntityLink,
    DocumentPermission,
    DocumentRecord,
    DocumentShare,
    DocumentSignatureMetadata,
    DocumentStatus,
    DocumentType,
    DocumentVersion,
    ShareStatus,
    SignatureVerificationStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Document Management"])


# ==============================================================================
# RESPONSE SCHEMAS
# ==============================================================================

class DocumentItemSummary(BaseModel):
    document_id: str
    case_id: Optional[str] = None
    title: str
    doc_type: str
    file_name: str
    file_size: int
    mime_type: str
    current_version: int
    uploaded_by: str
    created_at: str
    updated_at: str
    status: str
    confidentiality_level: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    sha256_hash: str


class VersionItemSummary(BaseModel):
    version_id: str
    document_id: str
    version_number: int
    parent_version_id: Optional[str] = None
    file_size: Optional[int] = None
    sha256_hash: str
    created_by: str
    created_at: str
    change_description: Optional[str] = None
    file_name: Optional[str] = None
    status: str = "SUBMITTED"


class DocumentUploadResponse(BaseModel):
    status: str = "SUCCESS"
    message: str = "Document uploaded and indexed successfully"
    document: DocumentItemSummary
    version: VersionItemSummary


class CaseDocumentListResponse(BaseModel):
    case_id: str
    items: List[DocumentItemSummary] = Field(default_factory=list)
    page: int = 1
    page_size: int = 20
    total: int = 0


class DocumentVersionListResponse(BaseModel):
    document_id: str
    current_version: int
    versions: List[VersionItemSummary] = Field(default_factory=list)
    total_versions: int = 0


class DocumentSearchResponse(BaseModel):
    items: List[DocumentItemSummary] = Field(default_factory=list)
    page: int = 1
    page_size: int = 20
    total: int = 0


class DocumentIntegrityResponse(BaseModel):
    document_id: str
    version_id: Optional[str] = None
    version_number: Optional[int] = None
    expected_sha256: str
    actual_sha256: str
    integrity_valid: bool
    verified_at: str
    status_message: str


class DocumentTextResponse(BaseModel):
    document_id: str
    extracted_text: Optional[str] = None
    character_count: int = 0
    message: str = "OCR and digital text extracted successfully"


class DocumentEntitiesResponse(BaseModel):
    document_id: str
    total_entities: int = 0
    entities: List[Dict[str, Any]] = Field(default_factory=list)


class PermissionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None
    can_read: bool = True
    can_edit: bool = False
    can_download: bool = True
    can_share: bool = False
    can_verify: bool = True


class ShareCreateRequest(BaseModel):
    shared_with: str = Field(..., min_length=1, description="Target User ID or Role")
    permission: str = Field("READ", description="READ, DOWNLOAD, or FULL")
    expires_at: Optional[str] = Field(None, description="Optional ISO expiration timestamp")
    notes: Optional[str] = Field(None, description="Operational justification")


class SignatureCreateRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="Specific version ID to sign; defaults to current version")
    signer_role: Optional[str] = Field(None, description="Investigative designation")
    comments: Optional[str] = Field(None, description="Attestation remarks")


class SignatureVerifyRequest(BaseModel):
    version_id: Optional[str] = None
    signature_value: str
    signer_id: str


class AiSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language investigative query")
    case_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


class AiCitation(BaseModel):
    document_id: str
    title: str
    case_id: Optional[str] = None
    version_number: int
    doc_type: str
    snippet: str
    relevance_score: float


class AiSearchResponse(BaseModel):
    query: str
    summary_answer: str
    citations: List[AiCitation] = Field(default_factory=list)
    total_authorized_documents_searched: int = 0


# ==============================================================================
# AUTHORIZATION & CONFIDENTIALITY HELPERS
# ==============================================================================

def can_access_confidentiality(user: User, level: str | ConfidentialityLevel, document_id: Optional[str] = None) -> bool:
    """
    Check if the user has sufficient clearance for the document confidentiality level.
    - TOP_SECRET / CONFIDENTIAL: requires ADMIN/INVESTIGATOR role or VIEW_SENSITIVE_DATA permission,
      OR an explicit DocumentPermission / DocumentShare grant.
    - RESTRICTED / PUBLIC: accessible to all authenticated active users.
    """
    lvl = level.value if hasattr(level, "value") else str(level).upper()
    if lvl in ("CONFIDENTIAL", "TOP_SECRET"):
        allowed_perms = ROLE_PERMISSIONS.get(user.role, [])
        if Permission.VIEW_SENSITIVE_DATA in allowed_perms or user.role in (Role.ADMIN, Role.INVESTIGATOR):
            return True
        if document_id:
            registry = get_document_registry()
            if registry.has_user_access(document_id, user.user_id, user.role.value):
                return True
        return False
    return True


def format_document_summary(doc_dict: Dict[str, Any]) -> DocumentItemSummary:
    """Convert stored document record into safe public API summary (omits raw server paths)."""
    return DocumentItemSummary(
        document_id=doc_dict["document_id"],
        case_id=doc_dict.get("case_id"),
        title=doc_dict["title"],
        doc_type=doc_dict.get("doc_type", "OTHER"),
        file_name=doc_dict.get("file_name", "document"),
        file_size=doc_dict.get("file_size", 0),
        mime_type=doc_dict.get("mime_type", "application/pdf"),
        current_version=doc_dict.get("current_version", 1),
        uploaded_by=doc_dict.get("uploaded_by", "system"),
        created_at=doc_dict.get("created_at", ""),
        updated_at=doc_dict.get("updated_at", ""),
        status=doc_dict.get("status", "SUBMITTED"),
        confidentiality_level=doc_dict.get("confidentiality_level", "RESTRICTED"),
        description=doc_dict.get("description"),
        tags=doc_dict.get("tags", []),
        sha256_hash=doc_dict.get("sha256_hash", ""),
    )


def format_version_summary(ver_dict: Dict[str, Any]) -> VersionItemSummary:
    """Convert stored version record into safe public API summary."""
    return VersionItemSummary(
        version_id=ver_dict["version_id"],
        document_id=ver_dict["document_id"],
        version_number=ver_dict.get("version_number", 1),
        parent_version_id=ver_dict.get("parent_version_id"),
        file_size=ver_dict.get("file_size"),
        sha256_hash=ver_dict.get("sha256_hash", ""),
        created_by=ver_dict.get("created_by", "system"),
        created_at=ver_dict.get("created_at", ""),
        change_description=ver_dict.get("change_description") or ver_dict.get("change_reason"),
        file_name=ver_dict.get("file_name"),
        status=ver_dict.get("status", "SUBMITTED"),
    )


# ==============================================================================
# 1. DOCUMENT UPLOAD (VERSION 1)
# ==============================================================================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register an investigation/court document",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="Document file to upload (.pdf, .png, .jpg, .jpeg, .webp, .txt)"),
    case_id: str = Form(..., min_length=1, description="Associated Legal/FIR Case ID"),
    title: str = Form(..., min_length=1, description="Official title or description of filing"),
    document_type: str = Form("OTHER", description="Classification (FIR, WITNESS_STATEMENT, CHARGE_SHEET, etc.)"),
    confidentiality: str = Form("RESTRICTED", description="PUBLIC, RESTRICTED, CONFIDENTIAL, TOP_SECRET"),
    description: Optional[str] = Form(None, description="Executive remarks or notes"),
    tags: Optional[str] = Form(None, description="Comma-separated categorization tags"),
    current_user: User = Depends(get_current_user),
) -> DocumentUploadResponse:
    """
    Securely uploads, validates, hashes, and indexes a legal document into the registry.
    Creates DocumentRecord and immutable DocumentVersion v1.
    """
    allowed_perms = ROLE_PERMISSIONS.get(current_user.role, [])
    if Permission.INGEST_DATA not in allowed_perms and current_user.role not in (Role.ADMIN, Role.INVESTIGATOR):
        ip = request.client.host if request.client else None
        audit_logger.log(
            action=AuditAction.DOCUMENT_ACCESS_DENIED,
            user_id=current_user.user_id,
            username=current_user.username,
            role=current_user.role,
            resource_type="DOCUMENT_UPLOAD",
            ip_address=ip,
            status="DENIED",
            details={"reason": "User lacks INGEST_DATA permission"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: role '{current_user.role.value}' cannot upload documents",
        )

    try:
        doc_type_enum = DocumentType(document_type.upper().strip())
    except ValueError:
        doc_type_enum = DocumentType.OTHER

    try:
        conf_enum = ConfidentialityLevel(confidentiality.upper().strip())
    except ValueError:
        conf_enum = ConfidentialityLevel.RESTRICTED

    try:
        content_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file payload: {e}",
        )

    if not content_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file payload is empty (0 bytes).",
        )

    storage_svc = get_document_storage_service()
    try:
        save_meta = storage_svc.save_document(
            file_bytes=content_bytes,
            original_filename=file.filename or "document.pdf",
            client_mime=file.content_type,
            subdirectory=case_id.replace("/", "_").replace("\\", "_")[:32],
        )
    except FileSizeLimitExceededError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except (InvalidExtensionError, InvalidMimeTypeError) as e:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e))
    except PathTraversalError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Storage error during document upload: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store document file.")

    doc_id = save_meta["document_id"]
    version_id = f"VER-{doc_id}-v1"
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    version_dict = {
        "version_id": version_id,
        "document_id": doc_id,
        "version_number": 1,
        "parent_version_id": None,
        "file_path": save_meta["storage_reference"],
        "sha256_hash": save_meta["sha256_hash"],
        "created_by": current_user.user_id,
        "created_at": save_meta["created_at"],
        "change_reason": "Initial registration",
        "change_description": "Initial registration",
        "file_name": save_meta["original_filename"],
        "file_size": save_meta["file_size"],
        "status": "SUBMITTED",
        "metadata": {
            "mime_type": save_meta["mime_type"],
        },
    }

    record_model = DocumentRecord(
        document_id=doc_id,
        case_id=case_id.strip(),
        title=title.strip(),
        doc_type=doc_type_enum,
        file_name=save_meta["original_filename"],
        file_size=save_meta["file_size"],
        mime_type=save_meta["mime_type"],
        file_path=save_meta["storage_reference"],
        sha256_hash=save_meta["sha256_hash"],
        current_version=1,
        uploaded_by=current_user.user_id,
        status=DocumentStatus.SUBMITTED,
        confidentiality_level=conf_enum,
        description=description.strip() if description else None,
        tags=parsed_tags,
    )

    registry = get_document_registry()
    registry.save_document(record_model.model_dump(), version_dict)

    # Run Document Intelligence / OCR asynchronously or synchronously
    try:
        await process_document_intelligence(
            document_id=doc_id,
            file_bytes=content_bytes,
            filename=save_meta["original_filename"],
            case_id=case_id.strip(),
            doc_type=doc_type_enum.value,
            sync_to_graph=True,
        )
    except Exception as ocr_err:
        logger.warning(f"OCR background processing error for doc '{doc_id}': {ocr_err}")

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_UPLOAD,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT",
        resource_id=doc_id,
        ip_address=ip,
        status="SUCCESS",
        details={
            "case_id": case_id,
            "title": title,
            "doc_type": doc_type_enum.value,
            "file_name": save_meta["original_filename"],
            "file_size": save_meta["file_size"],
            "sha256": save_meta["sha256_hash"],
            "confidentiality": conf_enum.value,
            "version": 1,
        },
    )

    return DocumentUploadResponse(
        status="SUCCESS",
        message="Document uploaded, hashed, and indexed into legal repository.",
        document=format_document_summary(record_model.model_dump()),
        version=format_version_summary(version_dict),
    )


# ==============================================================================
# 2. DOCUMENT SEARCH (METADATA SEARCH API)
# ==============================================================================

@router.get(
    "/search",
    response_model=DocumentSearchResponse,
    summary="Search and filter legal documents by metadata attributes",
)
def search_documents(
    request: Request,
    q: Optional[str] = Query(None, max_length=100, description="General keyword search across title, description, case ID, tags, and document ID"),
    document_id: Optional[str] = Query(None, description="Exact document ID filter"),
    case_id: Optional[str] = Query(None, description="Associated FIR/Case ID filter"),
    document_type: Optional[str] = Query(None, description="Document type classification"),
    status_filter: Optional[str] = Query(None, alias="status", description="Lifecycle status (SUBMITTED, SEALED, ARCHIVED, etc.)"),
    confidentiality: Optional[str] = Query(None, description="Confidentiality level filter (PUBLIC, RESTRICTED, CONFIDENTIAL, TOP_SECRET)"),
    created_by: Optional[str] = Query(None, description="Uploader badge / user ID filter"),
    version_number: Optional[int] = Query(None, ge=1, description="Match documents having specified version number"),
    created_from: Optional[str] = Query(None, description="Start date filter in ISO-8601 format"),
    created_to: Optional[str] = Query(None, description="End date filter in ISO-8601 format"),
    sort_by: Optional[str] = Query("created_at", description="Sort attribute (created_at, updated_at, title, doc_type, status, file_size)"),
    sort_order: Optional[str] = Query("desc", description="Sort direction ('asc' or 'desc')"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    current_user: User = Depends(get_current_user),
) -> DocumentSearchResponse:
    """
    Search and filter registered documents using stored metadata attributes.
    Enforces authorization clearance and confidentiality filtering prior to result collation.
    """
    # 1. Validate enum filters if provided
    clean_doc_type: Optional[str] = None
    if document_type:
        try:
            clean_doc_type = DocumentType(document_type.strip().upper()).value
        except ValueError:
            allowed_types = ", ".join([e.value for e in DocumentType])
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document_type '{document_type}'. Allowed values: {allowed_types}",
            )

    clean_status: Optional[str] = None
    if status_filter:
        try:
            clean_status = DocumentStatus(status_filter.strip().upper()).value
        except ValueError:
            allowed_statuses = ", ".join([e.value for e in DocumentStatus])
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{status_filter}'. Allowed values: {allowed_statuses}",
            )

    clean_conf: Optional[str] = None
    if confidentiality:
        try:
            clean_conf = ConfidentialityLevel(confidentiality.strip().upper()).value
        except ValueError:
            allowed_confs = ", ".join([e.value for e in ConfidentialityLevel])
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid confidentiality '{confidentiality}'. Allowed values: {allowed_confs}",
            )

    # 2. Validate sort fields
    allowed_sort_fields = {"created_at", "updated_at", "title", "doc_type", "status", "file_size"}
    selected_sort_by = (sort_by or "created_at").strip().lower()
    if selected_sort_by not in allowed_sort_fields:
        selected_sort_by = "created_at"

    selected_sort_order = (sort_order or "desc").strip().lower()
    if selected_sort_order not in ("asc", "desc"):
        selected_sort_order = "desc"

    # 3. Search registry with authorization filter
    registry = get_document_registry()
    try:
        raw_items, total_count = registry.search_documents(
            query=q,
            document_id=document_id,
            case_id=case_id,
            doc_type=clean_doc_type,
            status=clean_status,
            confidentiality=clean_conf,
            created_by=created_by,
            version_number=version_number,
            created_from=created_from,
            created_to=created_to,
            sort_by=selected_sort_by,
            sort_order=selected_sort_order,
            page=page,
            page_size=page_size,
            can_access_fn=lambda d: can_access_confidentiality(current_user, d.get("confidentiality_level", "RESTRICTED"), d.get("document_id")),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 4. Format public response
    formatted_items = [format_document_summary(item) for item in raw_items]

    # 5. Audit search action
    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_SEARCH,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_SEARCH",
        ip_address=ip,
        status="SUCCESS",
        details={
            "q": q,
            "document_id": document_id,
            "case_id": case_id,
            "document_type": clean_doc_type,
            "status": clean_status,
            "confidentiality": clean_conf,
            "created_by": created_by,
            "version_number": version_number,
            "total_results": total_count,
            "page": page,
            "page_size": page_size,
        },
    )

    return DocumentSearchResponse(
        items=formatted_items,
        page=page,
        page_size=page_size,
        total=total_count,
    )


# ==============================================================================
# 3. DOCUMENT DETAILS
# ==============================================================================

@router.get(
    "/{document_id}",
    response_model=DocumentItemSummary,
    summary="Retrieve document metadata and current version info",
)
def get_document_details(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> DocumentItemSummary:
    """
    Retrieves metadata for a specific document.
    Enforces role-based clearance and confidentiality level restrictions.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in registry.",
        )

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        ip = request.client.host if request.client else None
        audit_logger.log(
            action=AuditAction.DOCUMENT_ACCESS_DENIED,
            user_id=current_user.user_id,
            username=current_user.username,
            role=current_user.role,
            resource_type="DOCUMENT",
            resource_id=document_id,
            ip_address=ip,
            status="DENIED",
            details={"confidentiality": doc.get("confidentiality_level")},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient clearance for this confidential document",
        )

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_VIEW,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT",
        resource_id=document_id,
        ip_address=ip,
        status="SUCCESS",
        details={"case_id": doc.get("case_id"), "doc_type": doc.get("doc_type")},
    )

    return format_document_summary(doc)


# ==============================================================================
# 3. DOCUMENT DOWNLOAD (CURRENT VERSION)
# ==============================================================================

@router.get(
    "/{document_id}/download",
    summary="Download physical document binary file (current version)",
)
def download_document(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    """
    Securely streams the current version binary document file.
    Validates user authorization, storage isolation, and prevents path traversal.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in registry.",
        )

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        ip = request.client.host if request.client else None
        audit_logger.log(
            action=AuditAction.DOCUMENT_ACCESS_DENIED,
            user_id=current_user.user_id,
            username=current_user.username,
            role=current_user.role,
            resource_type="DOCUMENT_DOWNLOAD",
            resource_id=document_id,
            ip_address=ip,
            status="DENIED",
            details={"confidentiality": doc.get("confidentiality_level")},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient clearance to download this confidential document",
        )

    storage_svc = get_document_storage_service()
    storage_ref = doc.get("file_path")
    if not storage_ref:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Physical storage reference missing for this document.",
        )

    try:
        file_path = storage_svc.get_document_path(storage_ref)
    except (DocumentNotFoundError, PathTraversalError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Physical document file missing or inaccessible on server storage.",
        )

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_DOWNLOAD,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT",
        resource_id=document_id,
        ip_address=ip,
        status="SUCCESS",
        details={
            "case_id": doc.get("case_id"),
            "file_name": doc.get("file_name"),
            "sha256": doc.get("sha256_hash"),
            "version": doc.get("current_version", 1),
        },
    )

    return FileResponse(
        path=str(file_path),
        filename=doc.get("file_name", "document.pdf"),
        media_type=doc.get("mime_type", "application/pdf"),
    )


# ==============================================================================
# 4. CASE DOCUMENT LISTING
# ==============================================================================

@router.get(
    "/case/{case_id}",
    response_model=CaseDocumentListResponse,
    summary="List all authorized documents registered under a Case/FIR",
)
def list_documents_by_case(
    case_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
) -> CaseDocumentListResponse:
    """
    Returns paginated list of documents associated with a specific legal or FIR case.
    Filters out documents exceeding user's confidentiality clearance.
    """
    registry = get_document_registry()
    raw_docs, total_raw = registry.list_documents(case_id=case_id, page=page, page_size=page_size)

    filtered_items: List[DocumentItemSummary] = []
    for d in raw_docs:
        if can_access_confidentiality(current_user, d.get("confidentiality_level", "RESTRICTED"), d.get("document_id")):
            filtered_items.append(format_document_summary(d))

    return CaseDocumentListResponse(
        case_id=case_id,
        items=filtered_items,
        page=page,
        page_size=page_size,
        total=total_raw,
    )


# ==============================================================================
# 5. CREATE NEW IMMUTABLE VERSION (PHASE 3)
# ==============================================================================

@router.post(
    "/{document_id}/versions",
    response_model=VersionItemSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register a new immutable version of an existing document",
)
async def create_document_version(
    document_id: str,
    request: Request,
    file: UploadFile = File(..., description="Updated version file (.pdf, .png, .jpg, .jpeg, .webp, .txt)"),
    change_description: Optional[str] = Form(None, description="Detailed explanation of legal/investigation amendments"),
    current_user: User = Depends(get_current_user),
) -> VersionItemSummary:
    """
    Creates a new immutable version for an existing document.
    Never overwrites previous versions or their SHA-256 integrity hashes.
    Monotonically increments version number (e.g. v2, v3).
    """
    allowed_perms = ROLE_PERMISSIONS.get(current_user.role, [])
    if Permission.INGEST_DATA not in allowed_perms and current_user.role not in (Role.ADMIN, Role.INVESTIGATOR):
        ip = request.client.host if request.client else None
        audit_logger.log(
            action=AuditAction.DOCUMENT_ACCESS_DENIED,
            user_id=current_user.user_id,
            username=current_user.username,
            role=current_user.role,
            resource_type="DOCUMENT_VERSION_CREATE",
            resource_id=document_id,
            ip_address=ip,
            status="DENIED",
            details={"reason": "User lacks permission to create document versions"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: role '{current_user.role.value}' cannot create document versions",
        )

    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in registry.",
        )

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient clearance for this document",
        )

    try:
        content_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file payload: {e}",
        )

    if not content_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded version file payload is empty (0 bytes).",
        )

    storage_svc = get_document_storage_service()
    try:
        updated_doc, new_version = registry.create_next_version(
            document_id=document_id,
            file_bytes=content_bytes,
            original_filename=file.filename or "document.pdf",
            created_by=current_user.user_id,
            change_description=change_description,
            client_mime=file.content_type,
            storage_svc=storage_svc,
        )
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (DocumentSealedError, DocumentArchivedError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileSizeLimitExceededError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except (InvalidExtensionError, InvalidMimeTypeError) as e:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e))
    except PathTraversalError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating version for document '{document_id}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create document version.")

    try:
        await process_document_intelligence(
            document_id=document_id,
            file_bytes=content_bytes,
            filename=file.filename or "document.pdf",
            case_id=updated_doc.get("case_id"),
            doc_type=updated_doc.get("doc_type"),
            sync_to_graph=True,
        )
    except Exception as ocr_err:
        logger.warning(f"OCR background processing error for doc '{document_id}' v{new_version.get('version_number')}: {ocr_err}")

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_VERSION_CREATED,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_VERSION",
        resource_id=new_version["version_id"],
        ip_address=ip,
        status="SUCCESS",
        details={
            "document_id": document_id,
            "version_number": new_version["version_number"],
            "parent_version_id": new_version.get("parent_version_id"),
            "sha256": new_version["sha256_hash"],
            "change_description": change_description,
        },
    )

    return format_version_summary(new_version)


# ==============================================================================
# 6. VERSION HISTORY & INDIVIDUAL VERSION METADATA (PHASE 3)
# ==============================================================================

@router.get(
    "/{document_id}/versions",
    response_model=DocumentVersionListResponse,
    summary="Retrieve complete version history for a document",
)
def list_document_versions(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> DocumentVersionListResponse:
    """
    Returns complete chronological version history for a document.
    Excludes internal filesystem storage paths.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in registry.",
        )

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient clearance for this confidential document",
        )

    versions = registry.get_versions(document_id)
    summaries = [format_version_summary(v) for v in versions]

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_VIEW,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_VERSIONS",
        resource_id=document_id,
        ip_address=ip,
        status="SUCCESS",
        details={"version_count": len(summaries)},
    )

    return DocumentVersionListResponse(
        document_id=document_id,
        current_version=doc.get("current_version", 1),
        versions=summaries,
        total_versions=len(summaries),
    )


@router.get(
    "/{document_id}/versions/{version_id}",
    response_model=VersionItemSummary,
    summary="Retrieve metadata for a specific document version",
)
def get_version_details(
    document_id: str,
    version_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> VersionItemSummary:
    """
    Retrieves metadata for an explicit version.
    Guards against IDOR by verifying version belongs to the specified document.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient clearance for this confidential document",
        )

    ver = registry.get_version(document_id, version_id)
    if not ver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version '{version_id}' not found for document '{document_id}'.",
        )

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_VIEW,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_VERSION",
        resource_id=version_id,
        ip_address=ip,
        status="SUCCESS",
        details={"document_id": document_id, "version_number": ver.get("version_number")},
    )

    return format_version_summary(ver)


# ==============================================================================
# 7. VERSION DOWNLOAD (PHASE 3)
# ==============================================================================

@router.get(
    "/{document_id}/versions/{version_id}/download",
    summary="Download specific document version binary file",
)
def download_document_version(
    document_id: str,
    version_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    """
    Securely streams the binary file for an explicit historical version.
    Validates document/version relationship, clearance, and storage isolation.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient clearance to download this confidential document",
        )

    ver = registry.get_version(document_id, version_id)
    if not ver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version '{version_id}' not found for document '{document_id}'.",
        )

    storage_svc = get_document_storage_service()
    storage_ref = ver.get("file_path")
    if not storage_ref:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Physical storage reference missing for this version.",
        )

    try:
        file_path = storage_svc.get_document_path(storage_ref)
    except (DocumentNotFoundError, PathTraversalError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Physical version file missing or inaccessible on server storage.",
        )

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_DOWNLOAD,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_VERSION",
        resource_id=version_id,
        ip_address=ip,
        status="SUCCESS",
        details={
            "document_id": document_id,
            "version_number": ver.get("version_number"),
            "sha256": ver.get("sha256_hash"),
        },
    )

    filename = ver.get("file_name") or doc.get("file_name", "document.pdf")
    mime_type = (ver.get("metadata") or {}).get("mime_type") or doc.get("mime_type", "application/pdf")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=mime_type,
    )


# ==============================================================================
# 8. INTEGRITY VERIFICATION (PHASE 3)
# ==============================================================================

@router.post(
    "/{document_id}/verify-integrity",
    response_model=DocumentIntegrityResponse,
    summary="Cryptographically verify SHA-256 integrity of stored document",
)
def verify_document_integrity(
    document_id: str,
    request: Request,
    version_id: Optional[str] = Query(None, description="Optional specific version ID to verify; defaults to current version"),
    current_user: User = Depends(get_current_user),
) -> DocumentIntegrityResponse:
    """
    Reads physical file from disk, recalculates SHA-256 digest, and constant-time
    compares against stored integrity record. Detects tampering and bit-rot.
    Logs DOCUMENT_INTEGRITY_VERIFIED or DOCUMENT_INTEGRITY_FAILED.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient clearance for this confidential document",
        )

    storage_svc = get_document_storage_service()

    if version_id:
        target_ver = registry.get_version(document_id, version_id)
        if not target_ver:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version '{version_id}' not found for document '{document_id}'.",
            )
        storage_ref = target_ver.get("file_path")
        expected_hash = target_ver.get("sha256_hash", "")
        v_num = target_ver.get("version_number", 1)
        v_id = target_ver.get("version_id")
    else:
        storage_ref = doc.get("file_path")
        expected_hash = doc.get("sha256_hash", "")
        v_num = doc.get("current_version", 1)
        v_id = f"VER-{document_id}-v{v_num}"

    if not storage_ref or not expected_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Integrity metadata or storage reference missing for document.",
        )

    try:
        is_valid, actual_hash = storage_svc.verify_document_integrity(storage_ref, expected_hash)
    except (DocumentNotFoundError, PathTraversalError):
        is_valid = False
        actual_hash = "FILE_MISSING_ON_DISK"

    now_iso = datetime.now(timezone.utc).isoformat()
    ip = request.client.host if request.client else None

    if is_valid:
        audit_logger.log(
            action=AuditAction.DOCUMENT_INTEGRITY_VERIFIED,
            user_id=current_user.user_id,
            username=current_user.username,
            role=current_user.role,
            resource_type="DOCUMENT_INTEGRITY",
            resource_id=document_id,
            ip_address=ip,
            status="SUCCESS",
            details={
                "version_id": v_id,
                "version_number": v_num,
                "sha256": actual_hash,
                "integrity_valid": True,
            },
        )
        status_msg = "Cryptographic integrity verified. File matches immutable recorded SHA-256 digest."
    else:
        audit_logger.log(
            action=AuditAction.DOCUMENT_INTEGRITY_FAILED,
            user_id=current_user.user_id,
            username=current_user.username,
            role=current_user.role,
            resource_type="DOCUMENT_INTEGRITY",
            resource_id=document_id,
            ip_address=ip,
            status="FAILED",
            details={
                "version_id": v_id,
                "version_number": v_num,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "integrity_valid": False,
            },
        )
        status_msg = "SECURITY ALERT: Integrity violation detected! File contents do not match recorded SHA-256 digest."

    return DocumentIntegrityResponse(
        document_id=document_id,
        version_id=v_id,
        version_number=v_num,
        expected_sha256=expected_hash,
        actual_sha256=actual_hash,
        integrity_valid=is_valid,
        verified_at=now_iso,
        status_message=status_msg,
    )


# ==============================================================================
# 9. OCR & EXTRACTED INTELLIGENCE (PHASE 6)
# ==============================================================================

@router.get(
    "/{document_id}/text",
    response_model=DocumentTextResponse,
    summary="Retrieve extracted/OCR textual content for authorized document",
)
def get_document_text(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> DocumentTextResponse:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: insufficient clearance for this confidential document")

    text = registry.get_extracted_text(document_id) or doc.get("extracted_text") or ""
    return DocumentTextResponse(
        document_id=document_id,
        extracted_text=text,
        character_count=len(text),
        message="OCR text retrieved successfully" if text else "No text extracted for this document yet.",
    )


@router.get(
    "/{document_id}/entities",
    response_model=DocumentEntitiesResponse,
    summary="Retrieve extracted structured entities & provenance links",
)
def get_document_entities(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> DocumentEntitiesResponse:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: insufficient clearance for this confidential document")

    entities = registry.get_document_entities(document_id)
    return DocumentEntitiesResponse(
        document_id=document_id,
        total_entities=len(entities),
        entities=entities,
    )


@router.post(
    "/{document_id}/reprocess-ocr",
    summary="Re-execute OCR and entity extraction pipeline on stored document",
)
async def reprocess_document_ocr(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    allowed_perms = ROLE_PERMISSIONS.get(current_user.role, [])
    if Permission.INGEST_DATA not in allowed_perms and current_user.role not in (Role.ADMIN, Role.INVESTIGATOR):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permission to reprocess OCR.")

    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    storage_svc = get_document_storage_service()
    try:
        fpath = storage_svc.get_document_path(doc["file_path"])
        with open(fpath, "rb") as f:
            content_bytes = f.read()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Physical file missing on server: {e}")

    extracted_text, entity_links, triplets = await process_document_intelligence(
        document_id=document_id,
        file_bytes=content_bytes,
        filename=doc.get("file_name", "document.pdf"),
        case_id=doc.get("case_id"),
        doc_type=doc.get("doc_type"),
        sync_to_graph=True,
    )

    return {
        "status": "SUCCESS",
        "document_id": document_id,
        "extracted_characters": len(extracted_text),
        "entities_extracted": len(entity_links),
        "triplets_extracted": len(triplets),
    }


# ==============================================================================
# 10. KNOWLEDGE GRAPH INTEGRATION (PHASE 7)
# ==============================================================================

@router.post(
    "/{document_id}/sync-graph",
    summary="Synchronize extracted document entities to the Knowledge Graph",
)
def sync_document_graph(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    entities = registry.get_document_entities(document_id)
    sync_document_to_knowledge_graph(document_id, entities, [], doc.get("case_id"))
    return {
        "status": "SUCCESS",
        "document_id": document_id,
        "synced_entities": len(entities),
    }


@router.get(
    "/{document_id}/graph-lineage",
    summary="Retrieve Knowledge Graph nodes & edges derived from this document",
)
def get_document_graph_lineage(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    from app.core.graph_engine import get_graph_engine
    engine = get_graph_engine()

    matched_nodes = []
    for node in engine.get_all_nodes():
        if node.properties.get("source_document_id") == document_id:
            matched_nodes.append({
                "id": node.id,
                "label": node.label.value if hasattr(node.label, "value") else str(node.label),
                "name": node.name,
                "properties": node.properties,
            })

    matched_edges = []
    for edge in engine.get_all_edges():
        if edge.properties.get("source_document_id") == document_id:
            matched_edges.append({
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "relationship": edge.relationship.value if hasattr(edge.relationship, "value") else str(edge.relationship),
                "properties": edge.properties,
            })

    return {
        "document_id": document_id,
        "nodes": matched_nodes,
        "edges": matched_edges,
        "total_nodes": len(matched_nodes),
        "total_edges": len(matched_edges),
    }


# ==============================================================================
# 11. DOCUMENT RBAC & SHARING (PHASE 8)
# ==============================================================================

@router.post(
    "/{document_id}/permissions",
    status_code=status.HTTP_201_CREATED,
    summary="Grant or update document-specific RBAC permission rule",
)
def create_document_permission(
    document_id: str,
    body: PermissionCreateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if current_user.role not in (Role.ADMIN, Role.INVESTIGATOR) and doc.get("uploaded_by") != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins or document owners can grant permissions.")

    import uuid
    perm_id = f"PERM-{uuid.uuid4().hex[:12].upper()}"
    perm_obj = DocumentPermission(
        permission_id=perm_id,
        document_id=document_id,
        user_id=body.user_id,
        role=body.role,
        can_read=body.can_read,
        can_edit=body.can_edit,
        can_download=body.can_download,
        can_share=body.can_share,
        can_verify=body.can_verify,
    )

    saved = registry.add_permission(perm_obj.model_dump())

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_PERMISSION_CHANGED,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_PERMISSION",
        resource_id=perm_id,
        ip_address=ip,
        status="SUCCESS",
        details={"document_id": document_id, "target_user": body.user_id, "target_role": body.role},
    )

    return saved


@router.get(
    "/{document_id}/permissions",
    summary="List document permissions",
)
def list_document_permissions(
    document_id: str,
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return registry.get_permissions(document_id)


@router.delete(
    "/{document_id}/permissions/{permission_id}",
    summary="Revoke document permission",
)
def delete_document_permission(
    document_id: str,
    permission_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if current_user.role not in (Role.ADMIN, Role.INVESTIGATOR) and doc.get("uploaded_by") != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins or owners can revoke permissions.")

    deleted = registry.delete_permission(document_id, permission_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found.")

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_PERMISSION_CHANGED,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_PERMISSION",
        resource_id=permission_id,
        ip_address=ip,
        status="SUCCESS",
        details={"document_id": document_id, "action": "REVOKED"},
    )
    return {"status": "SUCCESS", "message": f"Permission '{permission_id}' revoked."}


@router.post(
    "/{document_id}/shares",
    status_code=status.HTTP_201_CREATED,
    summary="Share document with external user, badge ID, or role",
)
def share_document(
    document_id: str,
    body: ShareCreateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if current_user.role not in (Role.ADMIN, Role.INVESTIGATOR) and doc.get("uploaded_by") != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins, investigators, or owners can share documents.")

    import uuid
    share_id = f"SHARE-{uuid.uuid4().hex[:12].upper()}"
    share_obj = DocumentShare(
        share_id=share_id,
        document_id=document_id,
        shared_by=current_user.user_id,
        shared_with=body.shared_with.strip(),
        permission=body.permission.upper().strip(),
        expires_at=body.expires_at,
        status=ShareStatus.ACTIVE,
        notes=body.notes,
    )

    saved = registry.add_share(share_obj.model_dump())

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_SHARED,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_SHARE",
        resource_id=share_id,
        ip_address=ip,
        status="SUCCESS",
        details={"document_id": document_id, "shared_with": body.shared_with, "permission": body.permission},
    )

    return saved


@router.get(
    "/{document_id}/shares",
    summary="List active and historical shares for a document",
)
def list_document_shares(
    document_id: str,
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return registry.get_shares(document_id)


@router.delete(
    "/{document_id}/shares/{share_id}",
    summary="Revoke an active document share",
)
def revoke_document_share(
    document_id: str,
    share_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if current_user.role not in (Role.ADMIN, Role.INVESTIGATOR) and doc.get("uploaded_by") != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins or owners can revoke shares.")

    revoked = registry.revoke_share(document_id, share_id, current_user.user_id)
    if not revoked:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Share '{share_id}' not found.")

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_SHARE_REVOKED,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_SHARE",
        resource_id=share_id,
        ip_address=ip,
        status="SUCCESS",
        details={"document_id": document_id, "share_id": share_id},
    )

    return {"status": "SUCCESS", "message": f"Share '{share_id}' revoked.", "share": revoked}


# ==============================================================================
# 12. DOCUMENT AUDIT TRAIL (PHASE 9)
# ==============================================================================

@router.get(
    "/{document_id}/audit",
    summary="Retrieve immutable audit log history for this specific document",
)
def get_document_audit_trail(
    document_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    all_logs = audit_logger.get_recent_logs(limit=500)
    matched = []
    for l in all_logs:
        l_dict = l.model_dump() if hasattr(l, "model_dump") else dict(l)
        r_id = l_dict.get("resource_id") or ""
        details = l_dict.get("details") or {}
        if r_id == document_id or details.get("document_id") == document_id or document_id in str(details):
            matched.append(l_dict)

    return matched


# ==============================================================================
# 13. DIGITAL SIGNATURES (PHASE 10)
# ==============================================================================

@router.post(
    "/{document_id}/sign",
    summary="Digitally sign document version SHA-256 digest for evidentiary non-repudiation",
)
def sign_document_version(
    document_id: str,
    body: SignatureCreateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    allowed_perms = ROLE_PERMISSIONS.get(current_user.role, [])
    if Permission.INGEST_DATA not in allowed_perms and current_user.role not in (Role.ADMIN, Role.INVESTIGATOR):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Signatures require INVESTIGATOR or ADMIN credentials.")

    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if body.version_id:
        ver = registry.get_version(document_id, body.version_id)
        if not ver:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Specified version not found.")
        target_hash = ver.get("sha256_hash", "")
        v_id = body.version_id
    else:
        target_hash = doc.get("sha256_hash", "")
        v_num = doc.get("current_version", 1)
        v_id = f"VER-{document_id}-v{v_num}"

    sig_value = DigitalSignatureService.sign_version_hash(target_hash, current_user.user_id)
    sig_id = f"SIG-{uuid.uuid4().hex[:12].upper()}"

    sig_metadata = DocumentSignatureMetadata(
        signature_id=sig_id,
        document_id=document_id,
        version_id=v_id,
        signer_id=current_user.user_id,
        algorithm="HMAC-SHA256",
        signature_value=sig_value,
        verification_status=SignatureVerificationStatus.VALID,
        signer_role=body.signer_role or current_user.role.value,
        comments=body.comments or "Evidentiary digital seal applied.",
    )

    saved = registry.add_signature(sig_metadata.model_dump())

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_SIGNED,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_SIGNATURE",
        resource_id=sig_id,
        ip_address=ip,
        status="SUCCESS",
        details={"document_id": document_id, "version_id": v_id, "signature": sig_value},
    )

    return saved


@router.post(
    "/{document_id}/verify-signature",
    summary="Cryptographically verify digital signature against document version digest",
)
def verify_document_signature(
    document_id: str,
    body: SignatureVerifyRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if body.version_id:
        ver = registry.get_version(document_id, body.version_id)
        target_hash = ver.get("sha256_hash", "") if ver else ""
    else:
        target_hash = doc.get("sha256_hash", "")

    if not target_hash:
        return {"valid": False, "status": "INVALID", "message": "No version hash found."}

    is_valid = DigitalSignatureService.verify_version_signature(target_hash, body.signer_id, body.signature_value)

    return {
        "valid": is_valid,
        "status": "VALID" if is_valid else "INVALID",
        "document_id": document_id,
        "signer_id": body.signer_id,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "message": "Signature verified with non-repudiation." if is_valid else "Signature does not match.",
    }


@router.get(
    "/{document_id}/signatures",
    summary="List all digital signatures for a document",
)
def list_document_signatures(
    document_id: str,
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return registry.get_signatures(document_id)


# ==============================================================================
# 14. AI RETRIEVAL / RAG SEARCH (PHASE 11)
# ==============================================================================

@router.post(
    "/ai/search",
    response_model=AiSearchResponse,
    summary="Natural language AI-assisted document retrieval (strictly authorization pre-filtered)",
)
def ai_document_search(
    body: AiSearchRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> AiSearchResponse:
    registry = get_document_registry()
    all_docs, _ = registry.search_documents(
        page=1,
        page_size=100,
        can_access_fn=lambda d: can_access_confidentiality(current_user, d.get("confidentiality_level", "RESTRICTED"), d.get("document_id")),
    )

    if body.case_id:
        all_docs = [d for d in all_docs if (d.get("case_id") or "").lower() == body.case_id.lower()]

    terms = [t.lower() for t in body.query.split() if len(t) >= 2]
    citations: List[AiCitation] = []

    for d in all_docs:
        text = registry.get_extracted_text(d["document_id"]) or d.get("extracted_text") or ""
        title = d.get("title", "")
        desc = d.get("description") or ""
        combined = f"{title} {desc} {text}".lower()

        score = 0.0
        for t in terms:
            if t in combined:
                score += 1.0

        if score > 0 or not terms:
            # Generate best relevant citation snippet based on matching term density
            snippet = ""
            if text:
                sentences = [s.strip() for s in text.replace("\r", "\n").split("\n") if s.strip()]
                best_sent = ""
                best_match_count = -1
                for sent in sentences:
                    m_count = sum(1 for t in terms if t in sent.lower())
                    if m_count > best_match_count:
                        best_match_count = m_count
                        best_sent = sent
                snippet = best_sent if best_match_count > 0 else (sentences[0] if sentences else "")
            if not snippet:
                snippet = desc or title

            citations.append(AiCitation(
                document_id=d["document_id"],
                title=title,
                case_id=d.get("case_id"),
                version_number=d.get("current_version", 1),
                doc_type=d.get("doc_type", "OTHER"),
                snippet=snippet[:250],
                relevance_score=round(score / (len(terms) or 1), 2),
            ))

    citations.sort(key=lambda c: c.relevance_score, reverse=True)
    selected_citations = citations[:body.top_k]

    if selected_citations:
        summary = f"Found {len(selected_citations)} authorized document(s) matching '{body.query}'. Top finding from '{selected_citations[0].title}' ({selected_citations[0].document_id})."
    else:
        summary = f"No authorized records or evidence found matching '{body.query}'."

    ip = request.client.host if request.client else None
    audit_logger.log(
        action=AuditAction.AI_QUERY,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="AI_RETRIEVAL",
        ip_address=ip,
        status="SUCCESS",
        details={"query": body.query, "results_returned": len(selected_citations)},
    )

    return AiSearchResponse(
        query=body.query,
        summary_answer=summary,
        citations=selected_citations,
        total_authorized_documents_searched=len(all_docs),
    )


# ==============================================================================
# 15. INTEGRITY ANCHOR / BLOCKCHAIN (PHASE 12)
# ==============================================================================

@router.post(
    "/{document_id}/anchor",
    summary="Anchor document cryptographic commitment to immutable integrity ledger",
)
def anchor_document_integrity(
    document_id: str,
    version_id: Optional[str] = Query(None, description="Optional version ID to anchor; defaults to current"),
    request: Request = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if version_id:
        ver = registry.get_version(document_id, version_id)
        target_hash = ver.get("sha256_hash") if ver else ""
        v_id = version_id
    else:
        target_hash = doc.get("sha256_hash")
        v_id = f"VER-{document_id}-v{doc.get('current_version', 1)}"

    anchor_record = IntegrityAnchorService.anchor_hash(
        document_id=document_id,
        version_id=v_id,
        version_hash=target_hash,
        creator=current_user.user_id,
    )
    saved = registry.add_anchor(anchor_record)

    ip = request.client.host if request and request.client else None
    audit_logger.log(
        action=AuditAction.DOCUMENT_ANCHORED,
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
        resource_type="DOCUMENT_ANCHOR",
        resource_id=anchor_record["anchor_id"],
        ip_address=ip,
        status="SUCCESS",
        details={"document_id": document_id, "anchor_id": anchor_record["anchor_id"]},
    )

    return saved


@router.get(
    "/{document_id}/anchor",
    summary="List integrity anchor receipts for document",
)
def list_document_anchors(
    document_id: str,
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    if not can_access_confidentiality(current_user, doc.get("confidentiality_level", "RESTRICTED"), document_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return registry.get_anchors(document_id)

