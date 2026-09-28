# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Document Repository & Evidentiary Registry Data Models (SIH26190 Phase 1).

Defines structured data models for:
- DocumentRecord (Centralized legal/investigation document registry)
- DocumentVersion (Immutable version control tracking)
- DocumentPermission (Granular user/role document-level access control)
- DocumentEntityLink (Linkages to extracted graph entities)
- DocumentShare (Time-bound and role-bound collaboration grants)
- DocumentSignatureMetadata (Cryptographic / PKI digital signature metadata)
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class DocumentType(str, Enum):
    """Statutory & investigative legal document classifications."""
    FIR = "FIR"
    POLICE_REPORT = "POLICE_REPORT"
    INVESTIGATION_RECORD = "INVESTIGATION_RECORD"
    WITNESS_STATEMENT = "WITNESS_STATEMENT"
    CHARGE_SHEET = "CHARGE_SHEET"
    COURT_FILING = "COURT_FILING"
    EVIDENCE = "EVIDENCE"
    FORENSIC_REPORT = "FORENSIC_REPORT"
    LEGAL_NOTICE = "LEGAL_NOTICE"
    JUDGMENT = "JUDGMENT"
    CASE_DIARY = "CASE_DIARY"
    SEIZURE_MEMO = "SEIZURE_MEMO"
    BAIL_APPLICATION = "BAIL_APPLICATION"
    REMAND_ORDER = "REMAND_ORDER"
    OTHER = "OTHER"


class DocumentStatus(str, Enum):
    """Lifecycle stages for investigative & court documents."""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    SEALED = "SEALED"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"


class ConfidentialityLevel(str, Enum):
    """Security classification levels for access control."""
    PUBLIC = "PUBLIC"
    RESTRICTED = "RESTRICTED"
    CONFIDENTIAL = "CONFIDENTIAL"
    TOP_SECRET = "TOP_SECRET"


class ShareStatus(str, Enum):
    """Status of a document collaboration grant."""
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class SignatureVerificationStatus(str, Enum):
    """Verification states for document signatures."""
    PENDING = "PENDING"
    VALID = "VALID"
    INVALID = "INVALID"
    REVOKED = "REVOKED"


def get_utc_now_iso() -> str:
    """Helper to return current ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class DocumentVersion(BaseModel):
    """Immutable version snapshot of an investigative or court filing."""
    version_id: str = Field(..., description="Unique version identifier, e.g. VER-DOC-2026-001-v1")
    document_id: str = Field(..., description="Parent document identifier")
    version_number: int = Field(1, ge=1, description="Incremental version number")
    parent_version_id: Optional[str] = Field(None, description="Previous version identifier if updated")
    file_path: Optional[str] = Field(None, description="Stored relative or absolute path on disk/blob storage")
    sha256_hash: str = Field(..., description="Cryptographic SHA-256 digest of this version's content")
    created_by: str = Field(..., description="User ID or Badge ID of creator")
    created_at: str = Field(default_factory=get_utc_now_iso, description="ISO UTC creation timestamp")
    change_reason: Optional[str] = Field(None, description="Summary of modifications or legal amendment reason")
    status: DocumentStatus = Field(DocumentStatus.SUBMITTED, description="Lifecycle status of this version")
    file_size: Optional[int] = Field(None, ge=0, description="Size in bytes of this version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual attributes")

    model_config = ConfigDict(populate_by_name=True)


class DocumentPermission(BaseModel):
    """Granular document-level access permission rule."""
    permission_id: str = Field(..., description="Unique permission rule identifier")
    document_id: str = Field(..., description="Target document identifier")
    user_id: Optional[str] = Field(None, description="Specific user identifier granted permission")
    role: Optional[str] = Field(None, description="Role identifier (ADMIN, INVESTIGATOR, ANALYST, VIEWER, etc.)")
    can_read: bool = Field(True, description="Permission to view document metadata and content")
    can_edit: bool = Field(False, description="Permission to upload new versions or edit metadata")
    can_download: bool = Field(True, description="Permission to download binary file")
    can_share: bool = Field(False, description="Permission to grant access to other officers")
    can_verify: bool = Field(True, description="Permission to perform integrity and signature checks")
    created_at: str = Field(default_factory=get_utc_now_iso, description="ISO UTC creation timestamp")

    model_config = ConfigDict(populate_by_name=True)


class DocumentEntityLink(BaseModel):
    """Traceable linkage connecting a document to a Knowledge Graph entity."""
    document_id: str = Field(..., description="Source document identifier")
    entity_id: str = Field(..., description="Target knowledge graph entity identifier")
    entity_type: str = Field(..., description="Entity category: Person, Phone, Vehicle, Case, BankAccount, Location, etc.")
    confidence: float = Field(0.95, ge=0.0, le=1.0, description="Extraction confidence score")
    extraction_method: str = Field("REGEX_NLP", description="Method: REGEX_NLP, SPACY_NER, OCR_PIPELINE, MANUAL_TAGGING")
    created_at: str = Field(default_factory=get_utc_now_iso, description="ISO UTC extraction timestamp")
    char_start: Optional[int] = Field(None, description="Character offset start in document text")
    char_end: Optional[int] = Field(None, description="Character offset end in document text")
    evidence_snippet: Optional[str] = Field(None, description="Verbatim text snippet containing mention")

    model_config = ConfigDict(populate_by_name=True)


class DocumentShare(BaseModel):
    """Record of an authorized document collaboration or cross-agency sharing grant."""
    share_id: str = Field(..., description="Unique sharing grant identifier")
    document_id: str = Field(..., description="Target document identifier")
    shared_by: str = Field(..., description="User ID of the granting officer")
    shared_with: str = Field(..., description="User ID, Agency ID, or Court Badge ID of recipient")
    permission: str = Field("READ", description="Granted access level: READ, COMMENT, DOWNLOAD, FULL")
    created_at: str = Field(default_factory=get_utc_now_iso, description="ISO UTC timestamp when share was initiated")
    expires_at: Optional[str] = Field(None, description="Optional ISO UTC expiration timestamp")
    status: ShareStatus = Field(ShareStatus.ACTIVE, description="Current share status")
    notes: Optional[str] = Field(None, description="Operational justification or court authorization notice")

    model_config = ConfigDict(populate_by_name=True)


class DocumentSignatureMetadata(BaseModel):
    """Digital signature metadata for legal non-repudiation and verification."""
    signature_id: str = Field(..., description="Unique digital signature metadata record identifier")
    document_id: str = Field(..., description="Document identifier signed")
    version_id: Optional[str] = Field(None, description="Specific document version signed")
    signer_id: str = Field(..., description="User ID or Certificate DN of the signing officer")
    algorithm: str = Field("SHA256withRSA", description="Cryptographic signing algorithm (e.g. SHA256withRSA, ECDSA-P256, Ed25519)")
    signature_value: str = Field(..., description="Hex or Base64 representation of digital signature")
    signed_at: str = Field(default_factory=get_utc_now_iso, description="ISO UTC signing timestamp")
    verification_status: SignatureVerificationStatus = Field(
        SignatureVerificationStatus.PENDING,
        description="Signature validity status"
    )
    certificate_fingerprint: Optional[str] = Field(None, description="SHA-256 fingerprint of the signer's X.509 certificate")
    signer_role: Optional[str] = Field(None, description="Signer's designation at time of signing")
    comments: Optional[str] = Field(None, description="Signing remarks or legal attestation notice")

    model_config = ConfigDict(populate_by_name=True)


class DocumentRecord(BaseModel):
    """
    Centralized Legal & Investigation Document Registry Record.
    Implements core SIH26190 document repository requirements with strict typing.
    """
    document_id: str = Field(..., description="Unique document record identifier (e.g. DOC-2026-001)")
    case_id: Optional[str] = Field(None, description="Associated legal or police Case / FIR ID")
    title: str = Field(..., min_length=1, description="Official title or description of filing")
    doc_type: DocumentType = Field(..., description="Categorized document type")
    file_name: str = Field(..., description="Original uploaded filename")
    file_size: int = Field(0, ge=0, description="File size in bytes")
    mime_type: str = Field("application/pdf", description="MIME content type")
    file_path: Optional[str] = Field(None, description="Storage location path")
    sha256_hash: str = Field(..., description="SHA-256 checksum of original file for evidentiary integrity")
    current_version: int = Field(1, ge=1, description="Current active version number")
    uploaded_by: str = Field(..., description="User ID or Badge number of uploading officer")
    created_at: str = Field(default_factory=get_utc_now_iso, description="ISO UTC timestamp of creation")
    updated_at: str = Field(default_factory=get_utc_now_iso, description="ISO UTC timestamp of last update")
    status: DocumentStatus = Field(DocumentStatus.SUBMITTED, description="Document lifecycle state")
    confidentiality_level: ConfidentialityLevel = Field(
        ConfidentialityLevel.RESTRICTED,
        description="Confidentiality and clearance classification"
    )
    description: Optional[str] = Field(None, description="Executive summary or legal remarks")
    tags: List[str] = Field(default_factory=list, description="Categorization tags or statute references")
    extracted_text: Optional[str] = Field(None, description="OCR / parsed textual content for full-text indexing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom statutory or forensic metadata")

    model_config = ConfigDict(populate_by_name=True)
