# -*- coding: utf-8 -*-
"""
Tests for SIH26190 Phase 1 Document Data Models.
Validates instantiation, serialization, defaults, and enum constraints for:
- DocumentRecord
- DocumentVersion
- DocumentPermission
- DocumentEntityLink
- DocumentShare
- DocumentSignatureMetadata
"""
import pytest
from app.models.document_models import (
    DocumentType,
    DocumentStatus,
    ConfidentialityLevel,
    ShareStatus,
    SignatureVerificationStatus,
    DocumentRecord,
    DocumentVersion,
    DocumentPermission,
    DocumentEntityLink,
    DocumentShare,
    DocumentSignatureMetadata,
)


def test_document_record_creation():
    doc = DocumentRecord(
        document_id="DOC-2026-0001",
        case_id="FIR-2026-NDPS-88",
        title="First Information Report - NDPS Seizure",
        doc_type=DocumentType.FIR,
        file_name="fir_2026_0001.pdf",
        file_size=102400,
        mime_type="application/pdf",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        uploaded_by="usr_investigator",
        tags=["NDPS", "Narcotics", "FIR"],
    )
    assert doc.document_id == "DOC-2026-0001"
    assert doc.doc_type == DocumentType.FIR
    assert doc.status == DocumentStatus.SUBMITTED
    assert doc.confidentiality_level == ConfidentialityLevel.RESTRICTED
    assert doc.current_version == 1
    assert isinstance(doc.created_at, str)
    assert isinstance(doc.updated_at, str)
    assert "NDPS" in doc.tags

    # Serialization test
    doc_dict = doc.model_dump()
    assert doc_dict["document_id"] == "DOC-2026-0001"
    assert doc_dict["doc_type"] == "FIR"


def test_document_type_enums():
    expected_types = [
        "FIR", "POLICE_REPORT", "INVESTIGATION_RECORD", "WITNESS_STATEMENT",
        "CHARGE_SHEET", "COURT_FILING", "EVIDENCE", "FORENSIC_REPORT",
        "LEGAL_NOTICE", "JUDGMENT", "CASE_DIARY", "SEIZURE_MEMO",
        "BAIL_APPLICATION", "REMAND_ORDER", "OTHER"
    ]
    for t in expected_types:
        assert DocumentType(t) is not None


def test_document_status_enums():
    expected_statuses = ["DRAFT", "SUBMITTED", "SEALED", "SUPERSEDED", "ARCHIVED"]
    for s in expected_statuses:
        assert DocumentStatus(s) is not None


def test_confidentiality_level_enums():
    expected_levels = ["PUBLIC", "RESTRICTED", "CONFIDENTIAL", "TOP_SECRET"]
    for cl in expected_levels:
        assert ConfidentialityLevel(cl) is not None


def test_document_version_model():
    version = DocumentVersion(
        version_id="VER-DOC-2026-0001-v1",
        document_id="DOC-2026-0001",
        version_number=1,
        sha256_hash="a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
        created_by="usr_investigator",
        change_reason="Initial FIR registration",
    )
    assert version.version_id == "VER-DOC-2026-0001-v1"
    assert version.version_number == 1
    assert version.parent_version_id is None
    assert version.status == DocumentStatus.SUBMITTED


def test_document_permission_model():
    perm = DocumentPermission(
        permission_id="PERM-001",
        document_id="DOC-2026-0001",
        role="INVESTIGATOR",
        can_read=True,
        can_edit=True,
        can_download=True,
        can_share=False,
        can_verify=True,
    )
    assert perm.permission_id == "PERM-001"
    assert perm.can_edit is True
    assert perm.can_share is False


def test_document_entity_link_model():
    link = DocumentEntityLink(
        document_id="DOC-2026-0001",
        entity_id="PERSON_TARIQ_AHMAD",
        entity_type="Person",
        confidence=0.98,
        extraction_method="REGEX_NLP",
        char_start=45,
        char_end=56,
        evidence_snippet="Accused Tariq Ahmad was apprehended",
    )
    assert link.entity_id == "PERSON_TARIQ_AHMAD"
    assert link.confidence == 0.98
    assert link.evidence_snippet == "Accused Tariq Ahmad was apprehended"


def test_document_share_model():
    share = DocumentShare(
        share_id="SHARE-001",
        document_id="DOC-2026-0001",
        shared_by="usr_investigator",
        shared_with="AGENCY_CBI_UNIT_4",
        permission="READ",
        status=ShareStatus.ACTIVE,
    )
    assert share.share_id == "SHARE-001"
    assert share.status == ShareStatus.ACTIVE
    assert share.shared_with == "AGENCY_CBI_UNIT_4"


def test_document_signature_metadata_model():
    sig = DocumentSignatureMetadata(
        signature_id="SIG-001",
        document_id="DOC-2026-0001",
        version_id="VER-DOC-2026-0001-v1",
        signer_id="usr_sho_officer",
        algorithm="SHA256withRSA",
        signature_value="deadbeef12345678",
        verification_status=SignatureVerificationStatus.VALID,
    )
    assert sig.signature_id == "SIG-001"
    assert sig.algorithm == "SHA256withRSA"
    assert sig.verification_status == SignatureVerificationStatus.VALID
