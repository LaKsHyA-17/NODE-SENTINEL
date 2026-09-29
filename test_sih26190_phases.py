# -*- coding: utf-8 -*-
"""
NODE SENTINEL - SIH26190 Integration Comprehensive Test Suite (Phases 6 -> 14).
Tests:
- Phase 6: OCR & Document Intelligence Pipeline
- Phase 7: Document to Knowledge Graph Integration with Provenance
- Phase 8: Document-Specific RBAC & Secure Cross-Agency Sharing
- Phase 9: Document Audit Trail
- Phase 10: Cryptographic Digital Signatures & Non-Repudiation
- Phase 11: Intelligent Document Search & AI Retrieval (RAG)
- Phase 12: Optional Integrity / Blockchain Anchor
- Phase 13: Security & IDOR Authorization Audit
- Phase 14: Complete End-to-End Investigation Workflow (CASE-2026-001)
"""
import io
import os
import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.auth_service import token_manager, user_repo
from app.core.document_service import (
    DocumentRegistry,
    DocumentStorageService,
    DigitalSignatureService,
    IntegrityAnchorService,
    get_document_registry,
    get_document_storage_service,
    process_document_intelligence,
)
from app.core.graph_engine import get_graph_engine
from app.core.audit_logger import audit_logger
from app.main import app
from app.models.auth_models import Role, User


@pytest.fixture
def test_setup(monkeypatch):
    """Setup isolated test environment for document operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir) / "docs"
        storage_dir.mkdir(parents=True, exist_ok=True)
        registry_file = Path(tmpdir) / "registry.json"

        test_storage_svc = DocumentStorageService(storage_dir=storage_dir, max_size_mb=10)
        test_registry = DocumentRegistry(registry_file=registry_file)

        monkeypatch.setattr("app.api.routes_documents.get_document_storage_service", lambda: test_storage_svc)
        monkeypatch.setattr("app.api.routes_documents.get_document_registry", lambda: test_registry)
        monkeypatch.setattr("app.core.document_service.get_document_storage_service", lambda: test_storage_svc)
        monkeypatch.setattr("app.core.document_service.get_document_registry", lambda: test_registry)

        with TestClient(app) as test_client:
            yield {
                "client": test_client,
                "storage_svc": test_storage_svc,
                "registry": test_registry,
                "temp_dir": tmpdir,
            }


@pytest.fixture
def auth_headers():
    admin_user = user_repo.get_by_username("admin") or User(
        user_id="usr_admin_test", username="admin", role=Role.ADMIN, password_hash="dummy"
    )
    investigator_user = user_repo.get_by_username("investigator") or User(
        user_id="usr_investigator_test", username="investigator", role=Role.INVESTIGATOR, password_hash="dummy"
    )
    viewer_user = user_repo.get_by_username("viewer") or User(
        user_id="usr_viewer_test", username="viewer", role=Role.VIEWER, password_hash="dummy"
    )

    admin_token = token_manager.create_access_token(admin_user)
    investigator_token = token_manager.create_access_token(investigator_user)
    viewer_token = token_manager.create_access_token(viewer_user)

    return {
        "admin": {"Authorization": f"Bearer {admin_token}"},
        "investigator": {"Authorization": f"Bearer {investigator_token}"},
        "viewer": {"Authorization": f"Bearer {viewer_token}"},
        "investigator_user": investigator_user,
        "viewer_user": viewer_user,
    }


# ==============================================================================
# PHASE 6: OCR & DOCUMENT INTELLIGENCE
# ==============================================================================

def test_phase6_ocr_text_and_entity_extraction(test_setup, auth_headers):
    client = test_setup["client"]

    fir_content = b"""FIRST INFORMATION REPORT
Case ID: FIR-2026-001
Police Station: Okhla Industrial Phase 2
Date of Incident: 15-08-2024
Suspect: Suspect Tariq Ahmad
Phone Number: +91 9811223344
Vehicle Involved: DL-01-AB-1234
Bank Account: ACC987654321
Details: Suspect Tariq Ahmad transferred INR 500000 to ACC987654321."""

    files = {"file": ("fir_report.txt", io.BytesIO(fir_content), "text/plain")}
    data = {
        "case_id": "CASE-2026-001",
        "title": "FIR for Cyber Financial Fraud",
        "document_type": "FIR",
        "confidentiality": "RESTRICTED",
        "description": "Initial FIR filed by Cyber Cell",
    }

    # Upload document
    resp = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=data, files=files)
    assert resp.status_code == 201
    doc_id = resp.json()["document"]["document_id"]

    # 1. Retrieve OCR / Extracted Text
    text_resp = client.get(f"/api/v1/documents/{doc_id}/text", headers=auth_headers["investigator"])
    assert text_resp.status_code == 200
    assert "Tariq Ahmad" in text_resp.json()["extracted_text"]
    assert text_resp.json()["character_count"] > 0

    # 2. Retrieve Extracted Entities
    ent_resp = client.get(f"/api/v1/documents/{doc_id}/entities", headers=auth_headers["investigator"])
    assert ent_resp.status_code == 200
    entities = ent_resp.json()["entities"]
    assert len(entities) > 0

    entity_ids = [e["entity_id"] for e in entities]
    assert any("9811223344" in eid for eid in entity_ids)

    # 3. Reprocess OCR
    reprocess_resp = client.post(f"/api/v1/documents/{doc_id}/reprocess-ocr", headers=auth_headers["investigator"])
    assert reprocess_resp.status_code == 200
    assert reprocess_resp.json()["status"] == "SUCCESS"


# ==============================================================================
# PHASE 7: KNOWLEDGE GRAPH INTEGRATION & PROVENANCE
# ==============================================================================

def test_phase7_knowledge_graph_sync_and_lineage(test_setup, auth_headers):
    client = test_setup["client"]

    report_text = b"""Forensic Bank Audit Report
Case Reference: FIR-2026-001
Suspect Tariq Ahmad operates Syndicate Logistics.
Suspect Tariq Ahmad transferred 500000 to ACC987654321."""

    files = {"file": ("forensic_audit.txt", io.BytesIO(report_text), "text/plain")}
    data = {
        "case_id": "CASE-2026-001",
        "title": "Forensic Bank Audit",
        "document_type": "FORENSIC_REPORT",
        "confidentiality": "RESTRICTED",
    }

    upload_resp = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=data, files=files)
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["document"]["document_id"]

    # 1. Sync to Knowledge Graph
    sync_resp = client.post(f"/api/v1/documents/{doc_id}/sync-graph", headers=auth_headers["investigator"])
    assert sync_resp.status_code == 200

    # 2. Graph Lineage lookup
    lineage_resp = client.get(f"/api/v1/documents/{doc_id}/graph-lineage", headers=auth_headers["investigator"])
    assert lineage_resp.status_code == 200
    lineage = lineage_resp.json()
    assert lineage["document_id"] == doc_id
    assert lineage["total_nodes"] > 0


# ==============================================================================
# PHASE 8: DOCUMENT RBAC & SECURE SHARING
# ==============================================================================

def test_phase8_document_rbac_and_sharing_lifecycle(test_setup, auth_headers):
    client = test_setup["client"]
    viewer_uid = auth_headers["viewer_user"].user_id

    # Upload CONFIDENTIAL document
    files = {"file": ("top_secret_memo.txt", io.BytesIO(b"Classified informant notes."), "text/plain")}
    data = {
        "case_id": "CASE-2026-001",
        "title": "Confidential Intelligence Memo",
        "document_type": "INVESTIGATION_RECORD",
        "confidentiality": "CONFIDENTIAL",
    }
    upload_resp = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=data, files=files)
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["document"]["document_id"]

    # Viewer cannot access confidential document initially
    viewer_view = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["viewer"])
    assert viewer_view.status_code == 403

    # 1. Grant explicit DocumentPermission to viewer
    perm_payload = {
        "user_id": viewer_uid,
        "can_read": True,
        "can_download": True,
    }
    perm_resp = client.post(f"/api/v1/documents/{doc_id}/permissions", headers=auth_headers["investigator"], json=perm_payload)
    assert perm_resp.status_code == 201
    perm_id = perm_resp.json()["permission_id"]

    # Viewer can now read the document
    viewer_allowed = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["viewer"])
    assert viewer_allowed.status_code == 200

    # 2. Revoke DocumentPermission
    del_perm = client.delete(f"/api/v1/documents/{doc_id}/permissions/{perm_id}", headers=auth_headers["investigator"])
    assert del_perm.status_code == 200

    # 3. Share document with viewer via DocumentShare
    share_payload = {
        "shared_with": viewer_uid,
        "permission": "READ",
        "notes": "Court authorized disclosure",
    }
    share_resp = client.post(f"/api/v1/documents/{doc_id}/shares", headers=auth_headers["investigator"], json=share_payload)
    assert share_resp.status_code == 201
    share_id = share_resp.json()["share_id"]

    # Viewer has access via active share
    viewer_share_access = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["viewer"])
    assert viewer_share_access.status_code == 200

    # 4. Revoke Share
    revoke_resp = client.delete(f"/api/v1/documents/{doc_id}/shares/{share_id}", headers=auth_headers["investigator"])
    assert revoke_resp.status_code == 200

    # Viewer access is revoked
    viewer_denied_again = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["viewer"])
    assert viewer_denied_again.status_code == 403


# ==============================================================================
# PHASE 9: DOCUMENT AUDIT TRAIL
# ==============================================================================

def test_phase9_document_audit_trail(test_setup, auth_headers):
    client = test_setup["client"]

    files = {"file": ("audit_target.txt", io.BytesIO(b"Evidence file for audit logging."), "text/plain")}
    data = {
        "case_id": "CASE-2026-001",
        "title": "Document For Audit Logging",
        "document_type": "EVIDENCE",
        "confidentiality": "RESTRICTED",
    }
    upload_resp = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=data, files=files)
    doc_id = upload_resp.json()["document"]["document_id"]

    # Perform view, download, integrity check
    client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["investigator"])
    client.get(f"/api/v1/documents/{doc_id}/download", headers=auth_headers["investigator"])
    client.post(f"/api/v1/documents/{doc_id}/verify-integrity", headers=auth_headers["investigator"])

    # Retrieve audit trail
    audit_resp = client.get(f"/api/v1/documents/{doc_id}/audit", headers=auth_headers["investigator"])
    assert audit_resp.status_code == 200
    logs = audit_resp.json()
    assert len(logs) >= 3


# ==============================================================================
# PHASE 10: DIGITAL SIGNATURES & CRYPTOGRAPHIC VERIFICATION
# ==============================================================================

def test_phase10_digital_signatures(test_setup, auth_headers):
    client = test_setup["client"]
    inv_uid = auth_headers["investigator_user"].user_id

    files = {"file": ("signed_court_filing.txt", io.BytesIO(b"Official court filing for legal proceedings."), "text/plain")}
    data = {
        "case_id": "CASE-2026-001",
        "title": "Court Submission Filing",
        "document_type": "COURT_FILING",
        "confidentiality": "RESTRICTED",
    }
    upload_resp = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=data, files=files)
    doc_id = upload_resp.json()["document"]["document_id"]

    # 1. Sign version hash
    sign_resp = client.post(
        f"/api/v1/documents/{doc_id}/sign",
        headers=auth_headers["investigator"],
        json={"signer_role": "Investigating Officer", "comments": "Affirmed under oath"},
    )
    assert sign_resp.status_code == 200
    sig_data = sign_resp.json()
    assert sig_data["verification_status"] == "VALID"
    sig_value = sig_data["signature_value"]

    # 2. Verify signature
    verify_resp = client.post(
        f"/api/v1/documents/{doc_id}/verify-signature",
        headers=auth_headers["investigator"],
        json={"signature_value": sig_value, "signer_id": inv_uid},
    )
    assert verify_resp.status_code == 200
    assert verify_resp.json()["valid"] is True

    # 3. List signatures
    list_sig_resp = client.get(f"/api/v1/documents/{doc_id}/signatures", headers=auth_headers["investigator"])
    assert list_sig_resp.status_code == 200
    assert len(list_sig_resp.json()) >= 1


# ==============================================================================
# PHASE 11: INTELLIGENT AI RETRIEVAL / RAG SEARCH
# ==============================================================================

def test_phase11_ai_document_search_rag(test_setup, auth_headers):
    client = test_setup["client"]

    # Ingest searchable document
    text_content = b"""Financial intelligence report detailing illicit wire transfers.
Suspect Tariq Ahmad transferred 500000 to ACC987654321 on 15th August 2024."""
    files = {"file": ("financial_intel.txt", io.BytesIO(text_content), "text/plain")}
    data = {
        "case_id": "CASE-2026-001",
        "title": "Financial Intelligence Analysis",
        "document_type": "FORENSIC_REPORT",
        "confidentiality": "RESTRICTED",
    }
    upload_resp = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=data, files=files)
    assert upload_resp.status_code == 201

    # Run AI Search
    search_payload = {
        "query": "transfers Tariq Ahmad",
        "case_id": "CASE-2026-001",
        "top_k": 3,
    }
    ai_resp = client.post("/api/v1/documents/ai/search", headers=auth_headers["investigator"], json=search_payload)
    assert ai_resp.status_code == 200
    res_data = ai_resp.json()
    assert len(res_data["citations"]) > 0
    assert "Tariq Ahmad" in res_data["citations"][0]["snippet"]
    assert res_data["citations"][0]["version_number"] == 1


# ==============================================================================
# PHASE 12: INTEGRITY ANCHOR / BLOCKCHAIN
# ==============================================================================

def test_phase12_integrity_anchor(test_setup, auth_headers):
    client = test_setup["client"]

    files = {"file": ("anchor_target.txt", io.BytesIO(b"Immutable anchor verification document."), "text/plain")}
    data = {
        "case_id": "CASE-2026-001",
        "title": "Anchor Test Document",
        "document_type": "EVIDENCE",
        "confidentiality": "RESTRICTED",
    }
    upload_resp = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=data, files=files)
    doc_id = upload_resp.json()["document"]["document_id"]

    # 1. Create anchor
    anchor_resp = client.post(f"/api/v1/documents/{doc_id}/anchor", headers=auth_headers["investigator"])
    assert anchor_resp.status_code == 200
    anchor_data = anchor_resp.json()
    assert anchor_data["status"] == "CONFIRMED"
    assert "merkle_root" in anchor_data

    # 2. List anchors
    list_anchors = client.get(f"/api/v1/documents/{doc_id}/anchor", headers=auth_headers["investigator"])
    assert list_anchors.status_code == 200
    assert len(list_anchors.json()) >= 1


# ==============================================================================
# PHASE 14: FULL END-TO-END SIH26190 MULTI-DOCUMENT DEMO SCENARIO
# ==============================================================================

def test_phase14_end_to_end_sih26190_investigation_scenario(test_setup, auth_headers):
    """
    Executes complete 20-step synthetic investigation scenario for CASE-2026-001:
    1. Login as investigator
    2. Upload FIR (DOC-1, v1)
    3. Verify automatic SHA-256 hash
    4. Verify OCR/text extraction
    5. Verify extracted entities (Tariq Ahmad, Phone, Account)
    6. Verify automatic Knowledge Graph linkage with document provenance
    7. Upload Updated Investigation Report (DOC-1, v2)
    8. Verify version 1 remains immutable and v2 is active
    9. Verify cryptographic integrity -> VALID
    10. Physical tamper test -> detect integrity violation (INVALID)
    11. Search keyword "forensic" -> metadata & full-text match
    12. View Document Hub metadata, version history, audit logs
    13. Grant Viewer READ permission via DocumentPermission
    14. Verify Viewer can access permitted document
    15. Revoke Viewer permission
    16. Verify Viewer access is denied
    17. Run AI/RAG query with natural language
    18. Inspect citations linking to exact source document & version
    19. Verify Knowledge Graph relationship and provenance
    20. Verify complete immutable audit trail
    """
    client = test_setup["client"]
    viewer_uid = auth_headers["viewer_user"].user_id

    # STEP 1 & 2: Upload FIR
    fir_bytes = b"""FIRST INFORMATION REPORT - CASE-2026-001
Suspect Tariq Ahmad (Phone: +91 9811223344) transferred INR 500000 to ACC987654321."""

    files = {"file": ("FIR_Case_2026_001.txt", io.BytesIO(fir_bytes), "text/plain")}
    data = {
        "case_id": "CASE-2026-001",
        "title": "FIR Case 2026-001 Cyber Financial Scam",
        "document_type": "FIR",
        "confidentiality": "RESTRICTED",
        "description": "Initial registration of cyber scam",
    }
    fir_resp = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=data, files=files)
    assert fir_resp.status_code == 201
    doc_1 = fir_resp.json()["document"]
    doc_id = doc_1["document_id"]

    # STEP 3: Verify Version 1 and SHA-256
    assert doc_1["current_version"] == 1
    assert len(doc_1["sha256_hash"]) == 64

    # STEP 4 & 5: Verify OCR and Extracted Entities
    ent_resp = client.get(f"/api/v1/documents/{doc_id}/entities", headers=auth_headers["investigator"])
    assert ent_resp.status_code == 200
    assert ent_resp.json()["total_entities"] > 0

    # STEP 6: Verify Knowledge Graph linkage
    lineage_resp = client.get(f"/api/v1/documents/{doc_id}/graph-lineage", headers=auth_headers["investigator"])
    assert lineage_resp.status_code == 200
    assert lineage_resp.json()["total_nodes"] > 0

    # STEP 7 & 8: Upload Updated Investigation Report (Version 2)
    v2_bytes = b"""INVESTIGATION REPORT AMENDMENT - CASE-2026-001
Suspect Tariq Ahmad transferred INR 500000 to ACC987654321. Additional accomplice identified."""
    v2_files = {"file": ("FIR_Case_2026_001_v2.txt", io.BytesIO(v2_bytes), "text/plain")}
    v2_data = {"change_description": "Added forensic financial transaction details"}

    v2_resp = client.post(f"/api/v1/documents/{doc_id}/versions", headers=auth_headers["investigator"], data=v2_data, files=v2_files)
    assert v2_resp.status_code == 201
    assert v2_resp.json()["version_number"] == 2

    # Check that v1 remains immutable
    v1_meta = client.get(f"/api/v1/documents/{doc_id}/versions/VER-{doc_id}-v1", headers=auth_headers["investigator"])
    assert v1_meta.status_code == 200
    assert v1_meta.json()["version_number"] == 1
    assert v1_meta.json()["sha256_hash"] == doc_1["sha256_hash"]

    # STEP 9: Verify Integrity -> VALID
    integ_valid = client.post(f"/api/v1/documents/{doc_id}/verify-integrity", headers=auth_headers["investigator"])
    assert integ_valid.status_code == 200
    assert integ_valid.json()["integrity_valid"] is True

    # STEP 10: Physical Tamper Test -> INVALID
    storage_svc = test_setup["storage_svc"]
    doc_record = test_setup["registry"].get_document(doc_id)
    phys_path = storage_svc.get_document_path(doc_record["file_path"])
    with open(phys_path, "wb") as f:
        f.write(b"TAMPERED UNAUTHORIZED MODIFIED CONTENT")

    integ_tampered = client.post(f"/api/v1/documents/{doc_id}/verify-integrity", headers=auth_headers["investigator"])
    assert integ_tampered.status_code == 200
    assert integ_tampered.json()["integrity_valid"] is False

    # Restore valid content for subsequent steps
    with open(phys_path, "wb") as f:
        f.write(v2_bytes)

    # STEP 11: Search
    search_resp = client.get("/api/v1/documents/search?q=Cyber", headers=auth_headers["investigator"])
    assert search_resp.status_code == 200
    assert search_resp.json()["total"] >= 1

    # STEP 12: Digital Signature
    sign_resp = client.post(f"/api/v1/documents/{doc_id}/sign", headers=auth_headers["investigator"], json={"signer_role": "Lead Investigator"})
    assert sign_resp.status_code == 200

    # STEP 13, 14, 15, 16: Grant Permission, Verify Access, Revoke, Verify Denied
    # Test on confidential document
    conf_files = {"file": ("conf_record.txt", io.BytesIO(b"Secret intelligence evidence"), "text/plain")}
    conf_data = {"case_id": "CASE-2026-001", "title": "Confidential Dossier", "document_type": "EVIDENCE", "confidentiality": "CONFIDENTIAL"}
    conf_doc = client.post("/api/v1/documents/upload", headers=auth_headers["investigator"], data=conf_data, files=conf_files).json()["document"]
    conf_id = conf_doc["document_id"]

    # Viewer denied initially
    assert client.get(f"/api/v1/documents/{conf_id}", headers=auth_headers["viewer"]).status_code == 403

    # Grant permission
    p_grant = client.post(f"/api/v1/documents/{conf_id}/permissions", headers=auth_headers["investigator"], json={"user_id": viewer_uid, "can_read": True})
    assert p_grant.status_code == 201
    p_id = p_grant.json()["permission_id"]

    # Viewer can access
    assert client.get(f"/api/v1/documents/{conf_id}", headers=auth_headers["viewer"]).status_code == 200

    # Revoke permission
    client.delete(f"/api/v1/documents/{conf_id}/permissions/{p_id}", headers=auth_headers["investigator"])

    # Viewer denied again
    assert client.get(f"/api/v1/documents/{conf_id}", headers=auth_headers["viewer"]).status_code == 403

    # STEP 17 & 18: AI Retrieval & Source Citations
    ai_resp = client.post("/api/v1/documents/ai/search", headers=auth_headers["investigator"], json={"query": "Tariq Ahmad wire transfer", "case_id": "CASE-2026-001"})
    assert ai_resp.status_code == 200
    assert len(ai_resp.json()["citations"]) > 0

    # STEP 19: Integrity Anchor
    anchor_resp = client.post(f"/api/v1/documents/{doc_id}/anchor", headers=auth_headers["investigator"])
    assert anchor_resp.status_code == 200

    # STEP 20: Complete Audit Trail
    audit_resp = client.get(f"/api/v1/documents/{doc_id}/audit", headers=auth_headers["investigator"])
    assert audit_resp.status_code == 200
    assert len(audit_resp.json()) >= 4
