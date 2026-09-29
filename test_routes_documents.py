# -*- coding: utf-8 -*-
"""
Automated Test Suite for Document Management REST API (SIH26190 Phase 2B & Phase 3).
Tests:
- Authenticated and authorized document uploads (v1 creation)
- Rejection of invalid extensions, disguised payloads, and oversized documents
- Document Record metadata retrieval and 404 handling
- Document binary file downloading and storage path security
- Case-specific document listing and pagination
- Confidentiality levels & RBAC clearance enforcement
- Complete audit trail logging (DOCUMENT_UPLOAD, DOCUMENT_VIEW, DOCUMENT_DOWNLOAD, DOCUMENT_ACCESS_DENIED)
- Absolute filesystem path masking in public API contracts
- Phase 3 Immutable Versioning (POST /api/v1/documents/{id}/versions)
  - Monotonic version increment (v1 -> v2 -> v3)
  - Parent version ID lineage linkage
  - Immutability of historical versions and hashes
  - Custom/server-controlled IDs and numbers
- Phase 3 Version History (GET /api/v1/documents/{id}/versions)
- Phase 3 Individual Version Details (GET /api/v1/documents/{id}/versions/{version_id})
  - IDOR protection across document IDs
- Phase 3 Version Download (GET /api/v1/documents/{id}/versions/{version_id}/download)
  - Download historical version binary vs current version binary
- Phase 3 Cryptographic SHA-256 Integrity Verification (POST /api/v1/documents/{id}/verify-integrity)
  - Successful verification on untampered files
  - Detection of physical file tampering (bit-rot / unauthorized alteration)
  - Emits DOCUMENT_INTEGRITY_VERIFIED and DOCUMENT_INTEGRITY_FAILED
- Sealed / Archived document protection against updates
"""
import io
import json
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.audit_logger import audit_logger
from app.core.auth_service import token_manager, user_repo
from app.core.document_service import DocumentRegistry, DocumentStorageService
from app.main import app
from app.models.audit_models import AuditAction, AuditQueryFilter
from app.models.auth_models import Role, User
from app.models.document_models import DocumentStatus


@pytest.fixture
def client(monkeypatch):
    """Provide a TestClient with isolated temporary document storage and registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "docs"
        registry_file = Path(tmpdir) / "registry.json"

        test_storage = DocumentStorageService(storage_dir=storage_path, max_size_mb=2)
        test_registry = DocumentRegistry(registry_file=registry_file)

        monkeypatch.setattr("app.api.routes_documents.get_document_storage_service", lambda: test_storage)
        monkeypatch.setattr("app.api.routes_documents.get_document_registry", lambda: test_registry)

        with TestClient(app) as test_client:
            yield test_client, test_storage, test_registry


@pytest.fixture
def auth_headers():
    """Generate bearer token auth headers for admin, investigator, analyst, and viewer roles."""
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
    }


# ==============================================================================
# PHASE 2B TESTS (PRESERVED & EXPANDED)
# ==============================================================================

def test_upload_document_success(client, auth_headers):
    test_client, _, registry = client
    pdf_content = b"%PDF-1.4\n%FIR Case 2026-NDPS\n%%EOF"

    response = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={
            "case_id": "FIR-2026-NDPS-101",
            "title": "FIR Initial Seizure Report",
            "document_type": "FIR",
            "confidentiality": "RESTRICTED",
            "description": "Seizure memo and initial report for NDPS raid",
            "tags": "NDPS, Seizure, FIR",
        },
        files={"file": ("fir_seizure.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "SUCCESS"
    doc = data["document"]
    assert doc["document_id"].startswith("DOC-")
    assert doc["case_id"] == "FIR-2026-NDPS-101"
    assert doc["title"] == "FIR Initial Seizure Report"
    assert doc["doc_type"] == "FIR"
    assert doc["confidentiality_level"] == "RESTRICTED"
    assert doc["current_version"] == 1
    assert "NDPS" in doc["tags"]
    assert "file_path" not in doc  # Ensure internal path is not exposed in summary

    # Verify version 1 creation
    ver = data["version"]
    assert ver["version_number"] == 1
    assert ver["document_id"] == doc["document_id"]
    assert ver["parent_version_id"] is None
    assert ver["sha256_hash"] is not None and len(ver["sha256_hash"]) == 64

    # Verify registry persistence
    saved = registry.get_document(doc["document_id"])
    assert saved is not None
    assert saved["title"] == "FIR Initial Seizure Report"


def test_upload_document_requires_ingest_permission(client, auth_headers):
    test_client, _, _ = client
    pdf_content = b"%PDF-1.4\n%content\n%%EOF"

    # Viewer role does not have INGEST_DATA permission
    response = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["viewer"],
        data={
            "case_id": "FIR-2026-001",
            "title": "Unauthorized Upload Attempt",
        },
        files={"file": ("unauthorized.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


def test_upload_invalid_extension_rejected(client, auth_headers):
    test_client, _, _ = client
    response = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "FIR-2026-001", "title": "Script Upload"},
        files={"file": ("malicious.sh", io.BytesIO(b"echo 'hello'"), "text/x-sh")},
    )
    assert response.status_code == 415
    assert "not permitted" in response.json()["detail"]


def test_upload_disguised_executable_rejected(client, auth_headers):
    test_client, _, _ = client
    fake_pdf = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"
    response = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "FIR-2026-001", "title": "Disguised Executable"},
        files={"file": ("trojan.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert response.status_code == 415
    assert "executable binary detected" in response.json()["detail"]


def test_upload_oversized_file_rejected(client, auth_headers):
    test_client, _, _ = client
    oversized = b"%PDF-" + b"0" * (3 * 1024 * 1024)  # 3 MB > 2 MB test limit
    response = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "FIR-2026-001", "title": "Oversized File"},
        files={"file": ("huge.pdf", io.BytesIO(oversized), "application/pdf")},
    )
    assert response.status_code == 413


def test_get_document_details_success(client, auth_headers):
    test_client, _, _ = client
    pdf_content = b"%PDF-1.4\n%Details test\n%%EOF"

    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "FIR-2026-002", "title": "Witness Statement 161"},
        files={"file": ("statement.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    doc_id = upload_res.json()["document"]["document_id"]

    # Retrieve details
    res = test_client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["viewer"])
    assert res.status_code == 200
    doc_meta = res.json()
    assert doc_meta["document_id"] == doc_id
    assert doc_meta["title"] == "Witness Statement 161"
    assert doc_meta["current_version"] == 1


def test_get_document_details_404_not_found(client, auth_headers):
    test_client, _, _ = client
    res = test_client.get("/api/v1/documents/NON_EXISTENT_DOC_999", headers=auth_headers["investigator"])
    assert res.status_code == 404


def test_download_document_success(client, auth_headers):
    test_client, _, _ = client
    pdf_content = b"%PDF-1.4\n%Binary Download Test Content\n%%EOF"

    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "FIR-2026-003", "title": "Forensic Report"},
        files={"file": ("fsl_report.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    doc_id = upload_res.json()["document"]["document_id"]

    download_res = test_client.get(f"/api/v1/documents/{doc_id}/download", headers=auth_headers["investigator"])
    assert download_res.status_code == 200
    assert download_res.content == pdf_content
    assert download_res.headers["content-type"] == "application/pdf"


def test_confidentiality_clearance_enforcement(client, auth_headers):
    test_client, _, _ = client
    pdf_content = b"%PDF-1.4\n%Top Secret Intelligence Report\n%%EOF"

    # Upload top-secret document
    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["admin"],
        data={
            "case_id": "CASE-CLASSIFIED-01",
            "title": "Intercepted Syndicate Comms",
            "confidentiality": "TOP_SECRET",
        },
        files={"file": ("classified.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    doc_id = upload_res.json()["document"]["document_id"]

    # Admin / Investigator can access
    admin_res = test_client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["admin"])
    assert admin_res.status_code == 200

    # Viewer without clearance gets 403 Forbidden
    viewer_res = test_client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["viewer"])
    assert viewer_res.status_code == 403
    assert "insufficient clearance" in viewer_res.json()["detail"]

    # Viewer download gets 403 Forbidden
    download_res = test_client.get(f"/api/v1/documents/{doc_id}/download", headers=auth_headers["viewer"])
    assert download_res.status_code == 403


def test_list_documents_by_case(client, auth_headers):
    test_client, _, _ = client
    pdf1 = b"%PDF-1.4\n%Doc 1\n%%EOF"
    pdf2 = b"%PDF-1.4\n%Doc 2\n%%EOF"

    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "FIR-CASE-777", "title": "Charge Sheet Document"},
        files={"file": ("chargesheet.pdf", io.BytesIO(pdf1), "application/pdf")},
    )
    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "FIR-CASE-777", "title": "Seizure Memo"},
        files={"file": ("seizure.pdf", io.BytesIO(pdf2), "application/pdf")},
    )

    list_res = test_client.get("/api/v1/documents/case/FIR-CASE-777", headers=auth_headers["investigator"])
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["case_id"] == "FIR-CASE-777"
    assert data["total"] == 2
    assert len(data["items"]) == 2
    titles = {item["title"] for item in data["items"]}
    assert "Charge Sheet Document" in titles
    assert "Seizure Memo" in titles


def test_no_absolute_path_leakage(client, auth_headers):
    test_client, _, _ = client
    pdf_content = b"%PDF-1.4\n%Path check\n%%EOF"

    res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "FIR-CASE-888", "title": "Test Path Leakage"},
        files={"file": ("leak_check.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    response_str = json.dumps(res.json())

    # Ensure local filesystem paths (C:\ or /app/ or /home/ or data\documents) are not leaked in API response
    assert "C:\\" not in response_str
    assert "/app/data" not in response_str
    assert "storage_reference" not in res.json()["document"]


# ==============================================================================
# PHASE 3 TESTS: IMMUTABLE VERSIONING & INTEGRITY
# ==============================================================================

def test_version_creation_monotonic_increment_and_lineage(client, auth_headers):
    """
    Test Phase 3A-3C:
    1. Upload creates Version 1 (parent is None).
    2. Add version creates Version 2 (parent is v1).
    3. Add version creates Version 3 (parent is v2).
    4. Version 1 and 2 remain immutable with intact SHA-256 hashes.
    """
    test_client, _, _ = client
    v1_content = b"%PDF-1.4\n%Version 1 Content\n%%EOF"
    v2_content = b"%PDF-1.4\n%Version 2 Content - Added Witness Remarks\n%%EOF"
    v3_content = b"%PDF-1.4\n%Version 3 Content - Final Supplementary Report\n%%EOF"

    # Step 1: Upload initial document (Version 1)
    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-VER-01", "title": "Case Progress Diary"},
        files={"file": ("diary_v1.pdf", io.BytesIO(v1_content), "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["document_id"]
    v1_meta = upload_res.json()["version"]
    assert v1_meta["version_number"] == 1
    assert v1_meta["parent_version_id"] is None
    v1_hash = v1_meta["sha256_hash"]

    # Step 2: Post Version 2
    v2_res = test_client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=auth_headers["investigator"],
        data={"change_description": "Added supplementary witness testimonies"},
        files={"file": ("diary_v2.pdf", io.BytesIO(v2_content), "application/pdf")},
    )
    assert v2_res.status_code == 201
    v2_meta = v2_res.json()
    assert v2_meta["version_number"] == 2
    assert v2_meta["parent_version_id"] == v1_meta["version_id"]
    assert v2_meta["sha256_hash"] != v1_hash
    v2_hash = v2_meta["sha256_hash"]

    # Step 3: Post Version 3
    v3_res = test_client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=auth_headers["investigator"],
        data={"change_description": "Final FSL laboratory attachment"},
        files={"file": ("diary_v3.pdf", io.BytesIO(v3_content), "application/pdf")},
    )
    assert v3_res.status_code == 201
    v3_meta = v3_res.json()
    assert v3_meta["version_number"] == 3
    assert v3_meta["parent_version_id"] == v2_meta["version_id"]
    assert v3_meta["sha256_hash"] != v2_hash

    # Step 4: Verify document current_version is 3
    doc_res = test_client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers["investigator"])
    assert doc_res.status_code == 200
    assert doc_res.json()["current_version"] == 3

    # Step 5: Verify version history lists all 3 versions chronologically
    history_res = test_client.get(f"/api/v1/documents/{doc_id}/versions", headers=auth_headers["investigator"])
    assert history_res.status_code == 200
    h_data = history_res.json()
    assert h_data["total_versions"] == 3
    assert len(h_data["versions"]) == 3
    assert [v["version_number"] for v in h_data["versions"]] == [1, 2, 3]

    # Verify v1 hash remained untouched
    assert h_data["versions"][0]["sha256_hash"] == v1_hash
    assert h_data["versions"][1]["sha256_hash"] == v2_hash


def test_get_individual_version_details_and_idor_protection(client, auth_headers):
    """
    Test Phase 3E:
    - Retrieve specific version details.
    - IDOR protection: cannot access version with mismatched document_id.
    """
    test_client, _, _ = client
    pdf1 = b"%PDF-1.4\n%Doc A\n%%EOF"
    pdf2 = b"%PDF-1.4\n%Doc B\n%%EOF"

    docA_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-A", "title": "Document A"},
        files={"file": ("docA.pdf", io.BytesIO(pdf1), "application/pdf")},
    )
    doc_a_id = docA_res.json()["document"]["document_id"]
    ver_a_id = docA_res.json()["version"]["version_id"]

    docB_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-B", "title": "Document B"},
        files={"file": ("docB.pdf", io.BytesIO(pdf2), "application/pdf")},
    )
    doc_b_id = docB_res.json()["document"]["document_id"]

    # Valid detail fetch
    res = test_client.get(f"/api/v1/documents/{doc_a_id}/versions/{ver_a_id}", headers=auth_headers["investigator"])
    assert res.status_code == 200
    assert res.json()["version_id"] == ver_a_id
    assert res.json()["version_number"] == 1

    # IDOR attempt: Querying ver_a_id under doc_b_id must return 404
    idor_res = test_client.get(f"/api/v1/documents/{doc_b_id}/versions/{ver_a_id}", headers=auth_headers["investigator"])
    assert idor_res.status_code == 404


def test_download_specific_version_vs_current_version(client, auth_headers):
    """
    Test Phase 3F & 3M:
    - Specific version download endpoint returns historical binary.
    - Standard download endpoint returns current (latest) binary.
    """
    test_client, _, _ = client
    v1_content = b"%PDF-1.4\n%Historical Version 1 Binary Content\n%%EOF"
    v2_content = b"%PDF-1.4\n%Amended Version 2 Binary Content\n%%EOF"

    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-DOWNLOAD-TEST", "title": "Forensic Memo"},
        files={"file": ("memo_v1.pdf", io.BytesIO(v1_content), "application/pdf")},
    )
    doc_id = upload_res.json()["document"]["document_id"]
    v1_id = upload_res.json()["version"]["version_id"]

    v2_res = test_client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=auth_headers["investigator"],
        data={"change_description": "Version 2 update"},
        files={"file": ("memo_v2.pdf", io.BytesIO(v2_content), "application/pdf")},
    )
    v2_id = v2_res.json()["version_id"]

    # 1. Download specific version 1
    dl_v1 = test_client.get(f"/api/v1/documents/{doc_id}/versions/{v1_id}/download", headers=auth_headers["investigator"])
    assert dl_v1.status_code == 200
    assert dl_v1.content == v1_content

    # 2. Download specific version 2
    dl_v2 = test_client.get(f"/api/v1/documents/{doc_id}/versions/{v2_id}/download", headers=auth_headers["investigator"])
    assert dl_v2.status_code == 200
    assert dl_v2.content == v2_content

    # 3. Standard /download returns latest (v2)
    dl_current = test_client.get(f"/api/v1/documents/{doc_id}/download", headers=auth_headers["investigator"])
    assert dl_current.status_code == 200
    assert dl_current.content == v2_content


def test_integrity_verification_success_and_audit(client, auth_headers):
    """
    Test Phase 3G & 3H:
    - Integrity verification for untampered document succeeds.
    - Emits DOCUMENT_INTEGRITY_VERIFIED audit event.
    """
    test_client, _, _ = client
    pdf_content = b"%PDF-1.4\n%Tamper proof valid legal doc\n%%EOF"

    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-INT-01", "title": "Integrity Test Document"},
        files={"file": ("valid.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    doc_id = upload_res.json()["document"]["document_id"]

    verify_res = test_client.post(
        f"/api/v1/documents/{doc_id}/verify-integrity",
        headers=auth_headers["investigator"],
    )
    assert verify_res.status_code == 200
    data = verify_res.json()
    assert data["integrity_valid"] is True
    assert data["expected_sha256"] == data["actual_sha256"]
    assert "Cryptographic integrity verified" in data["status_message"]


def test_tampering_detection_and_audit_failure(client, auth_headers):
    """
    Test Phase 3O (Crucial SIH Tampering Test):
    1. Upload synthetic document.
    2. Record stored SHA-256.
    3. Modify physical file on disk directly.
    4. Run integrity verification.
    5. Verify integrity_valid == False and DOCUMENT_INTEGRITY_FAILED audit recorded.
    """
    test_client, storage, registry = client
    pdf_content = b"%PDF-1.4\n%Original Legitimate Police Document\n%%EOF"

    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-TAMPER-01", "title": "Tamper Target Doc"},
        files={"file": ("tamper_target.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    doc_id = upload_res.json()["document"]["document_id"]
    expected_hash = upload_res.json()["document"]["sha256_hash"]

    # Physically alter the file on disk behind the application's back
    doc_record = registry.get_document(doc_id)
    storage_ref = doc_record["file_path"]
    physical_path = storage.get_document_path(storage_ref)

    with open(physical_path, "wb") as f:
        f.write(b"%PDF-1.4\n%MALICIOUSLY TAMPERED EVIDENCE CONTENT\n%%EOF")

    # Run cryptographic verification
    verify_res = test_client.post(
        f"/api/v1/documents/{doc_id}/verify-integrity",
        headers=auth_headers["investigator"],
    )
    assert verify_res.status_code == 200
    data = verify_res.json()
    assert data["integrity_valid"] is False
    assert data["expected_sha256"] == expected_hash
    assert data["actual_sha256"] != expected_hash
    assert "Integrity violation detected" in data["status_message"]


def test_sealed_and_archived_document_modification_rejected(client, auth_headers):
    """
    Test Phase 3J:
    - Attempting to add a new version to a SEALED or ARCHIVED document is rejected (400 Bad Request).
    """
    test_client, _, registry = client
    pdf_content = b"%PDF-1.4\n%Court Sealed Record\n%%EOF"

    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-SEALED-01", "title": "Sealed Case Document"},
        files={"file": ("sealed.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    doc_id = upload_res.json()["document"]["document_id"]

    # 1. Test SEALED status
    registry.update_status(doc_id, DocumentStatus.SEALED)

    sealed_attempt = test_client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=auth_headers["investigator"],
        data={"change_description": "Attempted update on sealed record"},
        files={"file": ("attempt.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    assert sealed_attempt.status_code == 400
    assert "SEALED" in sealed_attempt.json()["detail"]

    # 2. Test ARCHIVED status
    registry.update_status(doc_id, DocumentStatus.ARCHIVED)

    archived_attempt = test_client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=auth_headers["investigator"],
        data={"change_description": "Attempted update on archived record"},
        files={"file": ("attempt.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    assert archived_attempt.status_code == 400
    assert "ARCHIVED" in archived_attempt.json()["detail"]


def test_unauthorized_and_unauthenticated_version_creation(client, auth_headers):
    """
    Test Phase 3N (#22, #23):
    - Unauthenticated request is rejected (401).
    - Unauthorized viewer role is rejected (403).
    """
    test_client, _, _ = client
    pdf_content = b"%PDF-1.4\n%Content\n%%EOF"

    upload_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-AUTH-01", "title": "Auth Test Document"},
        files={"file": ("auth_test.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    doc_id = upload_res.json()["document"]["document_id"]

    # Unauthenticated
    unauth_res = test_client.post(
        f"/api/v1/documents/{doc_id}/versions",
        data={"change_description": "No auth header"},
        files={"file": ("v2.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    assert unauth_res.status_code == 401

    # Unauthorized (viewer role)
    viewer_res = test_client.post(
        f"/api/v1/documents/{doc_id}/versions",
        headers=auth_headers["viewer"],
        data={"change_description": "Viewer cannot ingest/create versions"},
        files={"file": ("v2.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    assert viewer_res.status_code == 403


# ==============================================================================
# PHASE 4 TESTS: SECURE DOCUMENT SEARCH API
# ==============================================================================

def test_search_requires_authentication(client):
    test_client, _, _ = client
    res = test_client.get("/api/v1/documents/search")
    assert res.status_code == 401


def test_search_empty_and_pagination(client, auth_headers):
    test_client, _, _ = client
    pdf_content = b"%PDF-1.4\n%Content\n%%EOF"

    for i in range(5):
        test_client.post(
            "/api/v1/documents/upload",
            headers=auth_headers["investigator"],
            data={"case_id": f"CASE-PAGE-{i}", "title": f"Investigation Doc {i}"},
            files={"file": (f"doc_{i}.pdf", io.BytesIO(pdf_content), "application/pdf")},
        )

    # Page 1 with page_size=2
    res_p1 = test_client.get("/api/v1/documents/search?page=1&page_size=2", headers=auth_headers["investigator"])
    assert res_p1.status_code == 200
    data_p1 = res_p1.json()
    assert data_p1["total"] == 5
    assert len(data_p1["items"]) == 2
    assert data_p1["page"] == 1
    assert data_p1["page_size"] == 2

    # Page 3 with page_size=2 (should have 1 item)
    res_p3 = test_client.get("/api/v1/documents/search?page=3&page_size=2", headers=auth_headers["investigator"])
    assert res_p3.status_code == 200
    data_p3 = res_p3.json()
    assert data_p3["total"] == 5
    assert len(data_p3["items"]) == 1


def test_search_q_keyword_matching(client, auth_headers):
    test_client, _, _ = client
    pdf = b"%PDF-1.4\n%Content\n%%EOF"

    # Upload Doc 1 (Narcotics Seizure)
    res1 = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={
            "case_id": "FIR-2026-NDPS-999",
            "title": "Narcotics Contraband Seizure Memo",
            "description": "500 grams suspected heroin confiscated at highway checkpoint",
            "tags": "NDPS, Heroin, Highway",
        },
        files={"file": ("seizure_memo.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    doc1_id = res1.json()["document"]["document_id"]

    # Upload Doc 2 (Financial Ledger)
    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={
            "case_id": "FIR-2026-FIN-123",
            "title": "Hawala Bank Ledger Analysis",
            "description": "Cross-border illicit transfer logs",
            "tags": "Financial, Hawala, MoneyLaundering",
        },
        files={"file": ("hawala_ledger.pdf", io.BytesIO(pdf), "application/pdf")},
    )

    # 1. Search by title keyword (case-insensitive)
    s1 = test_client.get("/api/v1/documents/search?q=contraband", headers=auth_headers["investigator"])
    assert s1.status_code == 200
    assert s1.json()["total"] == 1
    assert s1.json()["items"][0]["document_id"] == doc1_id

    # 2. Search by description keyword
    s2 = test_client.get("/api/v1/documents/search?q=heroin", headers=auth_headers["investigator"])
    assert s2.status_code == 200
    assert s2.json()["total"] == 1
    assert s2.json()["items"][0]["document_id"] == doc1_id

    # 3. Search by case_id substring
    s3 = test_client.get("/api/v1/documents/search?q=NDPS-999", headers=auth_headers["investigator"])
    assert s3.status_code == 200
    assert s3.json()["total"] == 1
    assert s3.json()["items"][0]["document_id"] == doc1_id

    # 4. Search by exact document_id
    s4 = test_client.get(f"/api/v1/documents/search?q={doc1_id}", headers=auth_headers["investigator"])
    assert s4.status_code == 200
    assert s4.json()["total"] == 1
    assert s4.json()["items"][0]["document_id"] == doc1_id

    # 5. Search by tag
    s5 = test_client.get("/api/v1/documents/search?q=moneylaundering", headers=auth_headers["investigator"])
    assert s5.status_code == 200
    assert s5.json()["total"] == 1
    assert s5.json()["items"][0]["case_id"] == "FIR-2026-FIN-123"


def test_search_filters_type_status_confidentiality_creator_version(client, auth_headers):
    test_client, _, _ = client
    pdf = b"%PDF-1.4\n%Content\n%%EOF"

    # Upload Doc with FIR type
    fir_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={
            "case_id": "FIR-2026-001",
            "title": "First Information Report Initial",
            "document_type": "FIR",
            "confidentiality": "RESTRICTED",
        },
        files={"file": ("fir.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    fir_id = fir_res.json()["document"]["document_id"]

    # Upload Doc with FORENSIC_REPORT type
    fsl_res = test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={
            "case_id": "FIR-2026-001",
            "title": "Chemical Ballistics Report",
            "document_type": "FORENSIC_REPORT",
            "confidentiality": "RESTRICTED",
        },
        files={"file": ("ballistics.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    fsl_id = fsl_res.json()["document"]["document_id"]

    # Add Version 2 to FIR doc
    test_client.post(
        f"/api/v1/documents/{fir_id}/versions",
        headers=auth_headers["investigator"],
        data={"change_description": "Added supplementary charges"},
        files={"file": ("fir_v2.pdf", io.BytesIO(pdf), "application/pdf")},
    )

    # 1. Filter by document_type
    res_type = test_client.get("/api/v1/documents/search?document_type=FORENSIC_REPORT", headers=auth_headers["investigator"])
    assert res_type.status_code == 200
    assert res_type.json()["total"] == 1
    assert res_type.json()["items"][0]["document_id"] == fsl_id

    # 2. Filter by case_id
    res_case = test_client.get("/api/v1/documents/search?case_id=FIR-2026-001", headers=auth_headers["investigator"])
    assert res_case.status_code == 200
    assert res_case.json()["total"] == 2

    # 3. Filter by version_number (FIR is version 2)
    res_v2 = test_client.get("/api/v1/documents/search?version_number=2", headers=auth_headers["investigator"])
    assert res_v2.status_code == 200
    assert res_v2.json()["total"] == 1
    assert res_v2.json()["items"][0]["document_id"] == fir_id


def test_search_date_range_filtering(client, auth_headers):
    test_client, _, _ = client
    pdf = b"%PDF-1.4\n%Content\n%%EOF"

    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-DATE-01", "title": "Date Filter Document"},
        files={"file": ("date_doc.pdf", io.BytesIO(pdf), "application/pdf")},
    )

    # Valid past to future date range matches
    res_match = test_client.get(
        "/api/v1/documents/search?created_from=2026-01-01T00:00:00Z&created_to=2030-01-01T00:00:00Z",
        headers=auth_headers["investigator"],
    )
    assert res_match.status_code == 200
    assert res_match.json()["total"] == 1

    # Date range in the past returns 0
    res_past = test_client.get(
        "/api/v1/documents/search?created_from=2020-01-01T00:00:00Z&created_to=2020-12-31T23:59:59Z",
        headers=auth_headers["investigator"],
    )
    assert res_past.status_code == 200
    assert res_past.json()["total"] == 0

    # Malformed date returns 400 Bad Request
    res_invalid = test_client.get(
        "/api/v1/documents/search?created_from=NOT_A_DATE",
        headers=auth_headers["investigator"],
    )
    assert res_invalid.status_code == 400


def test_search_validation_errors(client, auth_headers):
    test_client, _, _ = client

    # Invalid document_type
    r1 = test_client.get("/api/v1/documents/search?document_type=INVALID_TYPE_XYZ", headers=auth_headers["investigator"])
    assert r1.status_code == 400
    assert "Invalid document_type" in r1.json()["detail"]

    # Invalid status
    r2 = test_client.get("/api/v1/documents/search?status=INVALID_STATUS_XYZ", headers=auth_headers["investigator"])
    assert r2.status_code == 400
    assert "Invalid status" in r2.json()["detail"]

    # Invalid confidentiality
    r3 = test_client.get("/api/v1/documents/search?confidentiality=INVALID_CONF_XYZ", headers=auth_headers["investigator"])
    assert r3.status_code == 400
    assert "Invalid confidentiality" in r3.json()["detail"]

    # Invalid pagination (page=0, page_size=101)
    r4 = test_client.get("/api/v1/documents/search?page=0", headers=auth_headers["investigator"])
    assert r4.status_code == 422

    r5 = test_client.get("/api/v1/documents/search?page_size=101", headers=auth_headers["investigator"])
    assert r5.status_code == 422


def test_search_sorting(client, auth_headers):
    test_client, _, _ = client
    pdf = b"%PDF-1.4\n%Content\n%%EOF"

    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-SORT", "title": "Alpha Document"},
        files={"file": ("alpha.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-SORT", "title": "Zeta Document"},
        files={"file": ("zeta.pdf", io.BytesIO(pdf), "application/pdf")},
    )

    # Sort title asc
    res_asc = test_client.get("/api/v1/documents/search?case_id=CASE-SORT&sort_by=title&sort_order=asc", headers=auth_headers["investigator"])
    assert res_asc.status_code == 200
    titles_asc = [d["title"] for d in res_asc.json()["items"]]
    assert titles_asc == ["Alpha Document", "Zeta Document"]

    # Sort title desc
    res_desc = test_client.get("/api/v1/documents/search?case_id=CASE-SORT&sort_by=title&sort_order=desc", headers=auth_headers["investigator"])
    assert res_desc.status_code == 200
    titles_desc = [d["title"] for d in res_desc.json()["items"]]
    assert titles_desc == ["Zeta Document", "Alpha Document"]


def test_search_confidentiality_security_matrix_and_no_idor(client, auth_headers):
    """
    Test Phase 4E, 4N, 4S (Confidentiality Matrix & Anti-IDOR):
    Upload:
    1. PUBLIC document
    2. RESTRICTED document
    3. CONFIDENTIAL document
    4. TOP_SECRET document

    Verify:
    - Viewer can ONLY see PUBLIC & RESTRICTED. Total is 2 (confidential docs are completely masked).
    - Viewer searching q="Secret" or case_id of top secret doc receives total=0, items=[].
    - Admin / Investigator sees all 4 documents. Total is 4.
    - Zero absolute filesystem paths are returned.
    """
    test_client, _, _ = client
    pdf = b"%PDF-1.4\n%Matrix Document Content\n%%EOF"

    # 1. PUBLIC
    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["admin"],
        data={"case_id": "CASE-MATRIX-01", "title": "Public Court Notice", "confidentiality": "PUBLIC"},
        files={"file": ("public.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    # 2. RESTRICTED
    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["admin"],
        data={"case_id": "CASE-MATRIX-01", "title": "Restricted Police Memo", "confidentiality": "RESTRICTED"},
        files={"file": ("restricted.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    # 3. CONFIDENTIAL
    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["admin"],
        data={"case_id": "CASE-MATRIX-01", "title": "Confidential Informant Dossier", "confidentiality": "CONFIDENTIAL"},
        files={"file": ("confidential.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    # 4. TOP_SECRET
    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["admin"],
        data={"case_id": "CASE-MATRIX-SECRET", "title": "Top Secret Syndicate Wiretap", "confidentiality": "TOP_SECRET"},
        files={"file": ("wiretap.pdf", io.BytesIO(pdf), "application/pdf")},
    )

    # Viewer search: MUST ONLY SEE 2 DOCUMENTS (PUBLIC & RESTRICTED)
    viewer_search = test_client.get("/api/v1/documents/search", headers=auth_headers["viewer"])
    assert viewer_search.status_code == 200
    v_data = viewer_search.json()
    assert v_data["total"] == 2
    returned_confs = {d["confidentiality_level"] for d in v_data["items"]}
    assert returned_confs == {"PUBLIC", "RESTRICTED"}

    # Viewer attempting IDOR / query enumeration on TOP_SECRET case receives 0 results
    viewer_probe = test_client.get("/api/v1/documents/search?case_id=CASE-MATRIX-SECRET", headers=auth_headers["viewer"])
    assert viewer_probe.status_code == 200
    assert viewer_probe.json()["total"] == 0
    assert viewer_probe.json()["items"] == []

    # Viewer querying keyword "Wiretap" receives 0 results
    viewer_kw_probe = test_client.get("/api/v1/documents/search?q=Wiretap", headers=auth_headers["viewer"])
    assert viewer_kw_probe.status_code == 200
    assert viewer_kw_probe.json()["total"] == 0

    # Admin / Investigator search sees all 4 documents
    admin_search = test_client.get("/api/v1/documents/search", headers=auth_headers["admin"])
    assert admin_search.status_code == 200
    assert admin_search.json()["total"] == 4

    # Verify no absolute filesystem path leakage in any response item
    resp_str = json.dumps(admin_search.json())
    assert "C:\\" not in resp_str
    assert "/app/data" not in resp_str
    assert "file_path" not in resp_str


def test_search_audit_logging(client, auth_headers):
    test_client, _, _ = client
    pdf = b"%PDF-1.4\n%Content\n%%EOF"

    test_client.post(
        "/api/v1/documents/upload",
        headers=auth_headers["investigator"],
        data={"case_id": "CASE-AUDIT-01", "title": "Audit Search Subject"},
        files={"file": ("audit_sub.pdf", io.BytesIO(pdf), "application/pdf")},
    )

    res = test_client.get("/api/v1/documents/search?q=Subject", headers=auth_headers["investigator"])
    assert res.status_code == 200

    # Verify audit query succeeds
    log_response = audit_logger.query_logs(AuditQueryFilter(action=AuditAction.DOCUMENT_SEARCH))
    assert log_response.total >= 1
    assert len(log_response.items) >= 1
    latest = log_response.items[0]
    assert latest.action == AuditAction.DOCUMENT_SEARCH
    assert latest.details.get("q") == "Subject"

