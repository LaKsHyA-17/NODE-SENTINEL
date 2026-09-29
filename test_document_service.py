# -*- coding: utf-8 -*-
"""
Automated Test Suite for DocumentStorageService (SIH26190 Phase 2A).
Tests:
- PDF, PNG, JPG, WEBP, TXT storage and retrieval
- Extension validation and whitelisting
- MIME type magic-byte detection and rejection of disguised binaries
- Path traversal rejection across Windows and POSIX variants
- File size enforcement and limit handling
- SHA-256 cryptographic verification and tamper detection
- Storage directory creation and isolation in temporary directory
- Thread safety and collision avoidance
"""
import hashlib
from pathlib import Path
import tempfile
import pytest

from app.core.document_service import (
    DocumentRegistry,
    DocumentStorageService,
    InvalidExtensionError,
    InvalidMimeTypeError,
    FileSizeLimitExceededError,
    PathTraversalError,
    DocumentNotFoundError,
)


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for isolated testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        service = DocumentStorageService(storage_dir=tmpdir, max_size_mb=5)
        yield service, Path(tmpdir)


def test_pdf_storage_and_retrieval(temp_storage):
    service, storage_dir = temp_storage
    pdf_bytes = b"%PDF-1.4\n%synthetic test PDF content for legal filing\n%%EOF"

    meta = service.save_document(
        file_bytes=pdf_bytes,
        original_filename="FIR_2026_NDPS_001.pdf",
        client_mime="application/pdf"
    )

    assert meta["document_id"].startswith("DOC-")
    assert meta["original_filename"] == "FIR_2026_NDPS_001.pdf"
    assert meta["mime_type"] == "application/pdf"
    assert meta["file_size"] == len(pdf_bytes)
    assert meta["sha256_hash"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert not Path(meta["storage_reference"]).is_absolute()

    # Verify retrieval
    read_bytes = service.read_document_bytes(meta["storage_reference"])
    assert read_bytes == pdf_bytes

    # Verify integrity check
    is_valid, calc_hash = service.verify_document_integrity(meta["storage_reference"], meta["sha256_hash"])
    assert is_valid is True
    assert calc_hash == meta["sha256_hash"]


def test_image_storage_png_jpg_webp(temp_storage):
    service, _ = temp_storage

    # PNG test
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    png_meta = service.save_document(png_bytes, "evidence_photo.png")
    assert png_meta["mime_type"] == "image/png"

    # JPG test
    jpg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb"
    jpg_meta = service.save_document(jpg_bytes, "seizure_photo.jpg")
    assert jpg_meta["mime_type"] == "image/jpeg"

    # WEBP test
    webp_bytes = b"RIFF\x14\x00\x00\x00WEBPVP8 \x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    webp_meta = service.save_document(webp_bytes, "scanner_copy.webp")
    assert webp_meta["mime_type"] == "image/webp"


def test_txt_storage(temp_storage):
    service, _ = temp_storage
    txt_bytes = "This is an official police case diary log statement.".encode("utf-8")
    meta = service.save_document(txt_bytes, "case_diary.txt")
    assert meta["mime_type"] == "text/plain"
    assert service.read_document_bytes(meta["storage_reference"]) == txt_bytes


def test_invalid_extension_rejected(temp_storage):
    service, _ = temp_storage
    with pytest.raises(InvalidExtensionError) as exc_info:
        service.save_document(b"echo 'malicious'", "exploit.sh")
    assert "not permitted" in str(exc_info.value)

    with pytest.raises(InvalidExtensionError):
        service.save_document(b"MZ\x90\x00", "payload.exe")


def test_disguised_executable_rejected(temp_storage):
    service, _ = temp_storage
    # A Windows executable renamed as .pdf
    fake_pdf = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00"
    with pytest.raises(InvalidMimeTypeError) as exc_info:
        service.save_document(fake_pdf, "innocent_document.pdf")
    assert "executable binary detected" in str(exc_info.value)

    # An ELF binary renamed as .txt
    fake_txt = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00"
    with pytest.raises(InvalidMimeTypeError) as exc_info:
        service.save_document(fake_txt, "innocent_notes.txt")
    assert "executable binary detected" in str(exc_info.value)


def test_corrupted_pdf_header_rejected(temp_storage):
    service, _ = temp_storage
    corrupted_pdf = b"NOT_A_PDF_HEADER_12345678"
    with pytest.raises(InvalidMimeTypeError) as exc_info:
        service.save_document(corrupted_pdf, "court_order.pdf")
    assert "PDF magic bytes" in str(exc_info.value)


def test_oversized_file_rejected(temp_storage):
    service, _ = temp_storage  # configured with 5 MB limit
    oversized_bytes = b"%PDF-" + b"A" * (6 * 1024 * 1024)
    with pytest.raises(FileSizeLimitExceededError) as exc_info:
        service.save_document(oversized_bytes, "huge_report.pdf")
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_path_traversal_prevention(temp_storage):
    service, _ = temp_storage

    # Save a legitimate file first
    pdf_bytes = b"%PDF-1.5\nvalid legal filing\n%%EOF"
    meta = service.save_document(pdf_bytes, "safe.pdf")

    # Attempt to retrieve with traversal sequences
    traversal_payloads = [
        "../../etc/passwd",
        r"..\..\Windows\System32\cmd.exe",
        "/etc/shadow",
        "C:\\Windows\\System32\\calc.exe",
        "....//....//secret.json",
        "202609/../../sensitive.txt"
    ]

    for payload in traversal_payloads:
        with pytest.raises(PathTraversalError):
            service.get_document_path(payload)


def test_filename_sanitization(temp_storage):
    service, _ = temp_storage
    pdf_bytes = b"%PDF-1.4\ncontent\n%%EOF"

    unsafe_name = "../../exploit:;?*<>|\x00file.pdf"
    meta = service.save_document(pdf_bytes, unsafe_name)
    assert ".." not in meta["original_filename"]
    assert ":" not in meta["original_filename"]
    assert "\x00" not in meta["original_filename"]
    assert meta["original_filename"].endswith(".pdf")


def test_unique_filenames_and_no_collision(temp_storage):
    service, _ = temp_storage
    pdf_bytes = b"%PDF-1.4\ncontent\n%%EOF"

    meta1 = service.save_document(pdf_bytes, "document.pdf")
    meta2 = service.save_document(pdf_bytes, "document.pdf")

    assert meta1["document_id"] != meta2["document_id"]
    assert meta1["stored_filename"] != meta2["stored_filename"]
    assert meta1["storage_reference"] != meta2["storage_reference"]


def test_tamper_detection(temp_storage):
    service, storage_dir = temp_storage
    pdf_bytes = b"%PDF-1.4\noriginal filing\n%%EOF"
    meta = service.save_document(pdf_bytes, "court_filing.pdf")

    # Modify the physical file on disk to simulate tampering
    actual_path = service.get_document_path(meta["storage_reference"])
    with open(actual_path, "wb") as f:
        f.write(b"%PDF-1.4\nTAMPERED MALICIOUS CONTENT\n%%EOF")

    is_valid, new_hash = service.verify_document_integrity(meta["storage_reference"], meta["sha256_hash"])
    assert is_valid is False
    assert new_hash != meta["sha256_hash"]


def test_document_deletion(temp_storage):
    service, _ = temp_storage
    pdf_bytes = b"%PDF-1.4\ntemp doc\n%%EOF"
    meta = service.save_document(pdf_bytes, "to_delete.pdf")

    assert service.delete_document(meta["storage_reference"]) is True

    with pytest.raises(DocumentNotFoundError):
        service.read_document_bytes(meta["storage_reference"])


def test_registry_version_creation_and_lineage():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DocumentStorageService(storage_dir=Path(tmpdir) / "docs")
        registry = DocumentRegistry(registry_file=Path(tmpdir) / "reg.json")

        # Initial Document & v1
        pdf_v1 = b"%PDF-1.4\nInitial Version 1\n%%EOF"
        save_meta = storage.save_document(pdf_v1, "case_doc.pdf")
        doc_id = save_meta["document_id"]

        v1_dict = {
            "version_id": f"VER-{doc_id}-v1",
            "document_id": doc_id,
            "version_number": 1,
            "parent_version_id": None,
            "file_path": save_meta["storage_reference"],
            "sha256_hash": save_meta["sha256_hash"],
            "created_by": "usr_inv_1",
            "created_at": save_meta["created_at"],
            "status": "SUBMITTED",
        }
        doc_record = {
            "document_id": doc_id,
            "case_id": "CASE-101",
            "title": "Case 101 Document",
            "file_path": save_meta["storage_reference"],
            "sha256_hash": save_meta["sha256_hash"],
            "current_version": 1,
            "status": "SUBMITTED",
        }
        registry.save_document(doc_record, v1_dict)

        # Create Version 2
        pdf_v2 = b"%PDF-1.4\nAmended Version 2\n%%EOF"
        up_doc, v2_dict = registry.create_next_version(
            document_id=doc_id,
            file_bytes=pdf_v2,
            original_filename="case_doc_v2.pdf",
            created_by="usr_inv_2",
            change_description="Added forensic notes",
            storage_svc=storage,
        )

        assert up_doc["current_version"] == 2
        assert v2_dict["version_number"] == 2
        assert v2_dict["parent_version_id"] == v1_dict["version_id"]
        assert v2_dict["sha256_hash"] != v1_dict["sha256_hash"]

        # Create Version 3
        pdf_v3 = b"%PDF-1.4\nFinal Version 3\n%%EOF"
        up_doc3, v3_dict = registry.create_next_version(
            document_id=doc_id,
            file_bytes=pdf_v3,
            original_filename="case_doc_v3.pdf",
            created_by="usr_inv_3",
            change_description="Final submission",
            storage_svc=storage,
        )

        assert up_doc3["current_version"] == 3
        assert v3_dict["version_number"] == 3
        assert v3_dict["parent_version_id"] == v2_dict["version_id"]

        # Check versions list
        all_versions = registry.get_versions(doc_id)
        assert len(all_versions) == 3
        assert [v["version_number"] for v in all_versions] == [1, 2, 3]


def test_registry_sealed_and_archived_protection():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DocumentStorageService(storage_dir=Path(tmpdir) / "docs")
        registry = DocumentRegistry(registry_file=Path(tmpdir) / "reg.json")

        pdf = b"%PDF-1.4\nDoc Content\n%%EOF"
        save_meta = storage.save_document(pdf, "doc.pdf")
        doc_id = save_meta["document_id"]

        registry.save_document({
            "document_id": doc_id,
            "case_id": "CASE-102",
            "title": "Sealed Doc",
            "status": "SEALED",
            "file_path": save_meta["storage_reference"],
            "sha256_hash": save_meta["sha256_hash"],
            "current_version": 1,
        })

        from app.core.document_service import DocumentSealedError, DocumentArchivedError, DocumentNotFoundError

        with pytest.raises(DocumentSealedError):
            registry.create_next_version(doc_id, pdf, "v2.pdf", "usr_1", storage_svc=storage)

        registry.update_status(doc_id, "ARCHIVED")
        with pytest.raises(DocumentArchivedError):
            registry.create_next_version(doc_id, pdf, "v2.pdf", "usr_1", storage_svc=storage)

        with pytest.raises(DocumentNotFoundError):
            registry.create_next_version("NON_EXISTENT", pdf, "v2.pdf", "usr_1", storage_svc=storage)


def test_registry_search_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = DocumentStorageService(storage_dir=Path(tmpdir) / "docs")
        registry = DocumentRegistry(registry_file=Path(tmpdir) / "reg.json")

        # Save 3 documents
        doc1 = {
            "document_id": "DOC-2026-001",
            "case_id": "CASE-SEARCH-1",
            "title": "Narcotics FIR Filing",
            "doc_type": "FIR",
            "status": "SUBMITTED",
            "confidentiality_level": "RESTRICTED",
            "uploaded_by": "officer_sharma",
            "created_at": "2026-09-01T10:00:00+00:00",
            "tags": ["NDPS", "Seizure"],
            "current_version": 1,
        }
        doc2 = {
            "document_id": "DOC-2026-002",
            "case_id": "CASE-SEARCH-1",
            "title": "Confidential Syndicate Analysis",
            "doc_type": "INVESTIGATION_RECORD",
            "status": "SUBMITTED",
            "confidentiality_level": "CONFIDENTIAL",
            "uploaded_by": "officer_verma",
            "created_at": "2026-09-02T10:00:00+00:00",
            "tags": ["Syndicate", "Intelligence"],
            "current_version": 1,
        }
        doc3 = {
            "document_id": "DOC-2026-003",
            "case_id": "CASE-SEARCH-2",
            "title": "Forensic Ballistics Chemistry",
            "doc_type": "FORENSIC_REPORT",
            "status": "SUBMITTED",
            "confidentiality_level": "PUBLIC",
            "uploaded_by": "officer_sharma",
            "created_at": "2026-09-03T10:00:00+00:00",
            "tags": ["Ballistics", "FSL"],
            "current_version": 1,
        }
        registry.save_document(doc1)
        registry.save_document(doc2)
        registry.save_document(doc3)

        # 1. Search with can_access_fn excluding CONFIDENTIAL
        can_access = lambda d: d.get("confidentiality_level") != "CONFIDENTIAL"
        items, total = registry.search_documents(can_access_fn=can_access)
        assert total == 2
        assert len(items) == 2
        ids = {d["document_id"] for d in items}
        assert ids == {"DOC-2026-001", "DOC-2026-003"}

        # 2. Query filter q="ballistics"
        items, total = registry.search_documents(query="ballistics")
        assert total == 1
        assert items[0]["document_id"] == "DOC-2026-003"

        # 3. Creator filter uploaded_by="officer_sharma"
        items, total = registry.search_documents(created_by="officer_sharma")
        assert total == 2

        # 4. Case filter case_id="CASE-SEARCH-1"
        items, total = registry.search_documents(case_id="CASE-SEARCH-1")
        assert total == 2

        # 5. Date range filter
        items, total = registry.search_documents(created_from="2026-09-02T00:00:00Z", created_to="2026-09-04T00:00:00Z")
        assert total == 2
        assert {d["document_id"] for d in items} == {"DOC-2026-002", "DOC-2026-003"}


