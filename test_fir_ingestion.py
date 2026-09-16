# -*- coding: utf-8 -*-
"""
Automated Test Suite for Upgraded FIR & Incident Report Ingestion Pipeline in NODE SENTINEL.
Tests:
- Extraction of all 8 entity categories (Person, Phone, Vehicle, BankAccount, Location, Organization, Case, Date)
- Canonical normalizations (+91 E.164 phone, vehicle plates, bank account IDs)
- Cryptographic provenance (SHA-256 file hash, case ID, char offsets, confidence score, evidence snippet)
- Responsible AI decision-support unverified status on low-confidence entities
- Ingestion API endpoints (/api/ingest/text and /api/ingest/file with TXT/PDF)
- Graph node and edge persistence with preserved provenance
"""
import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.nlp_extractor import NLPExtractor
from app.core.document_parser import DocumentParser
from app.core.graph_engine import get_graph_engine
from app.models.graph_models import NodeType, EdgeType


from app.api.routes_ingest import load_dataset_by_name


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_graph():
    g = get_graph_engine()
    g.clear()
    yield
    g.clear()
    load_dataset_by_name(g, "syndicate_network.json")


def test_nlp_canonical_normalizations():
    """Test phone, vehicle, and bank account canonical normalizations."""
    extractor = NLPExtractor()

    # Phone normalizations
    assert extractor.normalize_phone_number("9811223344") == "+919811223344"
    assert extractor.normalize_phone_number("+91 98112 23344") == "+919811223344"
    assert extractor.normalize_phone_number("09811223344") == "+919811223344"
    assert extractor.normalize_phone_number("+91-98765-00112") == "+919876500112"

    # Vehicle normalizations
    assert extractor.normalize_vehicle_registration("dl 01 ab 1234") == "DL01AB1234"
    assert extractor.normalize_vehicle_registration("UP-14-CD-5678") == "UP14CD5678"
    assert extractor.normalize_vehicle_registration("HR 26 DQ 4321") == "HR26DQ4321"

    # Bank account normalizations
    node_id, clean_acc = extractor.normalize_bank_account("ACC987654321")
    assert node_id == "ACC_987654321"
    assert clean_acc == "987654321"

    node_id2, clean_acc2 = extractor.normalize_bank_account("A/C 442099114")
    assert node_id2 == "ACC_442099114"
    assert clean_acc2 == "442099114"


def test_extraction_of_all_8_entity_categories():
    """Verify extraction of Person, Phone, Vehicle, BankAccount, Location, Organization, Case, and Date."""
    extractor = NLPExtractor()
    fir_text = (
        "On 2024-07-16, Accused Tariq Ahmad driving vehicle DL01AB9988 met Mr. Kabir Mirza "
        "at Sector 62, Noida. He used phone 9811223344 to coordinate with Axis Bank "
        "and transferred ₹500000 to ACC990188231 in case FIR-2024-NDPS-01."
    )

    entities = extractor.extract_entities(fir_text)
    labels = {e["label"] for e in entities}

    assert NodeType.PERSON.value in labels
    assert NodeType.PHONE.value in labels
    assert NodeType.VEHICLE.value in labels
    assert NodeType.BANK_ACCOUNT.value in labels
    assert NodeType.LOCATION.value in labels
    assert NodeType.ORGANIZATION.value in labels
    assert NodeType.CASE.value in labels
    assert NodeType.DATE.value in labels

    # Check canonical values
    phone_ent = next(e for e in entities if e["label"] == NodeType.PHONE.value)
    assert phone_ent["name"] == "+919811223344"

    veh_ent = next(e for e in entities if e["label"] == NodeType.VEHICLE.value)
    assert veh_ent["name"] == "DL01AB9988"

    acc_ent = next(e for e in entities if e["label"] == NodeType.BANK_ACCOUNT.value)
    assert acc_ent["name"] == "990188231"


def test_entity_and_triplet_provenance_integrity():
    """Verify cryptographic SHA-256, character spans, confidence, and snippets."""
    extractor = NLPExtractor()
    text = "Suspect Rahul Sharma called +919876543210 from Ghaziabad in FIR-2024-311."
    meta = {"filename": "fir_incident_01.txt", "case_id": "FIR-2024-311"}

    entities = extractor.extract_entities_with_provenance(text, source_metadata=meta)
    triplets = extractor.extract_triplets_with_provenance(text, entities, source_metadata=meta)

    assert len(entities) > 0
    for ent in entities:
        assert "provenance" in ent
        prov = ent["provenance"]
        assert prov["source_file"] == "fir_incident_01.txt"
        assert prov["case_id"] == "FIR-2024-311"
        assert "file_hash" in prov and len(prov["file_hash"]) >= 8
        assert "char_start" in prov and "char_end" in prov
        assert prov["char_end"] >= prov["char_start"]
        assert "confidence" in prov and 0.0 <= prov["confidence"] <= 1.0
        assert "confidence_tier" in prov
        assert "is_verified" in prov
        assert "verification_status" in prov
        assert "evidence_snippet" in prov
        assert len(prov["evidence_snippet"]) > 0

    assert len(triplets) > 0
    for trip in triplets:
        assert "provenance" in trip
        prov = trip["provenance"]
        assert prov["source_file"] == "fir_incident_01.txt"
        assert "evidence_snippet" in prov


def test_responsible_ai_decision_support_unverified_flag():
    """Verify that lower-confidence entities are clearly flagged as requiring verification."""
    extractor = NLPExtractor()
    text = "Someone mentioned an unconfirmed associate named Dave at the scene."
    entities = extractor.extract_entities_with_provenance(text)

    for ent in entities:
        prov = ent["provenance"]
        if prov["confidence"] < 0.80:
            assert prov["is_verified"] is False
            assert prov["verification_status"] == "REQUIRES_VERIFICATION"
            assert prov["confidence_tier"] == "LOW_CONFIDENCE_UNVERIFIED"


def test_document_parser_txt_support():
    """Test that DocumentParser can extract text from .txt and .log plain text streams."""
    parser = DocumentParser()
    content = "FIR No. 902/2024 Police Station Okhla. Suspect Tariq Ahmad was spotted with DL01AB1234."
    file_bytes = content.encode("utf-8")

    text = parser.extract_text_from_txt(file_bytes)
    assert "Tariq Ahmad" in text
    assert "DL01AB1234" in text


def test_ingest_text_api_endpoint(client: TestClient):
    """Test POST /api/ingest/text returns structured extracted entities, summary, created/merged nodes, and provenance."""
    payload = {
        "text": "Accused Imran Qureshi driving vehicle UP14AB1234 transferred ₹250000 to ACC987654321 on 2024-05-12 in FIR-2024-555.",
        "source_case_id": "FIR-2024-555"
    }

    resp = client.post("/api/ingest/text", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["extracted_entities_count"] > 0
    assert "summary_by_type" in data
    assert "file_hash" in data
    assert data["case_id"] == "FIR-2024-555"
    assert "created_nodes" in data
    assert "created_edges" in data
    assert "merged_entities" in data
    assert "warnings" in data
    assert len(data["created_nodes"]) >= 3

    # Verify graph persistence
    graph = get_graph_engine()
    nodes = graph.get_all_nodes()
    assert len(nodes) >= 3

    # Check vehicle node
    veh_node = graph.get_node("VEH_UP14AB1234")
    assert veh_node is not None
    assert veh_node.label == NodeType.VEHICLE


def test_ingest_txt_file_api_endpoint(client: TestClient):
    """Test uploading a .txt file to POST /api/ingest/file."""
    content = "FIR-2024-888: Officer intercepted Suspect Vikram Malhotra on phone +919811223344 near Sector 62 on 15 August 2024."
    file_obj = io.BytesIO(content.encode("utf-8"))

    resp = client.post(
        "/api/ingest/file",
        files={"file": ("incident_report.txt", file_obj, "text/plain")},
        data={"source_case_id": "FIR-2024-888"}
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["filename"] == "incident_report.txt"
    assert "Vikram Malhotra" in data["extracted_text"]
    assert data["extracted_entities_count"] >= 3
    assert "summary_by_type" in data
    assert data["file_hash"] is not None
    assert "created_nodes" in data
    assert "created_edges" in data

    # Verify entities have provenance attached
    for ent in data["entities"]:
        assert "provenance" in ent
        assert ent["provenance"]["source_file"] == "incident_report.txt"
        assert ent["provenance"]["case_id"] == "FIR-2024-888"


def test_automatic_relationship_extraction_all_target_types():
    """
    Test automated relationship extraction across all 6 core relationships:
    - Person -> CALLS -> Phone
    - Person -> OWNS -> Vehicle
    - Person -> INVOLVED_IN -> Case
    - Person -> LOCATED_AT -> Location
    - Person -> OPERATES -> Organization
    - Person -> TRANSFERRED_MONEY -> BankAccount
    """
    extractor = NLPExtractor()
    fir_text = (
        "Accused Tariq Ahmad driving vehicle DL01AB9988 was spotted in Sector 62, Noida. "
        "He called phone +919811223344 and operates Delhi Logistics. "
        "Tariq transferred ₹750000 to ACC990188231 in case FIR-2024-NDPS-01."
    )
    meta = {"filename": "fir_ndps_01.txt", "case_id": "FIR-2024-NDPS-01"}

    entities = extractor.extract_entities_with_provenance(fir_text, source_metadata=meta)
    triplets = extractor.extract_triplets_with_provenance(fir_text, entities, source_metadata=meta)

    rel_types = {t["relationship"] for t in triplets}

    # Verify presence of all expected edge relationship types
    assert EdgeType.OWNS.value in rel_types  # Person -> OWNS -> Vehicle
    assert EdgeType.LOCATED_AT.value in rel_types  # Person -> LOCATED_AT -> Location
    assert EdgeType.CALLS.value in rel_types  # Person -> CALLS -> Phone
    assert EdgeType.OPERATES.value in rel_types  # Person -> OPERATES -> Organization
    assert EdgeType.TRANSFERRED_MONEY.value in rel_types  # Person -> TRANSFERRED_MONEY -> BankAccount
    assert EdgeType.INVOLVED_IN.value in rel_types  # Person -> INVOLVED_IN -> Case

    # Verify relationship provenance completeness
    for trip in triplets:
        assert "provenance" in trip
        prov = trip["provenance"]
        assert prov["source_file"] == "fir_ndps_01.txt"
        assert prov["case_id"] == "FIR-2024-NDPS-01"
        assert "evidence_snippet" in prov and len(prov["evidence_snippet"]) > 0
        assert "char_start" in prov and "char_end" in prov
        assert prov["char_end"] >= prov["char_start"]
        assert "confidence" in prov and prov["confidence"] >= 0.80


def test_no_relationship_inferred_from_mere_document_cooccurrence():
    """
    Verify that entities appearing in separate unrelated sentences without
    syntactic/semantic relational cues do NOT generate false edges.
    """
    extractor = NLPExtractor()
    unrelated_text = (
        "Suspect Rajesh Kumar was present. "
        "The vehicle HR26DQ4321 was parked in Mumbai. "
        "Phone 9876543210 is registered in Kolkata. "
        "Axis Bank has branches in Delhi. "
        "Case FIR-2023-001 is closed."
    )
    entities = extractor.extract_entities(unrelated_text)
    triplets = extractor.extract_triplets(unrelated_text, entities)

    # Rajesh is in sentence 1, vehicle in sentence 2, phone in sentence 3, bank in sentence 4, case in sentence 5
    # No relationships should be formed between Rajesh and the vehicle, phone, bank, or case!
    rajesh_edges = [t for t in triplets if t["source"] == "PERSON_Rajesh_Kumar"]
    assert len(rajesh_edges) == 0


def test_graph_deduplication_and_merged_entities_tracking(client: TestClient):
    """
    Verify that repeated ingestion correctly categorizes created_nodes vs merged_entities.
    """
    payload_1 = {
        "text": "Accused Kabir Mirza driving vehicle DL01AB1111 met at Sector 62 in FIR-2024-101.",
        "source_case_id": "FIR-2024-101"
    }
    resp1 = client.post("/api/ingest/text", json=payload_1)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert len(data1["created_nodes"]) >= 2
    assert len(data1["merged_entities"]) == 0

    # Second ingestion with same person Kabir Mirza but new information
    payload_2 = {
        "text": "Suspect Kabir Mirza called +919811223344 and transferred ₹100000 to ACC123456789 in FIR-2024-101.",
        "source_case_id": "FIR-2024-101"
    }
    resp2 = client.post("/api/ingest/text", json=payload_2)
    assert resp2.status_code == 200
    data2 = resp2.json()

    merged_ids = [n["id"] for n in data2["merged_entities"]]
    assert "PERSON_Kabir_Mirza" in merged_ids
    assert "CASE_FIR-2024-101" in merged_ids
    assert len(data2["created_edges"]) > 0


def test_low_confidence_warnings_returned_in_api(client: TestClient):
    """
    Verify that when entities with low confidence are detected,
    warnings are returned in the response object for decision support.
    """
    payload = {
        "text": "Someone mentioned an unconfirmed associate named Dave at the scene."
    }
    resp = client.post("/api/ingest/text", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "warnings" in data
    # At least one warning for Dave's unverified / low confidence extraction
    assert any("Dave" in w or "Low-confidence" in w for w in data["warnings"])


def test_document_parser_ocr_text_normalization():
    """Verify that OCR text normalizer cleans whitespace, hyphenation, FIR formats, and phone numbers."""
    parser = DocumentParser()
    raw_ocr = (
        "FIRST  INFORMATION   REPORT\r\n\r\n"
        "Investi- \n gation into case FIR - 2024 - 902 at Sector  62.\r\n"
        "Suspect used phone + 91 9811223344 with vehicle DL01AB9988.\x00\x08"
    )
    cleaned = parser.normalize_ocr_text(raw_ocr)
    assert "FIRST INFORMATION REPORT" in cleaned
    assert "Investigation" in cleaned
    assert "FIR-2024-902" in cleaned
    assert "+91 9811223344" in cleaned
    assert "\x00" not in cleaned


def test_document_parser_direct_pdf_extraction():
    """Verify PyMuPDF direct text extraction from generated vector PDF."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    sample_text = (
        "Case FIR-2024-999: Accused Tariq Ahmad driving DL01AB9988 met Kabir Mirza on 2024-08-20. "
        "Phone: +919811223344, Bank Account: ACC990188231 at Axis Bank."
    )
    page.insert_textbox(pymupdf.Rect(50, 50, 500, 700), sample_text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()

    parser = DocumentParser()
    import asyncio
    extracted = asyncio.run(parser.extract_text_from_pdf(pdf_bytes))
    assert "Tariq Ahmad" in extracted
    assert "DL01AB9988" in extracted
    assert "ACC990188231" in extracted


def test_ingest_synthetic_fir_pdf_via_api(client: TestClient):
    """
    Test uploading the synthetic FIR PDF (sample_data/synthetic_fir_node_sentinel.pdf)
    through POST /api/ingest/file and verifying:
    - Successful 200 response
    - Extracted text contains key narrative components
    - All 8 entity categories are detected and extracted
    - Relationship triplets are formed
    - Graph persistence (nodes & edges)
    """
    import os
    pdf_path = os.path.join("sample_data", "synthetic_fir_node_sentinel.pdf")
    assert os.path.exists(pdf_path), "synthetic_fir_node_sentinel.pdf must exist"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    resp = client.post(
        "/api/ingest/file",
        files={"file": ("synthetic_fir_node_sentinel.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"source_case_id": "FIR-2024-NDPS-902"}
    )
    assert resp.status_code == 200, f"Failed: {resp.text}"
    data = resp.json()

    assert data["status"] == "success"
    assert data["filename"] == "synthetic_fir_node_sentinel.pdf"
    assert "Tariq Ahmad" in data["extracted_text"]
    assert "DL01AB9988" in data["extracted_text"]
    assert "+919811223344" in data["extracted_text"]

    # Check extracted entity count and types
    assert data["extracted_entities_count"] >= 5
    entity_labels = {e["label"] for e in data["entities"]}
    assert NodeType.PERSON.value in entity_labels
    assert NodeType.VEHICLE.value in entity_labels
    assert NodeType.PHONE.value in entity_labels
    assert NodeType.BANK_ACCOUNT.value in entity_labels

    # Check graph engine persistence
    graph = get_graph_engine()
    tariq_node = graph.get_node("PERSON_Tariq_Ahmad")
    assert tariq_node is not None, "Tariq Ahmad node should exist in graph"

    veh_node = graph.get_node("VEH_DL01AB9988")
    assert veh_node is not None, "DL01AB9988 vehicle node should exist in graph"


def test_image_upload_without_ocr_engine_returns_clear_error(client: TestClient, monkeypatch):
    """
    Verify that if an image is uploaded and OCR is unavailable,
    the API returns a clear, actionable 422 error rather than a 500 crash.
    """
    # Create a tiny 10x10 blank PNG
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    img.save(buf, format="PNG")
    buf.seek(0)
    image_bytes = buf.getvalue()

    # Temporarily simulate OCR unavailable in DocumentParser
    original_init = DocumentParser.__init__

    def mock_init(self):
        original_init(self)
        self.winocr_available = False
        self.pytesseract_available = False

    monkeypatch.setattr(DocumentParser, "__init__", mock_init)

    resp = client.post(
        "/api/ingest/file",
        files={"file": ("scanned_evidence.png", io.BytesIO(image_bytes), "image/png")},
        data={"source_case_id": "FIR-TEST-001"}
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "No OCR engine" in data["detail"] or "Cannot process image" in data["detail"]

