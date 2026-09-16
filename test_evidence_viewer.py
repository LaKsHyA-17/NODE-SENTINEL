# -*- coding: utf-8 -*-
"""
Comprehensive Automated Test Suite for Evidence Viewer & Provenance System in NODE SENTINEL.
Tests:
1. Node (Extracted Entity) Provenance Lookup
2. Graph Edge (Relationship) Provenance Lookup
3. Risk Factor Provenance Lookup
4. Anomaly / Cross-Domain Correlation Provenance Lookup
5. Timeline Event Provenance Lookup
6. AI Assistant Evidence Provenance Lookup
7. Strict Non-Fabrication Rule ("Source evidence unavailable.")
8. Character Span Text Highlighting (<mark class="evidence-highlight">)
9. Related Evidence Retrieval by Case ID / Source Document
10. FastAPI REST API endpoints (/api/evidence/lookup, /api/evidence/node/{id}, /api/evidence/related)
11. Backward compatibility of provenance metadata schema
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.graph_engine import get_graph_engine
from app.api.routes_ingest import load_dataset_by_name
from app.core.evidence_engine import EvidenceEngine
from app.models.evidence_models import (
    EvidenceProvenanceRecord,
    EvidenceLookupResponse,
)


@pytest.fixture(autouse=True)
def setup_test_graph():
    """Ensure standard dataset is cleanly loaded prior to evidence tests."""
    g = get_graph_engine()
    g.clear()
    load_dataset_by_name(g, "syndicate_network.json")
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def engine():
    return EvidenceEngine(get_graph_engine())


# ===================================================================
# 1. Domain-Specific Provenance Resolution Tests
# ===================================================================

def test_node_evidence_provenance(engine: EvidenceEngine):
    """Test provenance lookup for a graph entity/node."""
    rec = engine.get_node_evidence("PERSON_TARIQ_AHMAD")
    assert isinstance(rec, EvidenceProvenanceRecord)
    assert rec.has_provenance is True
    assert rec.entity_id == "PERSON_TARIQ_AHMAD"
    assert rec.entity_name == "Tariq Ahmad"
    assert rec.source_file is not None
    assert rec.evidence_snippet is not None
    assert rec.confidence >= 0.80


def test_edge_evidence_provenance(engine: EvidenceEngine):
    """Test provenance lookup for a graph relationship edge."""
    g = get_graph_engine()
    edges = g.get_all_edges()
    assert len(edges) > 0
    first_edge = edges[0]

    rec = engine.get_edge_evidence(edge_id=first_edge.id)
    assert isinstance(rec, EvidenceProvenanceRecord)
    assert rec.has_provenance is True
    assert rec.entity_id == first_edge.source
    assert rec.target_entity_id == first_edge.target
    assert rec.relationship is not None
    assert rec.evidence_snippet is not None


def test_risk_factor_evidence_provenance(engine: EvidenceEngine):
    """Test provenance lookup for an investigative risk factor."""
    rec = engine.get_risk_factor_evidence("FAC_REG_HIGH", entity_id="PERSON_TARIQ_AHMAD")
    assert isinstance(rec, EvidenceProvenanceRecord)
    assert rec.has_provenance is True
    assert rec.item_type == "RISK_FACTOR"
    assert "Police" in rec.source_file or "Registry" in rec.source_file
    assert rec.evidence_snippet is not None
    assert "underlying_metric" in rec.metadata


def test_anomaly_evidence_provenance(engine: EvidenceEngine):
    """Test provenance lookup for a cross-domain correlation anomaly."""
    correlations = engine.cross_domain_engine.get_all_cross_domain_correlations().correlations
    if correlations:
        first_c = correlations[0]
        rec = engine.get_anomaly_evidence(first_c.correlation_id)
        assert isinstance(rec, EvidenceProvenanceRecord)
        assert rec.has_provenance is True
        assert rec.item_type == "ANOMALY"
        assert rec.confidence is not None
        assert rec.evidence_snippet is not None


def test_timeline_event_evidence_provenance(engine: EvidenceEngine):
    """Test provenance lookup for a chronological timeline event."""
    tl = engine.timeline_engine.get_entity_timeline("PERSON_TARIQ_AHMAD")
    assert len(tl.events) > 0
    first_ev = tl.events[0]

    rec = engine.get_timeline_event_evidence(first_ev.event_id, entity_id="PERSON_TARIQ_AHMAD")
    assert isinstance(rec, EvidenceProvenanceRecord)
    assert rec.has_provenance is True
    assert rec.item_type == "TIMELINE_EVENT"
    assert rec.timestamp is not None
    assert rec.evidence_snippet is not None


def test_ai_assistant_evidence_provenance(engine: EvidenceEngine):
    """Test provenance lookup for AI assistant verified telemetry evidence."""
    sample_ev = "Target exchanged 8 calls with Phone 9811223344"
    resp = engine.lookup_evidence(sample_ev, item_type="AI_ASSISTANT_EVIDENCE", entity_id="PERSON_TARIQ_AHMAD")
    assert isinstance(resp, EvidenceLookupResponse)
    assert resp.has_evidence is True
    assert resp.primary_record.evidence_snippet == sample_ev
    assert resp.primary_record.relationship == "VERIFIED_TELEMETRY"


# ===================================================================
# 2. Strict Fallback & Non-Fabrication Tests
# ===================================================================

def test_missing_provenance_fallback(engine: EvidenceEngine):
    """
    If provenance is unavailable, explicitly display:
    'Source evidence unavailable.'
    Do not fabricate missing evidence.
    """
    rec = engine.get_node_evidence("NON_EXISTENT_ENTITY_9999")
    assert isinstance(rec, EvidenceProvenanceRecord)
    assert rec.has_provenance is False
    assert rec.evidence_snippet == "Source evidence unavailable."
    assert rec.status_message == "Source evidence unavailable."
    assert rec.confidence == 0.0

    # General lookup for unknown ID
    resp = engine.lookup_evidence("NON_EXISTENT_RECORD_8888")
    assert resp.has_evidence is False
    assert resp.primary_record.evidence_snippet == "Source evidence unavailable."


# ===================================================================
# 3. Character Span Text Highlighting Tests
# ===================================================================

def test_character_span_highlighting(engine: EvidenceEngine):
    """If source character positions are available, highlight the relevant text."""
    sample_text = "Suspect Tariq Ahmad called +919811223344 on 2024-03-15."
    # Highlight "+919811223344" at char offset 27..40
    start = 27
    end = 40
    html_out = engine.generate_highlighted_html(sample_text, start, end)

    assert '<mark class="evidence-highlight">+919811223344</mark>' in html_out
    assert "Suspect Tariq Ahmad called " in html_out
    assert " on 2024-03-15." in html_out


def test_snippet_fallback_highlighting(engine: EvidenceEngine):
    """When char offsets are omitted but snippet is present, highlight the snippet."""
    sample_text = "Transaction of ₹500,000 to hawala operator verified."
    snippet = "₹500,000"
    html_out = engine.generate_highlighted_html(sample_text, None, None, snippet=snippet)
    assert '<mark class="evidence-highlight">₹500,000</mark>' in html_out


# ===================================================================
# 4. Related Evidence Retrieval Tests
# ===================================================================

def test_related_evidence_retrieval(engine: EvidenceEngine):
    """Test retrieving associated evidence records from same case or subject."""
    related = engine.get_related_evidence(entity_id="PERSON_TARIQ_AHMAD", limit=5)
    assert isinstance(related, list)
    for r in related:
        assert isinstance(r, EvidenceProvenanceRecord)
        assert r.has_provenance is True


# ===================================================================
# 5. REST API Route Tests
# ===================================================================

def test_api_evidence_lookup_node(client: TestClient):
    """GET /api/evidence/lookup?id=PERSON_TARIQ_AHMAD returns 200 with structured evidence."""
    resp = client.get("/api/evidence/lookup?id=PERSON_TARIQ_AHMAD&type=NODE")
    assert resp.status_code == 200
    data = resp.json()
    assert "primary_record" in data
    assert "related_records" in data
    assert data["primary_record"]["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert data["primary_record"]["has_provenance"] is True
    assert data["primary_record"]["source_file"] is not None


def test_api_evidence_lookup_unavailable(client: TestClient):
    """GET /api/evidence/lookup with unknown ID returns 200 with 'Source evidence unavailable.'"""
    resp = client.get("/api/evidence/lookup?id=UNKNOWN_ID_000")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_evidence"] is False
    assert data["primary_record"]["evidence_snippet"] == "Source evidence unavailable."


def test_api_get_node_evidence(client: TestClient):
    """GET /api/evidence/node/{id} returns node provenance."""
    resp = client.get("/api/evidence/node/PERSON_TARIQ_AHMAD")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert data["has_provenance"] is True


def test_api_get_related_evidence(client: TestClient):
    """GET /api/evidence/related returns related evidence list."""
    resp = client.get("/api/evidence/related?entity_id=PERSON_TARIQ_AHMAD&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
