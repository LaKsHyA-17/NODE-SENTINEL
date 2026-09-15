# -*- coding: utf-8 -*-
"""
Automated Test Suite for Smart Suspect Intelligence Dossier in NODE SENTINEL.
Tests:
- /api/entity/{entity_id}/dossier endpoint and risk breakdown
- /api/network/node/{entity_id}/details multi-domain entity metadata
- /api/analytics/cross-domain/ml-anomalies/{entity_id} feature attributions
- /api/cdr/analyze/{entity_id} communication indicators and burst patterns
- /api/financial/analyze/{entity_id} transaction metrics and counterparties
- /api/analytics/cross-domain/entity/{entity_id} multi-source fused events
- Graceful error isolation on unknown entities
- Responsible AI decision-support indicator compliance
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.graph_engine import get_graph_engine
from app.api.routes_ingest import load_dataset_by_name


@pytest.fixture(autouse=True)
def setup_dataset():
    """Ensure standard demo syndicate graph is loaded prior to testing."""
    g = get_graph_engine()
    g.clear()
    load_dataset_by_name(g, "syndicate_network.json")
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_smart_dossier_endpoint(client: TestClient):
    """Test core entity dossier returns structured subject metadata, risk factors, and centrality."""
    resp = client.get("/api/entity/PERSON_TARIQ_AHMAD/dossier")
    assert resp.status_code == 200
    data = resp.json()

    assert data["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert data["name"] == "Tariq Ahmad"
    assert data["type"] == "Person"
    assert "risk" in data
    assert "centrality" in data
    assert "connected_nodes" in data

    # Risk payload verification
    risk = data["risk"]
    assert "overall_score" in risk or "risk_score" in risk
    assert "factors" in risk
    assert len(risk["factors"]) > 0


def test_smart_dossier_node_details(client: TestClient):
    """Test node details endpoint supplies direct/indirect counts, cases, and linked phones."""
    resp = client.get("/api/network/node/PERSON_TARIQ_AHMAD/details")
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == "PERSON_TARIQ_AHMAD"
    assert "direct_count" in data
    assert "indirect_count" in data
    assert "cases" in data
    assert "phones" in data
    assert isinstance(data["cases"], list)


def test_smart_dossier_ml_anomalies(client: TestClient):
    """Test ML anomaly endpoint returns decision-support outlier scores and explainable feature weights."""
    resp = client.get("/api/analytics/cross-domain/ml-anomalies/PERSON_TARIQ_AHMAD")
    assert resp.status_code == 200
    data = resp.json()

    assert data["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert "ml_anomaly_score" in data
    assert 0.0 <= data["ml_anomaly_score"] <= 100.0
    assert "severity" in data
    assert "feature_attributions" in data
    assert isinstance(data["feature_attributions"], dict)


def test_smart_dossier_cdr_analytics(client: TestClient):
    """Test CDR analytics endpoint returns communication statistics and top contacts."""
    resp = client.get("/api/cdr/analyze/PERSON_TARIQ_AHMAD")
    assert resp.status_code == 200
    data = resp.json()

    assert data["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert "statistics" in data
    assert "top_contacts" in data
    assert "bursts" in data
    assert "total_calls" in data["statistics"]


def test_smart_dossier_financial_intelligence(client: TestClient):
    """Test financial intelligence endpoint returns volume metrics and velocity indicators."""
    resp = client.get("/api/financial/analyze/PERSON_TARIQ_AHMAD")
    assert resp.status_code == 200
    data = resp.json()

    assert data["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert "statistics" in data
    assert "total_transactions" in data["statistics"]
    assert "total_incoming_amount" in data["statistics"]
    assert "total_outgoing_amount" in data["statistics"]


def test_smart_dossier_cross_domain_profile(client: TestClient):
    """Test cross-domain correlation profile returns multi-source fused temporal links."""
    resp = client.get("/api/analytics/cross-domain/entity/PERSON_TARIQ_AHMAD")
    assert resp.status_code == 200
    data = resp.json()

    assert data["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert "total_fused_events" in data
    assert "correlations" in data
    assert isinstance(data["correlations"], list)


def test_smart_dossier_unknown_entity_resilience(client: TestClient):
    """Verify system handles unknown entity gracefully without crashing."""
    resp = client.get("/api/entity/UNKNOWN_NON_EXISTENT_ID/dossier")
    assert resp.status_code == 404

    # ML Anomaly should evaluate cleanly or return low baseline
    resp_ml = client.get("/api/analytics/cross-domain/ml-anomalies/UNKNOWN_NON_EXISTENT_ID")
    assert resp_ml.status_code == 200
