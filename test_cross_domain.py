# -*- coding: utf-8 -*-
"""
Automated Test Suite for Cross-Domain Telemetry Fusion Engine
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.graph_engine import get_graph_engine
from app.core.cross_domain_engine import get_cross_domain_fusion_engine
from app.api.routes_ingest import load_dataset_by_name

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_graph():
    graph = get_graph_engine()
    if len(graph.get_all_nodes()) == 0:
        load_dataset_by_name(graph, "syndicate_network.json")


def test_cross_domain_correlations_engine():
    engine = get_cross_domain_fusion_engine()
    summary = engine.get_all_cross_domain_correlations()
    assert summary is not None
    assert summary.total_correlations >= 0
    assert "CDR_TELEPHONY" in summary.channels_monitored


def test_cross_domain_entity_profile():
    engine = get_cross_domain_fusion_engine()
    profile = engine.get_entity_cross_domain_profile("PERSON_TARIQ_AHMAD")
    assert profile.entity_id == "PERSON_TARIQ_AHMAD"
    assert profile.multi_channel_risk_index >= 0.0


def test_cross_domain_api_endpoints():
    res = client.get("/api/analytics/cross-domain/correlations")
    assert res.status_code == 200
    data = res.json()
    assert "correlations" in data
    assert "channels_monitored" in data

    res_ent = client.get("/api/analytics/cross-domain/entity/PERSON_TARIQ_AHMAD")
    assert res_ent.status_code == 200
    data_ent = res_ent.json()
    assert data_ent["entity_id"] == "PERSON_TARIQ_AHMAD"
