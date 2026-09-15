# -*- coding: utf-8 -*-
"""
Automated Test Suite for ML Anomaly Detection Engine
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.graph_engine import get_graph_engine
from app.core.ml_anomaly import get_ml_anomaly_detector
from app.api.routes_ingest import load_dataset_by_name

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_graph():
    graph = get_graph_engine()
    if len(graph.get_all_nodes()) == 0:
        load_dataset_by_name(graph, "syndicate_network.json")


def test_ml_anomaly_feature_extraction():
    detector = get_ml_anomaly_detector()
    feats = detector.extract_entity_feature_vector("PERSON_TARIQ_AHMAD")
    assert "betweenness_centrality" in feats
    assert "cdr_call_density" in feats
    assert "fin_velocity_spikes" in feats
    assert isinstance(feats["betweenness_centrality"], float)


def test_ml_anomaly_evaluation():
    detector = get_ml_anomaly_detector()
    res = detector.evaluate_entity_anomaly("PERSON_TARIQ_AHMAD")
    assert "ml_anomaly_score" in res
    assert 0.0 <= res["ml_anomaly_score"] <= 100.0
    assert "feature_attributions" in res
    assert "explanation" in res


def test_ml_anomaly_api_endpoints():
    res = client.get("/api/analytics/cross-domain/ml-anomalies?threshold=0.0")
    assert res.status_code == 200
    data = res.json()
    assert "total_anomalies" in data
    assert "anomalies" in data

    res_single = client.get("/api/analytics/cross-domain/ml-anomalies/PERSON_TARIQ_AHMAD")
    assert res_single.status_code == 200
    data_single = res_single.json()
    assert data_single["entity_id"] == "PERSON_TARIQ_AHMAD"
