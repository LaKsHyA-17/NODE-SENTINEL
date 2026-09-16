# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for STEP 12: Explainable Risk & Anomaly Intelligence for NODE SENTINEL.
Tests:
- Risk factor generation across categories (REGISTRY, GRAPH, CDR, FINANCIAL, CASE, LOCATION)
- Score calculation and mathematical reproducibility
- Threshold classifications (0–24 LOW, 25–49 MODERATE, 50–74 ELEVATED, 75–100 HIGH)
- CDR communication burst and volume factors
- Financial velocity and high-value transfer factors
- Graph centrality broker and degree factors
- Case linkage and statutory offense severity factors
- Location co-location factors
- Evidence traceability (every factor has source, evidence string, and contextual narrative)
- Neutral decision-support language verification (no guilt implication)
- Unknown entity handling (returns valid low-risk object without crashing)
- Empty/no-evidence entity handling
- API response via FastAPI TestClient (GET /api/risk/{entity_id})
- Integration with Timeline and Node Dossier
- Preservation of existing CDR, Financial, Timeline, and Graph functionality
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.graph_engine import get_graph_engine
from app.api.routes_ingest import load_dataset_by_name
from app.core.anomaly_detector import AnomalyDetector
from app.models.risk_models import (
    RiskCategory,
    RiskFactor,
    RiskIntelligenceResult,
    RiskSeverity,
)


@pytest.fixture(autouse=True)
def setup_test_graph():
    """Ensure standard demo graph is cleanly loaded prior to risk intelligence tests."""
    g = get_graph_engine()
    g.clear()
    load_dataset_by_name(g, "syndicate_network.json")
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def detector():
    return AnomalyDetector(get_graph_engine())


# ===================================================================
# 1. Risk Factor Generation & Score Reproducibility Tests
# ===================================================================

def test_risk_factor_generation_high_risk_subject(detector: AnomalyDetector):
    """Test full risk factor generation for a high-risk entity (Tariq Ahmad)."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    assert isinstance(res, RiskIntelligenceResult)
    assert res.entity_id == "PERSON_TARIQ_AHMAD"
    assert res.entity_name == "Tariq Ahmad"
    assert res.risk_score >= 75.0
    assert res.risk_level == "HIGH"
    assert len(res.factors) >= 4
    assert len(res.evidence_summary) == len(res.factors)


def test_score_reproducibility(detector: AnomalyDetector):
    """The final score must be reproducible from the bounded sum of individual factor contributions."""
    for eid in ["PERSON_TARIQ_AHMAD", "PERSON_POOJA_RATHI", "PERSON_SURESH_RAWAT", "PERSON_ANITA_DESHMUKH"]:
        res = detector.calculate_investigative_risk_score(eid)
        raw_sum = sum(f.score_contribution for f in res.factors)
        expected_bounded = min(100.0, max(0.0, round(raw_sum, 1)))
        assert res.risk_score == expected_bounded, (
            f"Score {res.risk_score} does not match factor sum {expected_bounded} for {eid}"
        )


def test_backward_compatibility_aliases(detector: AnomalyDetector):
    """Ensure overall_score, severity_level, and factor/points fields exist for legacy callers."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    assert res.overall_score == res.risk_score
    assert res.severity_level == res.risk_level
    for f in res.factors:
        assert hasattr(f, "factor")
        assert f.factor is not None
        assert hasattr(f, "points")
        assert f.points == f.score_contribution


# ===================================================================
# 2. Threshold Classification Tests
# ===================================================================

def test_threshold_classification_spectrum(detector: AnomalyDetector):
    """Verify that different network entities map correctly across the 0-100 threshold spectrum."""
    # Tariq Ahmad (Syndicate Kingpin, betweenness 0.60, linked cases, high risk tag) -> HIGH
    tariq = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    assert tariq.risk_score >= 75.0
    assert tariq.risk_level == "HIGH"

    # Anita Deshmukh (LOW risk tag, ordinary degree, no severe cases) -> LOW
    anita = detector.calculate_investigative_risk_score("PERSON_ANITA_DESHMUKH")
    assert anita.risk_score < 25.0
    assert anita.risk_level == "LOW"

    # Rohit Saxena (MEDIUM risk tag) -> MODERATE
    rohit = detector.calculate_investigative_risk_score("PERSON_ROHIT_SAXENA")
    assert 25.0 <= rohit.risk_score <= 49.9
    assert rohit.risk_level == "MODERATE"


# ===================================================================
# 3. Individual Factor Source & Evidence Tests
# ===================================================================

def test_registry_factor(detector: AnomalyDetector):
    """Test registry factor extraction from node properties."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    reg_factors = [f for f in res.factors if f.category == RiskCategory.REGISTRY]
    assert len(reg_factors) >= 1
    rf = reg_factors[0]
    assert rf.source == "Police FIR Registry"
    assert rf.score_contribution in (15.0, 35.0)
    assert "Registry" in rf.evidence or "Suspect" in rf.evidence


def test_graph_centrality_factor(detector: AnomalyDetector):
    """Test betweenness or degree centrality factor generation."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    graph_factors = [f for f in res.factors if f.category == RiskCategory.GRAPH]
    assert len(graph_factors) >= 1
    gf = graph_factors[0]
    assert gf.source == "Graph Analytics"
    assert "centrality" in gf.evidence.lower()
    assert gf.score_contribution > 0


def test_cdr_communication_factor(detector: AnomalyDetector):
    """Test communication factor extraction."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    cdr_factors = [f for f in res.factors if f.category == RiskCategory.CDR]
    assert len(cdr_factors) >= 1
    cf = cdr_factors[0]
    assert cf.source == "CDR Analysis"
    assert cf.timeline_event_type == "CALL"
    assert "CDR" in cf.action_hint


def test_financial_factor(detector: AnomalyDetector):
    """Test financial transaction factor extraction."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    fin_factors = [f for f in res.factors if f.category == RiskCategory.FINANCIAL]
    assert len(fin_factors) >= 1
    ff = fin_factors[0]
    assert ff.source == "Financial Analysis"
    assert ff.timeline_event_type == "FINANCIAL"
    assert "Financial" in ff.action_hint or "Transactions" in ff.action_hint


def test_case_linkage_factor(detector: AnomalyDetector):
    """Test case connection factor generation."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    case_factors = [f for f in res.factors if f.category == RiskCategory.CASE]
    assert len(case_factors) >= 1
    cf = case_factors[0]
    assert cf.source == "Case Data"
    assert cf.timeline_event_type == "CASE"
    assert "case" in cf.evidence.lower() or "fir" in cf.evidence.lower()


# ===================================================================
# 4. Evidence Traceability & AI Decision-Support Language Tests
# ===================================================================

def test_evidence_traceability(detector: AnomalyDetector):
    """Every risk factor must identify its source, evidence, and explanation."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    for f in res.factors:
        assert f.factor_id != ""
        assert f.source != ""
        assert f.evidence != ""
        assert f.explanation != ""
        assert f.score_contribution > 0


def test_neutral_decision_support_language(detector: AnomalyDetector):
    """Language must emphasize decision support and avoid declaring guilt."""
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    assert "Decision-Support Signal" in res.conclusion or "Does not prove criminal activity" in res.conclusion
    for f in res.factors:
        # Must not declare guilt
        assert "is guilty" not in f.explanation.lower()
        assert "is a criminal" not in f.explanation.lower()
        assert "proves crime" not in f.explanation.lower()


# ===================================================================
# 5. Edge Cases & Error Tolerance
# ===================================================================

def test_unknown_entity_handling(detector: AnomalyDetector):
    """Unknown entity returns a valid LOW risk response without crashing."""
    res = detector.calculate_investigative_risk_score("NON_EXISTENT_ENTITY_9999")
    assert isinstance(res, RiskIntelligenceResult)
    assert res.risk_score == 0.0
    assert res.risk_level == "LOW"
    assert len(res.factors) == 0
    assert "Decision-Support Signal" in res.conclusion or "Does not prove" in res.conclusion


def test_no_evidence_entity(detector: AnomalyDetector):
    """Entity with no negative factors returns clean 0 score."""
    res = detector.calculate_investigative_risk_score("VEH_DL01AB9988")
    assert res.risk_score == 0.0
    assert res.risk_level == "LOW"
    assert len(res.factors) == 0


# ===================================================================
# 6. API Route Tests (FastAPI TestClient)
# ===================================================================

def test_api_get_risk_success(client: TestClient):
    """GET /api/risk/{entity_id} returns 200 with structured risk intelligence."""
    resp = client.get("/api/risk/PERSON_TARIQ_AHMAD")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert "risk_score" in data
    assert "risk_level" in data
    assert "factors" in data
    assert "evidence_summary" in data
    assert "conclusion" in data
    assert "thresholds" in data
    assert len(data["factors"]) > 0


def test_api_get_risk_blank_id(client: TestClient):
    """GET /api/risk/ with whitespace ID returns 400 Bad Request."""
    resp = client.get("/api/risk/%20%20")
    assert resp.status_code in (400, 404)


def test_api_get_risk_unknown_entity(client: TestClient):
    """GET /api/risk/{unknown} returns 200 with 0 risk score."""
    resp = client.get("/api/risk/UNKNOWN_PERSON_000")
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_score"] == 0.0
    assert data["risk_level"] == "LOW"
    assert data["factors"] == []


# ===================================================================
# 7. Integration with Graph & Timeline Endpoints
# ===================================================================

def test_graph_dossier_contains_risk_intelligence(client: TestClient):
    """Entity dossier endpoint /api/entity/{id}/dossier includes risk factors."""
    resp = client.get("/api/entity/PERSON_TARIQ_AHMAD/dossier")
    assert resp.status_code == 200
    data = resp.json()
    assert "risk" in data
    risk = data["risk"]
    assert "risk_score" in risk or "overall_score" in risk
    assert "factors" in risk
    assert len(risk["factors"]) > 0


def test_timeline_integration_with_risk_entity(client: TestClient):
    """Entity with high risk intelligence has accessible timeline events."""
    risk_resp = client.get("/api/risk/PERSON_TARIQ_AHMAD")
    assert risk_resp.status_code == 200
    assert risk_resp.json()["risk_score"] >= 75.0

    timeline_resp = client.get("/api/timeline/PERSON_TARIQ_AHMAD")
    assert timeline_resp.status_code == 200
    assert timeline_resp.json()["total_events"] > 0


# ===================================================================
# 8. Explainable Breakdown & Regression Tests
# ===================================================================

def test_explainable_breakdown_all_required_fields(detector: AnomalyDetector):
    """
    Regression Test: For every anomaly/risk result, expose an explainable breakdown:
    - total score
    - severity
    - individual contributing factors
    - numerical contribution
    - underlying metric
    - source
    - evidence
    - timestamp where available
    - involved entities
    """
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    assert res.total_score is not None
    assert res.total_score == res.risk_score
    assert res.severity is not None
    assert res.severity == res.risk_level
    assert len(res.factors) > 0

    for factor in res.factors:
        # 1. Numerical contribution
        assert factor.numerical_contribution is not None
        assert factor.numerical_contribution == factor.score_contribution
        assert factor.numerical_contribution > 0

        # 2. Underlying metric
        assert factor.underlying_metric is not None
        assert len(factor.underlying_metric.strip()) > 0

        # 3. Source
        assert factor.source is not None
        assert len(factor.source.strip()) > 0

        # 4. Evidence
        assert factor.evidence is not None
        assert len(factor.evidence.strip()) > 0

        # 5. Involved entities
        assert isinstance(factor.involved_entities, list)
        assert len(factor.involved_entities) >= 1

        # 6. Action type & hint for UI navigation
        assert factor.action_hint is not None
        assert factor.action_type is not None


def test_cdr_communication_burst_factor_breakdown(detector: AnomalyDetector):
    """
    Regression Test:
    Example:
    'Communication burst'
    → 8 calls in 24 hours
    → source: CDR file
    → involved entities: Person A, Person B
    → [View CDR]
    """
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    cdr_factors = [f for f in res.factors if f.category == RiskCategory.CDR]
    assert len(cdr_factors) >= 1

    cf = cdr_factors[0]
    assert "cdr" in cf.source.lower()
    assert cf.underlying_metric is not None
    assert "call" in cf.underlying_metric.lower() or "duration" in cf.underlying_metric.lower()
    assert len(cf.involved_entities) >= 1
    assert cf.action_type == "VIEW_CDR"
    assert "CDR" in cf.action_hint


def test_financial_velocity_factor_breakdown(detector: AnomalyDetector):
    """
    Regression Test:
    Example:
    'Financial velocity'
    → 4 transfers in 120 minutes / high-value volume
    → source: financial ledger
    → [View Transactions]
    """
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    fin_factors = [f for f in res.factors if f.category == RiskCategory.FINANCIAL]
    assert len(fin_factors) >= 1

    ff = fin_factors[0]
    assert "financ" in ff.source.lower()
    assert ff.underlying_metric is not None
    assert "₹" in ff.underlying_metric or "transfer" in ff.underlying_metric.lower() or "transaction" in ff.underlying_metric.lower()
    assert len(ff.involved_entities) >= 1
    assert ff.action_type == "VIEW_FINANCE"
    assert "Financial" in ff.action_hint or "Transaction" in ff.action_hint


def test_strict_decision_support_language_compliance(detector: AnomalyDetector):
    """
    Regression Test:
    Never say the score proves criminal activity.
    Use:
    - 'Investigative risk indicator'
    - 'Anomalous pattern'
    - 'Decision-support signal'
    """
    res = detector.calculate_investigative_risk_score("PERSON_TARIQ_AHMAD")
    
    # 1. Result conclusion & disclaimer
    assert "proves criminal activity" not in res.conclusion.lower() or "does not prove" in res.conclusion.lower()
    assert "Decision-Support Signal" in res.conclusion or "Investigative risk indicator" in res.conclusion
    assert "Decision-Support Signal" in res.disclaimer
    assert "Does not prove criminal activity" in res.disclaimer or "do not prove criminal activity" in res.disclaimer

    # 2. Factor titles & explanations
    for f in res.factors:
        assert "is guilty" not in f.explanation.lower()
        assert "proven criminal" not in f.explanation.lower()
        assert "proves crime" not in f.explanation.lower()
        # Verify factor categories are aligned
        assert f.category in [RiskCategory.REGISTRY, RiskCategory.GRAPH, RiskCategory.CDR, RiskCategory.FINANCIAL, RiskCategory.CASE, RiskCategory.LOCATION, RiskCategory.BEHAVIORAL]


def test_api_returns_explainable_breakdown_contract(client: TestClient):
    """
    Regression Test: Verify GET /api/risk/{entity_id} contract returns all 8 required explainable fields.
    """
    resp = client.get("/api/risk/PERSON_TARIQ_AHMAD")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_score" in data
    assert "severity" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert "factors" in data
    assert "disclaimer" in data

    for factor in data["factors"]:
        assert "numerical_contribution" in factor
        assert "underlying_metric" in factor
        assert "source" in factor
        assert "evidence" in factor
        assert "involved_entities" in factor
        assert "action_hint" in factor
        assert "action_type" in factor
        assert "confidence" in factor
