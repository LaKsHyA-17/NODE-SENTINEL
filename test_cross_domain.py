# -*- coding: utf-8 -*-
"""
Automated Test Suite for Cross-Domain Intelligence & Multi-Source Fusion Engine
Tests:
1. System-wide cross-domain fusion across all monitored channels (CDR, Bank, FIR, Graph, Location, Biometrics).
2. Presence and integrity of all 9 required correlation attributes on every correlation item:
   - Signal type
   - Entities involved
   - Timestamp/date
   - Source
   - Confidence
   - Evidence snippet
   - Graph relationship
   - Link to timeline
   - Link to source evidence
3. Entity-specific cross-domain profile resolution (e.g. PERSON_TARIQ_AHMAD).
4. Case-specific cross-domain profile resolution (e.g. CASE_FIR_2024_405).
5. Strict neutral decision-support language compliance (no unfounded causal claims).
6. Multi-domain signal detection: CDR + Finance, Location Hotspots, FIR Records, Face Alignment, Graph Topology.
7. API endpoints validation (/correlations, /entity/{id}, /case/{id}).
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
    """Verify system-wide multi-source cross-domain correlation engine execution."""
    engine = get_cross_domain_fusion_engine()
    summary = engine.get_all_cross_domain_correlations()
    assert summary is not None
    assert summary.total_correlations >= 1
    assert len(summary.correlations) == summary.total_correlations
    assert "CDR_TELEPHONY" in summary.channels_monitored
    assert "POLICE_FIR_REGISTRY" in summary.channels_monitored
    assert "BANK_LEDGER" in summary.channels_monitored
    assert "LOCATION_SURVEILLANCE" in summary.channels_monitored
    assert "BIOMETRIC_FACE_DATABASE" in summary.channels_monitored
    assert "KNOWLEDGE_GRAPH" in summary.channels_monitored


def test_cross_domain_9_attributes_integrity():
    """
    Verify that EVERY correlation item contains all 9 required attributes:
    1. Signal type
    2. Entities involved
    3. Timestamp/date
    4. Source
    5. Confidence
    6. Evidence snippet
    7. Graph relationship
    8. Link to timeline
    9. Link to source evidence
    """
    engine = get_cross_domain_fusion_engine()
    summary = engine.get_all_cross_domain_correlations()

    for item in summary.correlations:
        # 1. Signal type
        assert item.signal_type is not None and len(item.signal_type) > 0, f"Missing signal_type in {item.correlation_id}"

        # 2. Entities involved
        assert isinstance(item.entities_involved, list) and len(item.entities_involved) > 0, f"Missing entities_involved in {item.correlation_id}"

        # 3. Timestamp / date
        assert item.timestamp is not None or item.first_timestamp is not None, f"Missing timestamp in {item.correlation_id}"

        # 4. Source
        assert item.source is not None and len(item.source) > 0, f"Missing source in {item.correlation_id}"

        # 5. Confidence
        assert 0.0 <= item.confidence_score <= 1.0, f"Invalid confidence_score in {item.correlation_id}"
        assert item.confidence is not None

        # 6. Evidence snippet
        assert item.evidence_snippet is not None and len(item.evidence_snippet) > 0, f"Missing evidence_snippet in {item.correlation_id}"

        # 7. Graph relationship
        assert item.graph_relationship is not None and len(item.graph_relationship) > 0, f"Missing graph_relationship in {item.correlation_id}"

        # 8. Link to timeline
        assert item.timeline_link is not None and isinstance(item.timeline_link, dict), f"Missing timeline_link in {item.correlation_id}"
        assert "entity_id" in item.timeline_link

        # 9. Link to source evidence
        assert item.source_evidence_link is not None and isinstance(item.source_evidence_link, dict), f"Missing source_evidence_link in {item.correlation_id}"


def test_neutral_decision_support_language_compliance():
    """
    Ensure all correlation descriptions and titles use strictly neutral, non-causal language.
    E.g. 'correlated activity', 'observed relationship', 'potentially relevant pattern', 'co-occurring'.
    Must NOT contain speculative causal assertions like 'caused', 'guilty', 'orchestrated crime because'.
    """
    engine = get_cross_domain_fusion_engine()
    summary = engine.get_all_cross_domain_correlations()

    prohibited_speculative_words = ["is guilty of", "caused the crime", "perpetrated murder because", "definitely proves crime"]

    for item in summary.correlations:
        combined_text = f"{item.title} {item.description} {item.evidence_snippet}".lower()
        for bad_phrase in prohibited_speculative_words:
            assert bad_phrase not in combined_text, f"Unfounded causal language '{bad_phrase}' found in {item.correlation_id}"

        # Must contain neutral indicator phrasing
        has_neutral = any(
            phrase in combined_text
            for phrase in [
                "correlated activity",
                "observed relationship",
                "potentially relevant pattern",
                "co-presence",
                "co-involvement",
                "co-occurring",
                "observed",
                "documents",
                "aligns with",
                "linked",
                "telemetry",
            ]
        )
        assert has_neutral, f"Expected neutral decision support phrasing in {item.correlation_id}"


def test_cdr_plus_financial_coordination_signal():
    """Verify CDR + Financial rapid transfer correlation detection."""
    engine = get_cross_domain_fusion_engine()
    c2t = engine.detect_call_to_transfer_correlations(max_latency_minutes=120.0)

    # In syndicate_network.json, calls and transfers exist
    if len(c2t) > 0:
        first = c2t[0]
        assert first.signal_type == "CDR + Financial Coordination"
        assert "CDR_TELEPHONY" in first.evidence_channels
        assert "BANK_LEDGER" in first.evidence_channels
        assert "CALLS → TRANSFERRED_MONEY" in first.graph_relationship
        assert first.timeline_link is not None
        assert "₹" in first.evidence_snippet or "transfer" in first.evidence_snippet.lower()


def test_fir_case_association_signal():
    """Verify Police FIR Case record multi-entity association detection."""
    engine = get_cross_domain_fusion_engine()
    case_corrs = engine.detect_fir_case_correlations()
    assert len(case_corrs) >= 1

    first = case_corrs[0]
    assert first.signal_type == "FIR Case Association"
    assert "POLICE_FIR_REGISTRY" in first.evidence_channels
    assert "INVOLVED_IN" in first.graph_relationship
    assert "Police FIR / Case" in first.evidence_snippet


def test_location_hotspot_signal():
    """Verify Location surveillance co-presence hotspot signal detection."""
    engine = get_cross_domain_fusion_engine()
    loc_corrs = engine.detect_cotemporal_colocation_correlations()
    assert len(loc_corrs) >= 1

    first = loc_corrs[0]
    assert first.signal_type == "Location Surveillance Hotspot"
    assert "LOCATION_SURVEILLANCE" in first.evidence_channels
    assert "LOCATED_AT" in first.graph_relationship


def test_cross_domain_entity_profile():
    """Verify entity-specific cross-domain profile aggregation."""
    engine = get_cross_domain_fusion_engine()
    profile = engine.get_entity_cross_domain_profile("PERSON_TARIQ_AHMAD")
    assert profile.entity_id == "PERSON_TARIQ_AHMAD"
    assert profile.total_fused_events >= 1
    assert len(profile.correlations) == profile.total_fused_events
    assert profile.multi_channel_risk_index >= 0.0
    assert len(profile.associated_channels) >= 1


def test_cross_domain_case_profile():
    """Verify case-specific cross-domain profile aggregation."""
    engine = get_cross_domain_fusion_engine()
    # In syndicate_network.json, CASE_FIR_2024_405 is present
    case_profile = engine.get_case_cross_domain_profile("CASE_FIR_2024_405")
    assert case_profile.case_id == "CASE_FIR_2024_405"
    assert isinstance(case_profile.involved_entities, list)
    assert case_profile.total_fused_events >= 0


def test_cross_domain_api_endpoints():
    """Verify API routes for system-wide, entity-specific, and case-specific queries."""
    # 1. System correlations
    res = client.get("/api/analytics/cross-domain/correlations")
    assert res.status_code == 200
    data = res.json()
    assert "correlations" in data
    assert "channels_monitored" in data
    assert "disclaimer" in data

    # 2. Entity profile
    res_ent = client.get("/api/analytics/cross-domain/entity/PERSON_TARIQ_AHMAD")
    assert res_ent.status_code == 200
    data_ent = res_ent.json()
    assert data_ent["entity_id"] == "PERSON_TARIQ_AHMAD"
    assert "correlations" in data_ent

    # 3. Case profile
    res_case = client.get("/api/analytics/cross-domain/case/CASE_FIR_2024_405")
    assert res_case.status_code == 200
    data_case = res_case.json()
    assert data_case["case_id"] == "CASE_FIR_2024_405"
    assert "correlations" in data_case

    # 4. Blank input resilience
    res_blank = client.get("/api/analytics/cross-domain/entity/%20")
    assert res_blank.status_code == 400


def test_hidden_link_shared_phone_detection():
    """Verify discovery of subjects connected through shared phone numbers."""
    engine = get_cross_domain_fusion_engine()
    phone_links = engine.detect_shared_phone_correlations()
    assert isinstance(phone_links, list)
    for link in phone_links:
        assert link.correlation_type == "SHARED_PHONE"
        assert link.signal_type == "Hidden Link: Shared Phone Lead"
        assert len(link.entities_involved) >= 2
        assert link.details.get("provenance_nature") == "INFERRED_CONNECTION"
        assert "shared phone" in link.evidence_snippet.lower() or "telephony" in link.evidence_snippet.lower()


def test_hidden_link_shared_bank_account_detection():
    """Verify discovery of subjects connected through shared bank accounts."""
    engine = get_cross_domain_fusion_engine()
    acc_links = engine.detect_shared_account_correlations()
    assert isinstance(acc_links, list)
    for link in acc_links:
        assert link.correlation_type == "SHARED_ACCOUNT"
        assert link.signal_type == "Hidden Link: Shared Bank Account Lead"
        assert link.severity == "CRITICAL"
        assert link.details.get("provenance_nature") == "INFERRED_CONNECTION"


def test_hidden_link_shared_vehicle_detection():
    """Verify discovery of subjects associated with the same vehicle registration."""
    engine = get_cross_domain_fusion_engine()
    veh_links = engine.detect_shared_vehicle_correlations()
    assert isinstance(veh_links, list)
    for link in veh_links:
        assert link.correlation_type == "SHARED_VEHICLE"
        assert link.signal_type == "Hidden Link: Shared Vehicle Lead"
        assert link.details.get("provenance_nature") == "INFERRED_CONNECTION"


def test_hidden_link_cross_case_detection():
    """Verify discovery of entities linked across multiple distinct FIR cases."""
    engine = get_cross_domain_fusion_engine()
    cross_cases = engine.detect_cross_case_link_correlations()
    assert isinstance(cross_cases, list)
    for link in cross_cases:
        assert link.correlation_type == "CROSS_CASE_LINK"
        assert link.signal_type == "Hidden Link: Cross-Case Association"
        assert len(link.details.get("case_ids", [])) >= 2 or len(link.entities_involved) >= 2


def test_hidden_link_intermediary_bridge_detection():
    """Verify discovery of intermediary bridge nodes connecting unlinked subjects."""
    engine = get_cross_domain_fusion_engine()
    bridges = engine.detect_intermediary_bridge_correlations()
    assert isinstance(bridges, list)
    for link in bridges:
        assert link.correlation_type == "INTERMEDIARY_BRIDGE"
        assert link.signal_type == "Hidden Link: Intermediary Bridge Node"
        assert len(link.entities_involved) >= 3
        assert link.details.get("provenance_nature") == "INFERRED_CONNECTION"


def test_hidden_link_direct_vs_derived_provenance():
    """Verify that direct evidence and derived hidden connections are clearly distinguished."""
    engine = get_cross_domain_fusion_engine()
    summary = engine.get_all_cross_domain_correlations()

    direct_count = 0
    derived_count = 0

    for item in summary.correlations:
        prov_nature = item.details.get("provenance_nature", "DIRECT_EVIDENCE")
        assert prov_nature in ("DIRECT_EVIDENCE", "INFERRED_CONNECTION")
        if prov_nature == "DIRECT_EVIDENCE":
            direct_count += 1
        elif prov_nature == "INFERRED_CONNECTION":
            derived_count += 1
            assert "intermediary_nodes" in item.details or "derivation_basis" in item.details

    assert direct_count + derived_count == summary.total_correlations


def test_hidden_link_no_fabricated_evidence():
    """Verify that all hidden link correlations have grounded evidence snippets without empty placeholders."""
    engine = get_cross_domain_fusion_engine()
    summary = engine.get_all_cross_domain_correlations()

    for item in summary.correlations:
        assert item.evidence_snippet is not None
        assert len(item.evidence_snippet.strip()) > 10
        assert item.source is not None
        assert len(item.source.strip()) > 0
        assert item.confidence_score > 0.0
