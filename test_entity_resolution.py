# -*- coding: utf-8 -*-
"""
Automated Test Suite for Entity Resolution, Canonicalization, and Provenance Preservation.

Tests:
1. Deterministic Phone Normalization (+919811223344, 9811223344, 9811-223-344 -> PHONE_9811223344)
2. Deterministic Vehicle Normalization (DL01AB9988, dl 01 ab 9988, DL-01-AB-9988 -> VEH_DL01AB9988)
3. Deterministic Bank Account Normalization (ACC987654321, A/C 987654321, 987654321 -> ACC_987654321)
4. Deterministic Case Normalization (FIR-2024-311, CASE_FIR_2024_311, FIR/2024/311 -> CASE_FIR-2024-311)
5. Person Name Matching: Exact Match vs Conservative Fuzzy Candidate Matching
6. Ambiguous matches return 'possible_match' with is_ambiguous=True and require investigator verification
7. Prevention of false merges on similar person names (Rahul Sharma vs Rahul Verma)
8. Preservation of aliases, observed values, and provenance history
9. Universal Search integration with canonical identifiers
10. FIR Ingestion integration updating canonical entities
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.entity_resolver import EntityResolver, get_entity_resolver
from app.core.graph_engine import get_graph_engine
from app.core.nlp_extractor import NLPExtractor
from app.core.universal_search import get_universal_search
from app.models.graph_models import Node, NodeType
from app.api.routes_ingest import load_dataset_by_name


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_graph():
    g = get_graph_engine()
    g.clear()
    load_dataset_by_name(g, "syndicate_network.json")
    yield
    g.clear()
    load_dataset_by_name(g, "syndicate_network.json")


# ---------------------------------------------------------------------------
# 1. Identifier Canonicalization Tests
# ---------------------------------------------------------------------------

def test_phone_canonicalization_variations():
    """Verify that all phone variations map to the exact same canonical ID and E.164 string."""
    resolver = get_entity_resolver()

    test_cases = [
        "+919811223344",
        "9811223344",
        "9811-223-344",
        "+91 98112 23344",
        "09811223344",
        "+91-9811-223344",
        "9811.223.344"
    ]

    for raw in test_cases:
        cid, e164, digits = resolver.canonicalize_phone(raw)
        assert cid == "PHONE_9811223344", f"Failed for {raw}: got {cid}"
        assert e164 == "+919811223344", f"Failed for {raw}: got {e164}"
        assert digits == "9811223344"


def test_vehicle_canonicalization_variations():
    """Verify vehicle plate variations map to canonical VEH_<PLATE> ID."""
    resolver = get_entity_resolver()

    test_cases = [
        "DL01AB9988",
        "dl 01 ab 9988",
        "DL-01-AB-9988",
        "dl.01.ab.9988",
        " dl  01  ab  9988 "
    ]

    for raw in test_cases:
        cid, clean_plate = resolver.canonicalize_vehicle(raw)
        assert cid == "VEH_DL01AB9988", f"Failed for {raw}: got {cid}"
        assert clean_plate == "DL01AB9988"


def test_bank_account_canonicalization_variations():
    """Verify bank account variations map to canonical ACC_<DIGITS> ID."""
    resolver = get_entity_resolver()

    test_cases = [
        "ACC987654321",
        "A/C 987654321",
        "987654321",
        "ACC_987654321",
        "A/C-987654321",
        "Account: 987654321"
    ]

    for raw in test_cases:
        cid, clean_num = resolver.canonicalize_account(raw)
        assert cid == "ACC_987654321", f"Failed for {raw}: got {cid}"
        assert clean_num == "987654321"


def test_case_id_canonicalization_variations():
    """Verify case/FIR ID variations map to canonical CASE_<CODE> ID."""
    resolver = get_entity_resolver()

    test_cases = [
        "FIR-2024-311",
        "CASE_FIR_2024_311",
        "fir 2024 311",
        "FIR/2024/311",
        "case-fir-2024-311"
    ]

    for raw in test_cases:
        cid, clean_code = resolver.canonicalize_case(raw)
        assert cid == "CASE_FIR-2024-311", f"Failed for {raw}: got {cid}"
        assert clean_code == "FIR-2024-311"


# ---------------------------------------------------------------------------
# 2. Person Name Matching & Ambiguity Tests
# ---------------------------------------------------------------------------

def test_person_exact_match():
    """Exact person name matching returns confirmed status."""
    resolver = get_entity_resolver()
    res = resolver.match_person_name("Tariq Ahmad", "Tariq Ahmad")
    assert res["match_score"] == 100
    assert res["match_type"] == "exact"
    assert res["is_ambiguous"] is False
    assert res["verification_status"] == "CONFIRMED_MATCH"


def test_person_conservative_fuzzy_match():
    """Conservative fuzzy candidate matching flags possible match with verification requirement."""
    resolver = get_entity_resolver()

    # Slight spelling difference: Tariq Ahmed vs Tariq Ahmad
    res = resolver.match_person_name("Tariq Ahmed", "Tariq Ahmad")
    assert res["match_score"] >= 85
    assert res["match_type"] == "possible_match"
    assert res["is_ambiguous"] is True
    assert res["verification_status"] == "REQUIRES_INVESTIGATOR_VERIFICATION"


def test_person_partial_token_candidate_match():
    """Single-token first name query matches candidate with ambiguity flag."""
    resolver = get_entity_resolver()
    res = resolver.match_person_name("Tariq", "Tariq Ahmad")
    assert res["match_score"] >= 90
    assert res["match_type"] == "possible_match"
    assert res["is_ambiguous"] is True
    assert res["verification_status"] == "REQUIRES_INVESTIGATOR_VERIFICATION"


def test_distinct_persons_never_merged_on_name_similarity():
    """
    Ensure distinct persons with similar names (e.g. Rahul Sharma vs Rahul Verma,
    or Rajesh Kumar vs Rakesh Kumar) are NOT considered matches.
    """
    resolver = get_entity_resolver()

    res1 = resolver.match_person_name("Rahul Sharma", "Rahul Verma")
    # Should not be treated as a match
    assert res1["match_type"] == "none" or res1["match_score"] < 80

    res2 = resolver.match_person_name("Rajesh Kumar", "Rakesh Kumar")
    # Low similarity / distinct identity
    assert res2["is_exact_match"] is False


# ---------------------------------------------------------------------------
# 3. Provenance, Observed Values, and Alias Preservation
# ---------------------------------------------------------------------------

def test_node_property_merging_preserves_aliases_and_provenance():
    """Verify that updating a node preserves observed values, aliases, and provenance history."""
    resolver = get_entity_resolver()
    node = Node("PERSON_TARIQ_AHMAD", NodeType.PERSON, "Tariq Ahmad", {
        "aliases": ["Tariq Bhai"],
        "observed_values": ["Tariq Ahmad"],
        "provenance_history": [{"source_file": "fir_01.txt", "file_hash": "hash111"}]
    })

    new_props = {
        "aliases": ["Don of Okhla"],
        "raw_match": "Tariq Ahmed",
        "provenance": {"source_file": "fir_02.txt", "file_hash": "hash222", "confidence": 0.95}
    }

    resolver.merge_node_properties(node, new_props)

    # Aliases preserved
    assert "Tariq Bhai" in node.properties["aliases"]
    assert "Don of Okhla" in node.properties["aliases"]

    # Observed values preserved
    assert "Tariq Ahmad" in node.properties["observed_values"]
    assert "Tariq Ahmed" in node.properties["observed_values"]

    # Provenance history preserved
    assert len(node.properties["provenance_history"]) == 2
    hashes = {p["file_hash"] for p in node.properties["provenance_history"]}
    assert "hash111" in hashes and "hash222" in hashes


# ---------------------------------------------------------------------------
# 4. Universal Search & Ingestion Integration Tests
# ---------------------------------------------------------------------------

def test_universal_search_resolves_all_phone_variations():
    """Universal Search resolves phone queries regardless of formatting."""
    search = get_universal_search()

    variations = ["+919811223344", "9811223344", "9811-223-344", "+91 98112 23344"]
    for q in variations:
        res = search.search(q)
        assert res.total_results > 0, f"No results for {q}"
        matched_ids = [r.entity_id for r in res.results]
        # In syndicate_network.json, phone ID is PHONE_9811223344
        assert "PHONE_9811223344" in matched_ids, f"PHONE_9811223344 not found for query {q}"


def test_universal_search_resolves_vehicle_variations():
    """Universal Search resolves vehicle queries with lowercase, spaces, or hyphens."""
    search = get_universal_search()

    variations = ["DL01AB9988", "dl 01 ab 9988", "DL-01-AB-9988"]
    for q in variations:
        res = search.search(q)
        assert res.total_results > 0
        matched_ids = [r.entity_id for r in res.results]
        assert "VEH_DL01AB9988" in matched_ids


def test_universal_search_fuzzy_person_ambiguity_flag():
    """Universal search on fuzzy person name returns possible match flagged as ambiguous."""
    search = get_universal_search()

    res = search.search("Tariq Ahmed")
    assert res.total_results > 0
    tariq_res = next((r for r in res.results if "TARIQ" in r.entity_id), None)
    assert tariq_res is not None
    assert tariq_res.match_type == "possible_match"


def test_fir_ingestion_canonicalizes_and_merges_duplicate_identifiers(client: TestClient):
    """
    Verify that sequential FIR uploads with different raw representations of the same
    phone (+919811223344 vs 9811-223-344) resolve to the exact same canonical node.
    """
    g = get_graph_engine()
    initial_nodes_count = len(g.get_all_nodes())

    payload1 = {
        "text": "Suspect Tariq Ahmad called +919811223344 in FIR-2024-991.",
        "source_case_id": "FIR-2024-991"
    }
    resp1 = client.post("/api/ingest/text", json=payload1)
    assert resp1.status_code == 200

    phone_node_1 = g.get_node("PHONE_9811223344")
    assert phone_node_1 is not None

    # Ingest with different phone format (hyphenated raw string)
    payload2 = {
        "text": "Accused Tariq Ahmad was intercepted on phone 9811-223-344 in FIR-2024-991.",
        "source_case_id": "FIR-2024-991"
    }
    resp2 = client.post("/api/ingest/text", json=payload2)
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Should be tracked under merged_entities, not created as a new duplicate node
    merged_ids = [n["id"] for n in data2["merged_entities"]]
    assert "PHONE_9811223344" in merged_ids
