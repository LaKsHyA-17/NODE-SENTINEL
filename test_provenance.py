# -*- coding: utf-8 -*-
"""
Automated Test Suite for Evidence Provenance and Hybrid NLP Extraction
"""
import pytest
from app.core.nlp_extractor import NLPExtractor


def test_entity_provenance_extraction():
    extractor = NLPExtractor()
    sample_text = "Suspect Tariq Ahmad called +919811223344 and drove vehicle DL01AB1234 in FIR-2024-NDPS-01."
    meta = {"filename": "incident_report_01.txt", "case_id": "FIR-2024-NDPS-01"}

    entities = extractor.extract_entities_with_provenance(sample_text, source_metadata=meta)
    assert len(entities) > 0

    for ent in entities:
        assert "provenance" in ent
        prov = ent["provenance"]
        assert prov["source_file"] == "incident_report_01.txt"
        assert prov["case_id"] == "FIR-2024-NDPS-01"
        assert "file_hash" in prov
        assert "char_start" in prov
        assert "confidence" in prov


def test_triplet_provenance_extraction():
    extractor = NLPExtractor()
    sample_text = "Tariq Ahmad transferred ₹500000 to ACC987654321."
    meta = {"filename": "bank_subpoena.pdf"}

    entities = extractor.extract_entities_with_provenance(sample_text, source_metadata=meta)
    triplets = extractor.extract_triplets_with_provenance(sample_text, entities, source_metadata=meta)

    assert len(triplets) > 0
    for trip in triplets:
        assert "provenance" in trip
        prov = trip["provenance"]
        assert prov["source_file"] == "bank_subpoena.pdf"
        assert "char_start" in prov
        assert "evidence_snippet" in prov
