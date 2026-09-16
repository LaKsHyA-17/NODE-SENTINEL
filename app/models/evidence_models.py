# -*- coding: utf-8 -*-
"""
Evidence & Provenance Data Models for NODE SENTINEL.
Provides structured models for the Evidence Viewer across:
- Risk Factors
- Anomalies & Cross-Domain Correlations
- Graph Relationships (Edges)
- Extracted Entities (Nodes)
- AI Assistant Evidence
- Timeline Events
Strictly adheres to neutral decision-support standards and provenance traceability.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EvidenceProvenanceRecord(BaseModel):
    """
    Standardized Evidence Provenance Record.
    """
    record_id: Optional[str] = Field(None, description="Unique record or transaction/call ID if available")
    item_type: str = Field("ENTITY", description="Type: RISK_FACTOR, ANOMALY, GRAPH_RELATIONSHIP, EXTRACTED_ENTITY, AI_ASSISTANT_EVIDENCE, TIMELINE_EVENT, NODE")
    source_file: Optional[str] = Field(None, description="Source document name, e.g. 'FIR_2024_NDPS.txt' or 'cdr_calls.csv'")
    case_id: Optional[str] = Field(None, description="Associated legal or police FIR case ID")
    timestamp: Optional[str] = Field(None, description="Event or extraction timestamp where available")
    entity_id: Optional[str] = Field(None, description="Primary subject entity ID")
    entity_name: Optional[str] = Field(None, description="Primary subject entity display name")
    relationship: Optional[str] = Field(None, description="Graph edge or observed relation label, e.g. CALLS, TRANSFERRED_MONEY")
    target_entity_id: Optional[str] = Field(None, description="Counterparty or connected entity ID")
    target_entity_name: Optional[str] = Field(None, description="Counterparty display name")
    confidence: Optional[float] = Field(0.95, description="Confidence metric of observation (0.0 - 1.0)")
    file_hash: Optional[str] = Field(None, description="Cryptographic SHA-256 hash of the source document")
    char_start: Optional[int] = Field(None, description="Character start offset in source document")
    char_end: Optional[int] = Field(None, description="Character end offset in source document")
    evidence_snippet: Optional[str] = Field(None, description="Empirical verbatim evidence quote or snippet")
    source_text: Optional[str] = Field(None, description="Full or surrounding source text excerpt")
    highlighted_text: Optional[str] = Field(None, description="Pre-highlighted HTML span of the evidence in source text")
    has_provenance: bool = Field(True, description="Whether valid source evidence was located")
    status_message: Optional[str] = Field(None, description="Notice message if provenance is unavailable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual metadata attributes")


class EvidenceLookupResponse(BaseModel):
    """
    Response returned by Evidence Viewer queries.
    """
    primary_record: EvidenceProvenanceRecord
    related_records: List[EvidenceProvenanceRecord] = Field(default_factory=list)
    query_id: str
    query_type: str
    has_evidence: bool = True
    notice: str = "Decision-Support Telemetry: Empirical evidence grounded in subpoenaed, ingested, or registry data."
