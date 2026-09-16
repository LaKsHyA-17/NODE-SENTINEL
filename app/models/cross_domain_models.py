# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Cross-Domain Correlation Data Models
Standardized schemas for multi-source evidence fusion across:
- Police FIR & Case records
- CDR Telephony logs
- Financial Ledger transfers
- Knowledge Graph topology
- Surveillance Locations & Hotspots
- Chronological Timelines
- Biometric Face match registry
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CrossDomainCorrelationItem(BaseModel):
    correlation_id: str
    correlation_type: str  # e.g. CALL_TO_TRANSFER, CO_TEMPORAL_PRESENCE, FIR_CASE_ASSOCIATION, BIOMETRIC_ALIGNMENT, GRAPH_TOPOLOGY
    signal_type: Optional[str] = Field(None, description="Descriptive category of multi-channel signal")
    severity: str          # CRITICAL, HIGH, ELEVATED, NOTICE
    title: str
    description: str
    entities_involved: List[str]
    time_window_minutes: Optional[float] = None
    first_timestamp: str
    second_timestamp: Optional[str] = None
    timestamp: Optional[str] = Field(None, description="Primary timestamp or temporal anchor")
    source: Optional[str] = Field(None, description="Data sources fused in this correlation")
    evidence_channels: List[str] = Field(default_factory=list, description="List of telemetry channel tags")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score")
    confidence: Optional[float] = Field(None, description="Alias for confidence_score")
    evidence_snippet: Optional[str] = Field(None, description="Supporting textual evidence quote from source")
    graph_relationship: Optional[str] = Field(None, description="Observed Knowledge Graph edge relationship or pathway")
    timeline_link: Optional[Dict[str, Any]] = Field(None, description="Actionable filter parameters for timeline navigation")
    source_evidence_link: Optional[Dict[str, Any]] = Field(None, description="Actionable parameters for provenance inspection")
    details: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: Optional[str] = None


class CrossDomainSummaryResponse(BaseModel):
    total_correlations: int
    critical_count: int
    high_count: int
    correlations: List[CrossDomainCorrelationItem]
    channels_monitored: List[str]
    disclaimer: str = "Decision-Support Notice: Cross-domain telemetry links represent observed statistical correlations across independent data channels. Correlation does not imply causation unless explicitly established in source records."


class EntityCrossDomainProfile(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str
    total_fused_events: int
    highest_severity: str
    correlations: List[CrossDomainCorrelationItem]
    associated_channels: List[str]
    multi_channel_risk_index: float
    disclaimer: str = "Decision-Support Notice: Correlated activities represent observed multi-channel patterns. Requires investigator verification."


class CaseCrossDomainProfile(BaseModel):
    case_id: str
    case_title: Optional[str] = None
    total_fused_events: int
    highest_severity: str
    involved_entities: List[str] = Field(default_factory=list)
    correlations: List[CrossDomainCorrelationItem]
    associated_channels: List[str]
    disclaimer: str = "Decision-Support Notice: Case-level multi-source correlations reflect observed relational and temporal patterns."
