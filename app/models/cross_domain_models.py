# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Cross-Domain Correlation Data Models
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CrossDomainCorrelationItem(BaseModel):
    correlation_id: str
    correlation_type: str  # e.g. CALL_TO_TRANSFER, CO_TEMPORAL_PRESENCE, MULTI_CHANNEL_MATCH
    severity: str          # CRITICAL, HIGH, ELEVATED, NOTICE
    title: str
    description: str
    entities_involved: List[str]
    time_window_minutes: Optional[float] = None
    first_timestamp: str
    second_timestamp: Optional[str] = None
    evidence_channels: List[str]  # e.g. ["CDR_TELEPHONY", "BANK_LEDGER", "LOCATION_SURVEILLANCE"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: Optional[str] = None


class CrossDomainSummaryResponse(BaseModel):
    total_correlations: int
    critical_count: int
    high_count: int
    correlations: List[CrossDomainCorrelationItem]
    channels_monitored: List[str]
    disclaimer: str = "Requires Investigator Verification: Cross-domain telemetry links represent statistical temporal correlations."


class EntityCrossDomainProfile(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str
    total_fused_events: int
    highest_severity: str
    correlations: List[CrossDomainCorrelationItem]
    associated_channels: List[str]
    multi_channel_risk_index: float
