# -*- coding: utf-8 -*-
"""
NODE SENTINEL - AI Investigation Assistant Data Models (STEP 13)
Defines structured request/response schemas, grounded citations, action links,
and conversation state models for investigative decision support.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class AssistantRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AssistantActionType(str, Enum):
    VIEW_PROFILE = "VIEW_PROFILE"
    VIEW_NETWORK = "VIEW_NETWORK"
    ANALYZE_CALLS = "ANALYZE_CALLS"
    ANALYZE_FINANCES = "ANALYZE_FINANCES"
    VIEW_TIMELINE = "VIEW_TIMELINE"
    VIEW_RISK = "VIEW_RISK"
    VIEW_CASE = "VIEW_CASE"
    VIEW_EVIDENCE = "VIEW_EVIDENCE"
    FIND_SHORTEST_PATH = "FIND_SHORTEST_PATH"


class AssistantAction(BaseModel):
    action_type: AssistantActionType
    label: str
    entity_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class AssistantSource(BaseModel):
    source_type: str  # GRAPH, CDR, FINANCIAL, TIMELINE, RISK, CASE, SEARCH, BIOMETRIC, PROVENANCE
    reference_id: str
    title: str
    details: Optional[str] = None


class AssistantMessage(BaseModel):
    role: AssistantRole
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InvestigationAssistantRequest(BaseModel):
    query: str
    selected_entity_id: Optional[str] = None
    conversation_id: Optional[str] = None
    history: Optional[List[AssistantMessage]] = None


class InvestigationAssistantResponse(BaseModel):
    answer: str
    selected_entity: Optional[Dict[str, Any]] = None
    is_ambiguous: bool = False
    clarification_needed: Optional[str] = None
    candidate_entities: Optional[List[Dict[str, Any]]] = None
    evidence: List[str] = Field(default_factory=list)
    relevant_entities: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_cases: List[str] = Field(default_factory=list)
    relevant_calls: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_financial: List[Dict[str, Any]] = Field(default_factory=list)
    risk_indicators: List[str] = Field(default_factory=list)
    confidence_score: float = 0.95
    confidence_label: str = "HIGH"
    uncertainty: Optional[str] = None
    sources: List[AssistantSource] = Field(default_factory=list)
    actions: List[AssistantAction] = Field(default_factory=list)
    conversation_id: str
