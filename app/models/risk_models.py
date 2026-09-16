# -*- coding: utf-8 -*-
"""
Risk & Anomaly Intelligence Data Models for NODE SENTINEL.
Provides explainable, evidence-backed investigative risk scoring models
integrating Graph, CDR, Financial, Case, Location, and Police Registry feeds.
Strictly adheres to neutral investigator decision-support standards.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class RiskCategory(str, Enum):
    GRAPH = "GRAPH"
    CDR = "CDR"
    FINANCIAL = "FINANCIAL"
    CASE = "CASE"
    LOCATION = "LOCATION"
    BEHAVIORAL = "BEHAVIORAL"
    REGISTRY = "REGISTRY"


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"


class RiskFactor(BaseModel):
    factor_id: str = Field(..., description="Unique identifier for the risk factor")
    category: RiskCategory = Field(..., description="Category of the indicator")
    title: str = Field(..., description="Short descriptive title of the risk factor / anomalous pattern")
    score_contribution: float = Field(..., description="Reproducible points contributed to composite score")
    numerical_contribution: Optional[float] = Field(None, description="Numerical contribution points contributed to total score")
    severity: str = Field(..., description="Severity level: LOW, MODERATE, ELEVATED, HIGH")
    underlying_metric: Optional[str] = Field(None, description="Quantitative metric underpinning the indicator (e.g. '8 calls in 24 hours')")
    explanation: str = Field(..., description="Clear contextual narrative explaining why this indicator matters")
    evidence: str = Field(..., description="Concrete, traceable empirical evidence from project data")
    source: str = Field(..., description="Specific analysis module or data registry source")
    entity_id: str = Field(..., description="Subject or related entity identifier")
    involved_entities: List[str] = Field(default_factory=list, description="All associated entity IDs involved in this factor")
    timestamp: Optional[str] = Field(None, description="Event or observation timestamp where available")
    confidence: Optional[float] = Field(0.95, description="Confidence metric of indicator observation (0.0 - 1.0)")
    timeline_event_type: Optional[str] = Field(None, description="Linked Timeline event type (e.g. FINANCIAL, CALL, CASE)")
    action_hint: Optional[str] = Field(None, description="Suggested UI investigation action label, e.g. 'View CDR', 'View Transactions'")
    action_type: Optional[str] = Field(None, description="Action hook type, e.g. 'VIEW_CDR', 'VIEW_FINANCE', 'VIEW_TIMELINE', 'VIEW_GRAPH'")

    # Backward-compatibility fields for existing templates expecting f.factor / f.points
    factor: Optional[str] = Field(None, description="Legacy factor summary string")
    points: Optional[float] = Field(None, description="Legacy points value")

    @model_validator(mode="after")
    def populate_legacy_fields(self) -> RiskFactor:
        if self.numerical_contribution is None:
            self.numerical_contribution = self.score_contribution
        if self.factor is None:
            self.factor = f"{self.title}: {self.evidence}"
        if self.points is None:
            self.points = self.score_contribution
        if not self.involved_entities and self.entity_id:
            self.involved_entities = [self.entity_id]
        return self


class RiskIntelligenceResult(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str
    risk_score: float = Field(..., description="Composite risk score bounded between 0.0 and 100.0")
    total_score: Optional[float] = Field(None, description="Total composite investigative risk score")
    risk_level: str = Field(..., description="LOW (0-24), MODERATE (25-49), ELEVATED (50-74), HIGH (75-100)")
    severity: Optional[str] = Field(None, description="Investigative severity classification")
    overall_score: float = Field(0.0, description="Legacy alias for risk_score")
    severity_level: str = Field("LOW", description="Legacy alias for risk_level")
    factors: List[RiskFactor] = Field(default_factory=list, description="Ordered list of explainable risk factors / anomalous patterns")
    evidence_summary: List[str] = Field(default_factory=list, description="Bullet summary of key observed facts")
    conclusion: str = Field(
        default="Decision-Support Signal: Observed indicators suggest elevated investigative priority based on available telemetry. Does not prove criminal activity.",
        description="Objective conclusion emphasizing decision-support nature"
    )
    disclaimer: str = Field(
        default="Decision-Support Signal: Investigative risk indicators and anomalous patterns reflect empirical telemetry density across data sources. Scores do not prove criminal activity and require independent investigator verification.",
        description="Legal and decision-support compliance notice"
    )
    thresholds: Dict[str, str] = Field(
        default_factory=lambda: {
            "LOW": "0–24: Baseline investigative priority. Minimal observed risk indicators.",
            "MODERATE": "25–49: Moderate investigative priority. Multiple indicators present.",
            "ELEVATED": "50–74: Elevated investigative priority based on observed multi-source indicators.",
            "HIGH": "75–100: High investigative priority based on observed indicators. Requires investigator verification."
        }
    )
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @model_validator(mode="after")
    def sync_legacy_fields(self) -> RiskIntelligenceResult:
        if self.total_score is None:
            self.total_score = self.risk_score
        if self.severity is None:
            self.severity = self.risk_level
        if not self.overall_score:
            self.overall_score = self.risk_score
        if not self.severity_level or self.severity_level == "LOW":
            self.severity_level = self.risk_level
        return self
