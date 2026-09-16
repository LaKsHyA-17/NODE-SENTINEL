# -*- coding: utf-8 -*-
"""
Evidence Viewer API Routes for NODE SENTINEL.
Exposes provenance lookup, character span highlighting, and related evidence
records across Graph Nodes, Edges, Risk Factors, Anomalies, Timeline Events,
and AI Assistant Evidence.
"""
from __future__ import annotations

import logging
from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException

from app.core.evidence_engine import EvidenceEngine
from app.core.graph_engine import get_graph_engine
from app.models.evidence_models import (
    EvidenceProvenanceRecord,
    EvidenceLookupResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.get("/lookup", response_model=EvidenceLookupResponse)
def lookup_evidence(
    id: str = Query(..., description="Target identifier (node ID, edge ID, factor ID, correlation ID, or event ID)"),
    type: Optional[str] = Query(None, description="Item type: NODE, GRAPH_RELATIONSHIP, RISK_FACTOR, ANOMALY, TIMELINE_EVENT, AI_ASSISTANT_EVIDENCE"),
    entity_id: Optional[str] = Query(None, description="Associated subject entity ID"),
    source: Optional[str] = Query(None, description="Edge source node ID"),
    target: Optional[str] = Query(None, description="Edge target node ID"),
    relationship: Optional[str] = Query(None, description="Relationship type label")
):
    """
    Unified Evidence Provenance Lookup Endpoint.
    Resolves complete empirical evidence, source file, case ID, timestamps,
    confidence, and highlighted character span.
    """
    clean_id = (id or "").strip()
    if not clean_id and not (source and target):
        raise HTTPException(status_code=400, detail="Identifier or source/target coordinates required.")

    engine = EvidenceEngine(get_graph_engine())
    return engine.lookup_evidence(
        identifier=clean_id,
        item_type=type,
        entity_id=entity_id,
        source=source,
        target=target,
        relationship=relationship
    )


@router.get("/node/{node_id}", response_model=EvidenceProvenanceRecord)
def get_node_evidence(node_id: str):
    """
    Retrieve provenance evidence record for a graph entity/node.
    """
    clean_id = (node_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Node ID cannot be blank.")

    engine = EvidenceEngine(get_graph_engine())
    return engine.get_node_evidence(clean_id)


@router.get("/edge/{edge_id}", response_model=EvidenceProvenanceRecord)
def get_edge_evidence(edge_id: str):
    """
    Retrieve provenance evidence record for a graph edge/relationship.
    """
    clean_id = (edge_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Edge ID cannot be blank.")

    engine = EvidenceEngine(get_graph_engine())
    return engine.get_edge_evidence(edge_id=clean_id)


@router.get("/related", response_model=List[EvidenceProvenanceRecord])
def get_related_evidence(
    case_id: Optional[str] = Query(None, description="Filter by case / FIR ID"),
    source_file: Optional[str] = Query(None, description="Filter by source file name"),
    entity_id: Optional[str] = Query(None, description="Filter by associated entity ID"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Retrieve related provenance records sharing case ID, source document, or entity context.
    """
    engine = EvidenceEngine(get_graph_engine())
    return engine.get_related_evidence(
        case_id=case_id,
        source_file=source_file,
        entity_id=entity_id,
        limit=limit
    )
