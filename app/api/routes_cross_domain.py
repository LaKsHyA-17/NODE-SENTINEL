# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Cross-Domain Correlation & ML Anomaly API Router
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query

from app.core.cross_domain_engine import get_cross_domain_fusion_engine
from app.core.ml_anomaly import get_ml_anomaly_detector
from app.models.cross_domain_models import (
    CaseCrossDomainProfile,
    CrossDomainSummaryResponse,
    EntityCrossDomainProfile,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics/cross-domain", tags=["cross-domain-analytics"])


@router.get("/correlations", response_model=CrossDomainSummaryResponse)
def get_all_correlations():
    """Retrieve system-wide fused cross-domain correlations across all monitored telemetry streams."""
    try:
        engine = get_cross_domain_fusion_engine()
        return engine.get_all_cross_domain_correlations()
    except Exception as e:
        logger.error(f"Error fetching cross-domain correlations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_id}", response_model=EntityCrossDomainProfile)
def get_entity_cross_domain_profile(entity_id: str):
    """Retrieve cross-domain correlation profile for a specific person or identifier."""
    clean_id = (entity_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID cannot be blank")
    try:
        engine = get_cross_domain_fusion_engine()
        return engine.get_entity_cross_domain_profile(clean_id)
    except Exception as e:
        logger.error(f"Error fetching cross-domain profile for {clean_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/case/{case_id}", response_model=CaseCrossDomainProfile)
def get_case_cross_domain_profile(case_id: str):
    """Retrieve cross-domain correlation profile for an entire case or police FIR record."""
    clean_case = (case_id or "").strip()
    if not clean_case:
        raise HTTPException(status_code=400, detail="Case ID cannot be blank")
    try:
        engine = get_cross_domain_fusion_engine()
        return engine.get_case_cross_domain_profile(clean_case)
    except Exception as e:
        logger.error(f"Error fetching cross-domain profile for case {clean_case}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml-anomalies")
def get_ml_anomalies(threshold: float = Query(45.0, ge=0.0, le=100.0)):
    """Retrieve unsupervised ML anomaly scores and feature attributions for all entities."""
    try:
        detector = get_ml_anomaly_detector()
        anomalies = detector.detect_population_anomalies(threshold=threshold)
        return {
            "total_anomalies": len(anomalies),
            "threshold_applied": threshold,
            "anomalies": anomalies,
            "disclaimer": "Decision-Support Notice: Unsupervised ML statistical outlier scores require investigator verification.",
        }
    except Exception as e:
        logger.error(f"Error computing ML anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml-anomalies/{entity_id}")
def get_entity_ml_anomaly(entity_id: str):
    """Retrieve detailed ML anomaly score and feature attributions for a single entity."""
    clean_id = (entity_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID cannot be blank")
    try:
        detector = get_ml_anomaly_detector()
        return detector.evaluate_entity_anomaly(clean_id)
    except Exception as e:
        logger.error(f"Error evaluating ML anomaly for {clean_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
