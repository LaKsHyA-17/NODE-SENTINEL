# -*- coding: utf-8 -*-
"""
CDR (Call Detail Record) API Routes for NODE SENTINEL.
Endpoints for CDR file/data ingestion, call analytics, contact interactions,
sliding-window burst detection, and chronological call timeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.core.audit_logger import audit_logger
from app.core.cdr_analytics import CDRService
from app.core.cdr_parser import CDRParser
from app.core.graph_engine import get_graph_engine
from app.models.audit_models import AuditAction
from app.models.cdr_models import (
    CDRAnalysisResult,
    CDRIngestResponse,
    CDRTimelineItem,
    CommunicationBurst,
    ContactSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cdr", tags=["cdr"])


def get_cdr_service() -> CDRService:
    return CDRService(graph_engine=get_graph_engine())


@router.post("/ingest", response_model=CDRIngestResponse)
async def ingest_cdr(
    request: Request,
    file: Optional[UploadFile] = File(None, description="CSV or JSON CDR file to upload"),
    raw_data: Optional[str] = Form(None, description="Raw CSV or JSON text string"),
    case_id: Optional[str] = Form(None, description="Associated case reference code"),
):
    """
    Ingest Call Detail Records (CDR) into NODE SENTINEL.
    Accepts:
    1. Multipart file upload (CSV or JSON format).
    2. Form data containing raw text.
    3. Direct JSON body with records array.
    
    Parses, validates, normalizes phone numbers, wires CALLS relationships into Knowledge Graph,
    and returns structured counts of processed vs rejected records with explanatory details.
    """
    valid_records = []
    rejected_count = 0
    warnings: List[str] = []
    errors: List[str] = []
    source_name = None

    # Check for direct JSON body first if content-type is application/json
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            body_json = await request.json()
            if not body_json:
                raise HTTPException(status_code=400, detail="JSON payload is empty")
            valid_records, rejected_count, warnings, errors = CDRParser.parse_json(
                body_json, source_name="api_json_body"
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON body: {str(e)}")

    # Handle file upload
    elif file is not None:
        filename = file.filename or "uploaded_cdr_file"
        source_name = filename
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext and ext not in {"csv", "json", "txt"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '.{ext}' for CDR ingestion. Allowed: CSV, JSON, TXT"
            )

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded CDR file is completely empty")

        if len(file_bytes) > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file size ({len(file_bytes) / (1024*1024):.2f} MB) exceeds maximum allowed limit of 25 MB"
            )

        filename_lower = filename.lower()
        if filename_lower.endswith(".json"):
            valid_records, rejected_count, warnings, errors = CDRParser.parse_json(
                file_bytes, source_name=filename
            )
        else:
            # Default to CSV parser
            valid_records, rejected_count, warnings, errors = CDRParser.parse_csv(
                file_bytes, source_name=filename
            )

    # Handle raw text parameter
    elif raw_data:
        text = raw_data.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Raw CDR text is empty")
        source_name = "raw_input"
        if text.startswith("{") or text.startswith("["):
            valid_records, rejected_count, warnings, errors = CDRParser.parse_json(
                text, source_name=source_name
            )
        else:
            valid_records, rejected_count, warnings, errors = CDRParser.parse_csv(
                text, source_name=source_name
            )

    else:
        # Check if raw text body was sent without multipart
        raw_body = await request.body()
        if raw_body:
            text = raw_body.decode("utf-8", errors="replace").strip()
            if text.startswith("{") or text.startswith("["):
                valid_records, rejected_count, warnings, errors = CDRParser.parse_json(
                    text, source_name="http_body"
                )
            else:
                valid_records, rejected_count, warnings, errors = CDRParser.parse_csv(
                    text, source_name="http_body"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No CDR data provided. Please upload a CSV/JSON file or supply raw text.",
            )

    # If case_id was supplied, tag any records without case_id
    if case_id:
        for r in valid_records:
            if not r.case_id:
                r.case_id = case_id

    # If no valid records found and all were rejected, return 422 with error breakdown
    if not valid_records and errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All CDR records failed validation. See rejection errors.",
                "records_rejected": rejected_count,
                "errors": errors[:20],
            },
        )

    # Ingest valid records into graph and storage
    service = get_cdr_service()
    records_processed, entities_created, relationships_created = service.ingest_records_into_graph(
        valid_records
    )

    audit_logger.log(
        action=AuditAction.INGEST_CDR,
        resource_type="CDR",
        resource_id=case_id or source_name,
        status="SUCCESS" if records_processed > 0 else "NO_RECORDS",
        details={"records_processed": records_processed, "records_rejected": rejected_count},
    )

    return CDRIngestResponse(
        status="success" if records_processed > 0 else "no_records",
        records_processed=records_processed,
        records_rejected=rejected_count,
        entities_created=entities_created,
        relationships_created=relationships_created,
        warnings=warnings[:30],
        errors=errors[:30],
    )


@router.get("/analyze/{entity_id}", response_model=CDRAnalysisResult)
def analyze_cdr(
    entity_id: str,
    burst_window_hours: int = Query(24, ge=1, le=168, description="Sliding window size in hours for burst analysis"),
    burst_min_calls: int = Query(5, ge=2, le=100, description="Minimum calls required to evaluate a burst"),
    timeline_limit: int = Query(25, ge=1, le=100, description="Number of chronological calls in preview"),
):
    """
    Perform comprehensive CDR analysis for a Person, Phone, or Phone Number.
    Returns:
    - Summary Call Statistics (total, incoming, outgoing, missed, total/avg duration)
    - Top Contacts with interaction details
    - Objective Communication Indicators (neutral terminology)
    - Communication Bursts
    - Graph Centrality metrics
    - Chronological timeline preview
    """
    clean_id = entity_id.strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID cannot be blank")

    service = get_cdr_service()
    result = service.analyze_entity_cdr(
        entity_id=clean_id,
        burst_window_hours=burst_window_hours,
        burst_min_calls=burst_min_calls,
        timeline_limit=timeline_limit,
    )
    audit_logger.log(
        action=AuditAction.CDR_ANALYSIS,
        resource_type="ENTITY",
        resource_id=clean_id,
        status="SUCCESS",
        details={"total_calls": result.statistics.total_calls, "total_contacts": len(result.top_contacts)},
    )
    return result


@router.get("/contacts/{entity_id}", response_model=List[ContactSummary])
def get_contacts(
    entity_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum contacts to return"),
):
    """
    Retrieve aggregated contact-level statistics for a Person or Phone.
    Includes call count, total conversation duration, average duration, and interaction window.
    """
    clean_id = entity_id.strip()
    service = get_cdr_service()
    _, phone_numbers = service.resolve_entity_phones(clean_id)
    if not phone_numbers:
        return []

    _, records = service.calculate_call_statistics(phone_numbers)
    return service.calculate_contact_summaries(phone_numbers, records, limit=limit)


@router.get("/bursts/{entity_id}", response_model=List[CommunicationBurst])
def get_bursts(
    entity_id: str,
    window_hours: int = Query(24, ge=1, le=168, description="Burst detection sliding window in hours"),
    min_calls: int = Query(5, ge=2, le=100, description="Minimum calls to qualify as a candidate burst"),
):
    """
    Detect communication bursts for a Person or Phone over a sliding time window.
    Flags sudden surges in calling volume compared to baseline communication rates.
    """
    clean_id = entity_id.strip()
    service = get_cdr_service()
    _, phone_numbers = service.resolve_entity_phones(clean_id)
    if not phone_numbers:
        return []

    _, records = service.calculate_call_statistics(phone_numbers)
    return service.detect_communication_bursts(
        records, window_hours=window_hours, min_calls=min_calls
    )


@router.get("/timeline/{entity_id}", response_model=List[CDRTimelineItem])
def get_timeline(
    entity_id: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum call records to return"),
):
    """
    Retrieve chronological call records for an entity, sorted newest first by default.
    """
    clean_id = entity_id.strip()
    service = get_cdr_service()
    return service.get_chronological_timeline(clean_id, limit=limit, reverse=True)
