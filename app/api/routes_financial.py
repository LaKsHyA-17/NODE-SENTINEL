# -*- coding: utf-8 -*-
"""
Financial Transaction API Routes for NODE SENTINEL.
Endpoints for financial statement/data ingestion, account analytics, counterparty interactions,
flow analysis, velocity burst detection, and chronological transaction tables.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.core.audit_logger import audit_logger
from app.core.financial_analytics import (
    DEFAULT_HIGH_VALUE_THRESHOLD,
    FinancialService,
    get_financial_storage,
)
from app.core.financial_parser import FinancialParser
from app.core.graph_engine import get_graph_engine
from app.models.audit_models import AuditAction
from app.models.financial_models import (
    CounterpartySummary,
    FinancialAnalysisResult,
    FinancialBurst,
    FinancialFlowSummary,
    FinancialIndicator,
    FinancialIngestResponse,
    TransactionItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/financial", tags=["financial"])


def get_financial_service() -> FinancialService:
    return FinancialService(graph_engine=get_graph_engine(), storage=get_financial_storage())


@router.post("/ingest", response_model=FinancialIngestResponse)
async def ingest_financial(
    request: Request,
    file: Optional[UploadFile] = File(None, description="CSV or JSON financial transaction file to upload"),
    raw_data: Optional[str] = Form(None, description="Raw CSV or JSON text string"),
    case_id: Optional[str] = Form(None, description="Associated case reference code"),
):
    """
    Ingest Financial Transaction Records into NODE SENTINEL.
    Accepts:
    1. Multipart file upload (CSV or JSON format).
    2. Form data containing raw text.
    3. Direct JSON body with transactions array.

    Parses, validates, normalizes account identifiers, creates/reuses BankAccount nodes in the Knowledge Graph,
    wires TRANSFERRED_MONEY relationships, links Case entities, and returns counts of processed vs rejected records.
    """
    valid_records = []
    rejected_count = 0
    warnings: List[str] = []
    errors: List[str] = []
    source_name = None

    # 1. Direct JSON body
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            body_json = await request.json()
            if not body_json:
                raise HTTPException(status_code=400, detail="JSON payload is empty")
            valid_records, rejected_count, warnings, errors = FinancialParser.parse_json(
                body_json, source_name="api_json_body"
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON body: {str(e)}")

    # 2. File upload
    elif file is not None:
        filename = file.filename or "uploaded_financial_file"
        source_name = filename
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext and ext not in {"csv", "json", "txt"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '.{ext}' for financial ingestion. Allowed: CSV, JSON, TXT"
            )

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded financial transaction file is completely empty")

        if len(file_bytes) > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file size ({len(file_bytes) / (1024*1024):.2f} MB) exceeds maximum allowed limit of 25 MB"
            )

        filename_lower = filename.lower()
        if filename_lower.endswith(".json"):
            valid_records, rejected_count, warnings, errors = FinancialParser.parse_json(
                file_bytes, source_name=filename
            )
        else:
            valid_records, rejected_count, warnings, errors = FinancialParser.parse_csv(
                file_bytes, source_name=filename
            )

    # 3. Form raw_data
    elif raw_data:
        text = raw_data.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Raw financial transaction text is empty")
        source_name = "raw_input"
        if text.startswith("{") or text.startswith("["):
            valid_records, rejected_count, warnings, errors = FinancialParser.parse_json(
                text, source_name=source_name
            )
        else:
            valid_records, rejected_count, warnings, errors = FinancialParser.parse_csv(
                text, source_name=source_name
            )

    # 4. Raw text body
    else:
        raw_body = await request.body()
        if raw_body:
            text = raw_body.decode("utf-8", errors="replace").strip()
            if text.startswith("{") or text.startswith("["):
                valid_records, rejected_count, warnings, errors = FinancialParser.parse_json(
                    text, source_name="http_body"
                )
            else:
                valid_records, rejected_count, warnings, errors = FinancialParser.parse_csv(
                    text, source_name="http_body"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No financial data provided. Please upload a CSV/JSON file or supply raw text.",
            )

    # If case_id was supplied, tag any records without case_id
    if case_id:
        for r in valid_records:
            if not r.case_id:
                r.case_id = case_id

    # If no valid records found and errors occurred
    if not valid_records and errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All financial records failed validation. See rejection errors.",
                "records_rejected": rejected_count,
                "errors": errors[:20],
            },
        )

    # Ingest valid records into Knowledge Graph and storage
    service = get_financial_service()
    records_processed, entities_created, relationships_created = service.ingest_records_into_graph(
        valid_records
    )

    audit_logger.log(
        action=AuditAction.INGEST_FINANCIAL,
        resource_type="FINANCIAL",
        resource_id=case_id or source_name,
        status="SUCCESS" if records_processed > 0 else "NO_RECORDS",
        details={"records_processed": records_processed, "records_rejected": rejected_count},
    )

    return FinancialIngestResponse(
        status="success" if records_processed > 0 else "no_records",
        records_processed=records_processed,
        records_rejected=rejected_count,
        entities_created=entities_created,
        relationships_created=relationships_created,
        warnings=warnings[:30],
        errors=errors[:30],
    )


@router.get("/analyze/{entity_id}", response_model=FinancialAnalysisResult)
def analyze_financial(
    entity_id: str,
    high_value_threshold: float = Query(DEFAULT_HIGH_VALUE_THRESHOLD, ge=0.0, description="Investigation threshold for high-value transaction detection"),
    burst_window_minutes: int = Query(120, ge=1, le=10080, description="Sliding window in minutes for rapid velocity burst detection"),
    timeline_limit: int = Query(50, ge=1, le=500, description="Maximum transactions in preview table"),
):
    """
    Perform comprehensive Financial Transaction Analysis for a Person or BankAccount.
    Returns:
    - Summary Transaction Statistics (totals, incoming, outgoing, net flow, average, extremes)
    - Top Counterparties with incoming/outgoing metrics
    - Objective Explainable Financial Indicators (neutral terminology)
    - Rapid Velocity Bursts
    - Flow Summary (top sources & destinations, net flow)
    - Graph Centrality metrics
    - Chronological transaction table preview
    """
    clean_id = entity_id.strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID cannot be blank")

    service = get_financial_service()
    res = service.analyze_entity_finances(
        entity_id=clean_id,
        high_value_threshold=high_value_threshold,
        burst_window_minutes=burst_window_minutes,
        timeline_limit=timeline_limit,
    )
    audit_logger.log(
        action=AuditAction.FINANCIAL_ANALYSIS,
        resource_type="ENTITY",
        resource_id=clean_id,
        status="SUCCESS",
        details={
            "total_transactions": res.statistics.total_transactions,
            "incoming": res.statistics.total_incoming_amount,
            "outgoing": res.statistics.total_outgoing_amount,
        },
    )
    return res


@router.get("/transactions/{entity_id}", response_model=List[TransactionItem])
def get_transactions(
    entity_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum transactions to return"),
):
    """
    Retrieve chronological transaction records for an entity, sorted newest first by default.
    """
    clean_id = entity_id.strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID cannot be blank")

    service = get_financial_service()
    _, accounts = service.resolve_entity_accounts(clean_id)
    if not accounts:
        return []

    records = service.get_all_records_for_accounts(accounts)
    items = []
    for r in records[:limit]:
        s_node = service.graph.get_node(r.sender)
        r_node = service.graph.get_node(r.receiver)
        s_name = s_node.name if s_node else r.sender
        r_name = r_node.name if r_node else r.receiver

        items.append(TransactionItem(
            transaction_id=r.transaction_id,
            timestamp=r.timestamp.isoformat(),
            sender=r.sender,
            sender_name=s_name,
            receiver=r.receiver,
            receiver_name=r_name,
            amount=r.amount,
            currency=r.currency,
            transaction_type=r.transaction_type.value,
            account_id=r.account_id,
            case_id=r.case_id,
            location=r.location,
            description=r.description,
        ))
    return items


@router.get("/counterparties/{entity_id}", response_model=List[CounterpartySummary])
def get_counterparties(
    entity_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum counterparties to return"),
):
    """
    Retrieve aggregated counterparty-level statistics for a Person or BankAccount.
    Includes transaction count, total amount, average amount, incoming/outgoing breakdown, and interaction window.
    """
    clean_id = entity_id.strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID cannot be blank")

    service = get_financial_service()
    _, accounts = service.resolve_entity_accounts(clean_id)
    if not accounts:
        return []

    records = service.get_all_records_for_accounts(accounts)
    return service.calculate_counterparty_summaries(accounts, records, limit=limit)


@router.get("/anomalies/{entity_id}", response_model=Dict[str, Any])
def get_anomalies(
    entity_id: str,
    high_value_threshold: float = Query(DEFAULT_HIGH_VALUE_THRESHOLD, ge=0.0, description="Investigation threshold for high-value transactions"),
    burst_window_minutes: int = Query(120, ge=1, le=10080, description="Window in minutes for burst detection"),
):
    """
    Retrieve explainable financial indicators and velocity bursts for an entity.
    """
    clean_id = entity_id.strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID cannot be blank")

    service = get_financial_service()
    _, accounts = service.resolve_entity_accounts(clean_id)
    if not accounts:
        return {"indicators": [], "bursts": []}

    records = service.get_all_records_for_accounts(accounts)
    bursts = service.detect_transaction_bursts(records, window_minutes=burst_window_minutes)
    indicators = service.evaluate_financial_indicators(
        accounts, records, bursts, high_value_threshold=high_value_threshold
    )

    return {
        "entity_id": clean_id,
        "indicators": [ind.model_dump() for ind in indicators],
        "bursts": [b.model_dump() for b in bursts],
    }


@router.get("/flow/{entity_id}", response_model=FinancialFlowSummary)
def get_flow(
    entity_id: str,
    top_k: int = Query(5, ge=1, le=20, description="Number of top sources and destinations to return"),
):
    """
    Retrieve money-flow summary (incoming, outgoing, top sources, top destinations, net flow).
    """
    clean_id = entity_id.strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Entity ID cannot be blank")

    service = get_financial_service()
    _, accounts = service.resolve_entity_accounts(clean_id)
    if not accounts:
        return FinancialFlowSummary()

    records = service.get_all_records_for_accounts(accounts)
    return service.calculate_flow_summary(accounts, records, top_k=top_k)
