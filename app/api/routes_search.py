# -*- coding: utf-8 -*-
"""
Universal Search API Routes for NODE SENTINEL.

Exposes:
- GET  /search          : Search by query string (Person ID, Name, Phone, Vehicle, Case ID, Account)
- POST /search          : Search via JSON payload
- POST /search/face     : Search by uploaded face image -> Person ID -> Universal Entity Profile
- POST /face/search     : Alias for face image biometric search
- GET  /face/identities : List all enrolled biometric identities
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.core.audit_logger import audit_logger
from app.core.face_engine import (
    FaceRecognitionError,
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
)
from app.core.face_storage import get_face_storage
from app.core.universal_search import get_universal_search
from app.core.auth_service import enforce_permission
from app.models.audit_models import AuditAction
from app.models.auth_models import Permission
from app.models.face_models import FaceMatch, FaceSearchResult, RegisteredIdentity
from app.models.schemas import (
    UniversalSearchEntityResult,
    UniversalSearchRequest,
    UniversalSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["universal_search"])


class FaceUniversalSearchResponse(BaseModel):
    """Response linking Biometric Face Search directly to Universal Search Entity Result."""
    status: str
    message: str
    face_count: int
    matched_person_id: Optional[str] = None
    matched_name: Optional[str] = None
    confidence_pct: Optional[float] = None
    similarity: Optional[float] = None
    face_matches: List[FaceMatch] = Field(default_factory=list)
    entity: Optional[UniversalSearchEntityResult] = None


@router.get("/search", response_model=UniversalSearchResponse)
def universal_search_get(
    q: str = Query(..., min_length=1, description="Search query string (Person ID, Name, Phone, Vehicle, Case ID, Account)"),
    limit: int = Query(25, ge=1, le=100)
):
    """
    Universal Search endpoint (GET).
    Supports:
    - Person ID (e.g. PERSON_TARIQ_AHMAD, P001)
    - Person Name (exact & partial matching)
    - Phone Number (normalized +91/raw digits)
    - Vehicle Number (normalized alphanumeric plate)
    - Case ID / Case Code (e.g. FIR-2024-311, CASE_FIR_2024_311)
    - Bank Account Number (e.g. ACC990188231, 990188231)
    """
    engine = get_universal_search()
    res = engine.search(query=q, limit=limit)
    audit_logger.log(
        action=AuditAction.SEARCH,
        resource_type="ENTITY",
        resource_id=q,
        status="SUCCESS",
        details={"query": q, "results_count": len(res.results)},
    )
    return res


@router.post("/search", response_model=UniversalSearchResponse)
def universal_search_post(request: UniversalSearchRequest):
    """
    Universal Search endpoint (POST JSON).
    """
    engine = get_universal_search()
    res = engine.search(query=request.query, limit=request.limit)
    audit_logger.log(
        action=AuditAction.SEARCH,
        resource_type="ENTITY",
        resource_id=request.query,
        status="SUCCESS",
        details={"query": request.query, "results_count": len(res.results)},
    )
    return res


@router.post("/search/face", response_model=FaceUniversalSearchResponse)
@router.post("/face/search", response_model=FaceUniversalSearchResponse)
async def search_by_face(
    file: UploadFile = File(..., description="Query face portrait image (PNG, JPG, JPEG, WEBP)"),
    threshold: float = Query(0.80, ge=0.0, le=1.0, description="Cosine similarity cutoff threshold"),
    _authorized_user=Depends(enforce_permission(Permission.FACE_SEARCH)),
):
    """
    Biometric Face Search -> Person ID -> Universal Search Profile -> Knowledge Graph.

    1. Receives face portrait image
    2. Matches against enrolled face database
    3. Resolves best matching Person ID
    4. Automatically queries Universal Search for that Person ID
    5. Returns unified profile with face confidence + full Knowledge Graph cases & connections!
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty (0 bytes)")

    storage = get_face_storage()
    search_engine = get_universal_search()

    try:
        face_result: FaceSearchResult = storage.search_face(
            query_image=file_bytes,
            threshold=threshold,
            raise_on_error=True
        )
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except NoFaceDetectedError as e:
        raise HTTPException(status_code=422, detail=f"No face detected: {e}")
    except MultipleFacesError as e:
        raise HTTPException(status_code=422, detail=f"Multiple faces detected: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face recognition error: {e}")

    # If match found
    if face_result.matches and face_result.matches[0].is_above_threshold:
        best_match = face_result.matches[0]
        matched_id = best_match.identity_id

        # Query Universal Search using the resolved Person ID
        u_res = search_engine.search(query=matched_id, limit=1)
        entity_res = u_res.results[0] if u_res.results else None

        audit_logger.log(
            action=AuditAction.FACE_SEARCH,
            resource_type="IDENTITY",
            resource_id=matched_id,
            status="SUCCESS",
            details={
                "matched_person_id": matched_id,
                "matched_name": best_match.name,
                "confidence_pct": best_match.confidence_pct,
                "similarity": best_match.similarity,
            },
        )

        return FaceUniversalSearchResponse(
            status="ok",
            message=f"Possible Match: '{best_match.name}' ({best_match.confidence_pct:.1f}% confidence, {best_match.similarity:.3f} similarity). Requires Investigator Verification. Decision-support indicator only, not proof of guilt.",
            face_count=face_result.face_count,
            matched_person_id=matched_id,
            matched_name=best_match.name,
            confidence_pct=best_match.confidence_pct,
            similarity=best_match.similarity,
            face_matches=face_result.matches,
            entity=entity_res
        )
    else:
        top_cand = face_result.matches[0] if face_result.matches else None
        top_name = top_cand.name if top_cand else "None"
        top_conf = top_cand.confidence_pct if top_cand else 0.0

        audit_logger.log(
            action=AuditAction.FACE_SEARCH,
            resource_type="IDENTITY",
            resource_id=top_cand.identity_id if top_cand else None,
            status="NO_MATCH",
            details={
                "top_candidate": top_name,
                "confidence_pct": top_conf,
            },
        )

        return FaceUniversalSearchResponse(
            status="no_match",
            message=f"No registered identity exceeded the {threshold * 100:.0f}% confidence threshold (top candidate: '{top_name}' at {top_conf}%). Requires Investigator Verification.",
            face_count=face_result.face_count,
            matched_person_id=None,
            matched_name=None,
            confidence_pct=top_conf,
            similarity=top_cand.similarity if top_cand else 0.0,
            face_matches=face_result.matches,
            entity=None
        )


@router.get("/face/identities")
def list_face_identities():
    """List all enrolled suspect identities in the demo face registry."""
    storage = get_face_storage()
    identities = storage.list_identities()
    return {
        "count": len(identities),
        "identities": [
            {
                "person_id": idn.person_id or idn.identity_id,
                "name": idn.name,
                "case_id": idn.case_id,
                "image_path": idn.image_path,
                "registration_timestamp": idn.registration_timestamp,
                "metadata": idn.metadata
            }
            for idn in identities
        ]
    }
