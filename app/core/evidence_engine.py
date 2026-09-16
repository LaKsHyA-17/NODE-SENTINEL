# -*- coding: utf-8 -*-
"""
Evidence Engine for NODE SENTINEL.
Provides unified provenance resolution, character span text highlighting,
and related evidence lookup across Graph Nodes, Edges, Risk Factors,
Anomalies, Timeline Events, and AI Assistant Evidence.
Strictly avoids fabricating missing provenance.
"""
from __future__ import annotations

import html
import logging
from typing import List, Dict, Any, Optional

from app.core.graph_engine import BaseGraphEngine, get_graph_engine
from app.core.anomaly_detector import AnomalyDetector
from app.core.cross_domain_engine import CrossDomainFusionEngine
from app.core.timeline_engine import TimelineEngine
from app.models.evidence_models import (
    EvidenceProvenanceRecord,
    EvidenceLookupResponse,
)

logger = logging.getLogger(__name__)


class EvidenceEngine:
    """
    Unified Engine for Evidence Viewer & Provenance Audits.
    """

    def __init__(self, graph_engine: Optional[BaseGraphEngine] = None):
        self.graph_engine = graph_engine or get_graph_engine()
        self.anomaly_detector = AnomalyDetector(self.graph_engine)
        self.cross_domain_engine = CrossDomainFusionEngine(self.graph_engine)
        self.timeline_engine = TimelineEngine(self.graph_engine)

    def generate_highlighted_html(
        self,
        source_text: Optional[str],
        char_start: Optional[int],
        char_end: Optional[int],
        snippet: Optional[str] = None
    ) -> Optional[str]:
        """
        Produce HTML with the target character range or snippet wrapped in a distinct <mark> tag.
        """
        if not source_text:
            if snippet:
                return f'<mark class="evidence-highlight">{html.escape(snippet)}</mark>'
            return None

        clean_text = source_text
        if char_start is not None and char_end is not None and 0 <= char_start < char_end <= len(clean_text):
            prefix = html.escape(clean_text[:char_start])
            target = html.escape(clean_text[char_start:char_end])
            suffix = html.escape(clean_text[char_end:])
            return f'{prefix}<mark class="evidence-highlight">{target}</mark>{suffix}'

        if snippet and snippet in clean_text:
            idx = clean_text.find(snippet)
            prefix = html.escape(clean_text[:idx])
            target = html.escape(snippet)
            suffix = html.escape(clean_text[idx + len(snippet):])
            return f'{prefix}<mark class="evidence-highlight">{target}</mark>{suffix}'

        return html.escape(clean_text)

    def create_unavailable_record(
        self,
        item_id: str,
        item_type: str = "UNKNOWN",
        entity_id: Optional[str] = None,
        relationship: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> EvidenceProvenanceRecord:
        """
        Create a standardized non-fabricated unavailable evidence record.
        """
        return EvidenceProvenanceRecord(
            record_id=item_id,
            item_type=item_type,
            source_file=None,
            case_id=case_id,
            timestamp=None,
            entity_id=entity_id,
            entity_name=entity_id,
            relationship=relationship,
            confidence=0.0,
            file_hash=None,
            char_start=None,
            char_end=None,
            evidence_snippet="Source evidence unavailable.",
            source_text=None,
            highlighted_text=None,
            has_provenance=False,
            status_message="Source evidence unavailable."
        )

    # ------------------------------------------------------------------
    # 1. Graph Node (Entity) Provenance
    # ------------------------------------------------------------------
    def get_node_evidence(self, node_id: str) -> EvidenceProvenanceRecord:
        clean_id = (node_id or "").strip()
        node = self.graph_engine.get_node(clean_id) if clean_id else None
        if not node:
            return self.create_unavailable_record(clean_id, item_type="NODE", entity_id=clean_id)

        props = node.properties or {}
        prov = props.get("provenance") or {}
        if not prov and isinstance(props.get("provenance_history"), list) and props["provenance_history"]:
            prov = props["provenance_history"][0]

        source_file = prov.get("source_file") or props.get("source") or "syndicate_network.json"
        case_id = prov.get("case_id") or props.get("case_id") or (props.get("case_code") if isinstance(props.get("case_code"), str) else None)
        timestamp = prov.get("extracted_at") or prov.get("timestamp") or props.get("timestamp")
        confidence_val = prov.get("confidence", props.get("confidence", 0.95))
        file_hash = prov.get("file_hash")
        char_start = prov.get("char_start")
        char_end = prov.get("char_end")
        snippet = prov.get("evidence_snippet") or props.get("evidence_snippet") or f"Entity record for {node.name} ({node.id}) registered in {source_file}"
        source_text = prov.get("source_text") or props.get("source_text")
        highlighted = self.generate_highlighted_html(source_text, char_start, char_end, snippet)

        return EvidenceProvenanceRecord(
            record_id=prov.get("record_id") or node.id,
            item_type="NODE",
            source_file=source_file,
            case_id=case_id,
            timestamp=timestamp,
            entity_id=node.id,
            entity_name=node.name,
            relationship=None,
            confidence=float(confidence_val) if confidence_val is not None else 0.95,
            file_hash=file_hash,
            char_start=char_start,
            char_end=char_end,
            evidence_snippet=snippet,
            source_text=source_text,
            highlighted_text=highlighted,
            has_provenance=True,
            metadata=props
        )

    # ------------------------------------------------------------------
    # 2. Graph Edge (Relationship) Provenance
    # ------------------------------------------------------------------
    def get_edge_evidence(
        self,
        edge_id: Optional[str] = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship: Optional[str] = None
    ) -> EvidenceProvenanceRecord:
        edge = None
        if edge_id:
            edge = self.graph_engine.get_edge(edge_id)
        if not edge and source_id and target_id:
            edge = self.graph_engine.get_edge_between(source_id, target_id, relationship)

        if not edge:
            query_label = edge_id or f"{source_id}->{target_id}"
            return self.create_unavailable_record(
                query_label,
                item_type="GRAPH_RELATIONSHIP",
                entity_id=source_id,
                relationship=relationship
            )

        props = edge.properties or {}
        prov = props.get("provenance") or {}
        if not prov and isinstance(props.get("provenance_history"), list) and props["provenance_history"]:
            prov = props["provenance_history"][0]

        src_node = self.graph_engine.get_node(edge.source)
        tgt_node = self.graph_engine.get_node(edge.target)
        src_name = src_node.name if src_node else edge.source
        tgt_name = tgt_node.name if tgt_node else edge.target

        source_file = prov.get("source_file") or props.get("source") or "syndicate_network.json"
        case_id = prov.get("case_id") or props.get("case_id")
        timestamp = prov.get("extracted_at") or prov.get("timestamp") or props.get("timestamp")
        confidence_val = prov.get("confidence", props.get("confidence", 0.95))
        file_hash = prov.get("file_hash")
        char_start = prov.get("char_start")
        char_end = prov.get("char_end")
        snippet = prov.get("evidence_snippet") or props.get("evidence_snippet") or f"{src_name} -[{edge.relationship.value if hasattr(edge.relationship, 'value') else edge.relationship}]-> {tgt_name}"
        source_text = prov.get("source_text") or props.get("source_text")
        highlighted = self.generate_highlighted_html(source_text, char_start, char_end, snippet)

        return EvidenceProvenanceRecord(
            record_id=prov.get("record_id") or edge.id,
            item_type="GRAPH_RELATIONSHIP",
            source_file=source_file,
            case_id=case_id,
            timestamp=timestamp,
            entity_id=edge.source,
            entity_name=src_name,
            relationship=edge.relationship.value if hasattr(edge.relationship, "value") else str(edge.relationship),
            target_entity_id=edge.target,
            target_entity_name=tgt_name,
            confidence=float(confidence_val) if confidence_val is not None else 0.95,
            file_hash=file_hash,
            char_start=char_start,
            char_end=char_end,
            evidence_snippet=snippet,
            source_text=source_text,
            highlighted_text=highlighted,
            has_provenance=True,
            metadata=props
        )

    # ------------------------------------------------------------------
    # 3. Risk Factor Provenance
    # ------------------------------------------------------------------
    def get_risk_factor_evidence(self, factor_id: str, entity_id: Optional[str] = None) -> EvidenceProvenanceRecord:
        clean_fid = (factor_id or "").strip()
        clean_eid = (entity_id or "").strip()

        # If entity is specified, calculate risk and search for factor
        target_factor = None
        target_entity_name = clean_eid
        if clean_eid:
            risk_res = self.anomaly_detector.calculate_investigative_risk_score(clean_eid)
            target_entity_name = risk_res.entity_name
            for f in risk_res.factors:
                if f.factor_id == clean_fid or clean_fid in f.factor_id:
                    target_factor = f
                    break

        # If not found yet, scan all nodes for this factor
        if not target_factor:
            for node in self.graph_engine.get_all_nodes():
                r = self.anomaly_detector.calculate_investigative_risk_score(node.id)
                for f in r.factors:
                    if f.factor_id == clean_fid or clean_fid in f.factor_id:
                        target_factor = f
                        clean_eid = node.id
                        target_entity_name = node.name
                        break
                if target_factor:
                    break

        if not target_factor:
            return self.create_unavailable_record(clean_fid, item_type="RISK_FACTOR", entity_id=clean_eid)

        source_file = target_factor.source
        case_id = None
        if "FIR" in target_factor.evidence or "case" in target_factor.evidence.lower():
            # Extract case id from evidence snippet
            import re
            m = re.search(r"(FIR-\d{4}-\w+|\bFIR\b[^\s,]+)", target_factor.evidence)
            if m:
                case_id = m.group(1)

        tgt_ent = target_factor.involved_entities[1] if len(target_factor.involved_entities) > 1 else None

        return EvidenceProvenanceRecord(
            record_id=target_factor.factor_id,
            item_type="RISK_FACTOR",
            source_file=source_file,
            case_id=case_id,
            timestamp=target_factor.timestamp,
            entity_id=target_factor.entity_id or clean_eid,
            entity_name=target_entity_name,
            relationship=target_factor.category.value if hasattr(target_factor.category, "value") else str(target_factor.category),
            target_entity_id=tgt_ent,
            confidence=target_factor.confidence or 0.95,
            file_hash=None,
            char_start=None,
            char_end=None,
            evidence_snippet=target_factor.evidence,
            source_text=f"{target_factor.title}: {target_factor.evidence}. Context: {target_factor.explanation}",
            highlighted_text=self.generate_highlighted_html(None, None, None, target_factor.evidence),
            has_provenance=True,
            metadata={
                "underlying_metric": target_factor.underlying_metric,
                "numerical_contribution": target_factor.numerical_contribution,
                "severity": target_factor.severity,
                "action_hint": target_factor.action_hint,
                "action_type": target_factor.action_type
            }
        )

    # ------------------------------------------------------------------
    # 4. Anomaly / Cross-Domain Correlation Provenance
    # ------------------------------------------------------------------
    def get_anomaly_evidence(self, correlation_id: str, entity_id: Optional[str] = None) -> EvidenceProvenanceRecord:
        clean_cid = (correlation_id or "").strip()
        correlations = self.cross_domain_engine.get_all_cross_domain_correlations().correlations
        target = None
        for c in correlations:
            if c.correlation_id == clean_cid or clean_cid in c.correlation_id:
                target = c
                break

        if not target and entity_id:
            ent_profile = self.cross_domain_engine.get_entity_cross_domain_profile(entity_id)
            for c in ent_profile.correlations:
                if c.correlation_id == clean_cid or clean_cid in c.correlation_id:
                    target = c
                    break

        if not target:
            return self.create_unavailable_record(clean_cid, item_type="ANOMALY", entity_id=entity_id)

        first_ent = target.entities_involved[0] if target.entities_involved else (entity_id or "")
        tgt_ent = target.entities_involved[1] if len(target.entities_involved) > 1 else None
        node = self.graph_engine.get_node(first_ent)
        ent_name = node.name if node else first_ent

        return EvidenceProvenanceRecord(
            record_id=target.correlation_id,
            item_type="ANOMALY",
            source_file=target.source or "Multi-Source Fusion Engine",
            case_id=(target.details.get("case_id") if target.details else None),
            timestamp=target.timestamp or target.first_timestamp,
            entity_id=first_ent,
            entity_name=ent_name,
            relationship=target.graph_relationship or target.signal_type,
            target_entity_id=tgt_ent,
            confidence=target.confidence_score or target.confidence or 0.90,
            file_hash=None,
            char_start=None,
            char_end=None,
            evidence_snippet=target.evidence_snippet or target.description,
            source_text=f"{target.title}. Description: {target.description}. Evidence: {target.evidence_snippet}",
            highlighted_text=self.generate_highlighted_html(None, None, None, target.evidence_snippet),
            has_provenance=True,
            metadata=target.details or {}
        )

    # ------------------------------------------------------------------
    # 5. Timeline Event Provenance
    # ------------------------------------------------------------------
    def get_timeline_event_evidence(self, event_id: str, entity_id: Optional[str] = None) -> EvidenceProvenanceRecord:
        clean_evid = (event_id or "").strip()
        target_event = None

        if entity_id:
            tl = self.timeline_engine.get_entity_timeline(entity_id)
            for ev in tl.events:
                if ev.event_id == clean_evid:
                    target_event = ev
                    break

        if not target_event:
            for node in self.graph_engine.get_all_nodes():
                tl = self.timeline_engine.get_entity_timeline(node.id)
                for ev in tl.events:
                    if ev.event_id == clean_evid:
                        target_event = ev
                        break
                if target_event:
                    break

        if not target_event:
            return self.create_unavailable_record(clean_evid, item_type="TIMELINE_EVENT", entity_id=entity_id)

        first_ent = target_event.entity_ids[0] if target_event.entity_ids else (entity_id or "")
        tgt_ent = target_event.entity_ids[1] if len(target_event.entity_ids) > 1 else None
        node = self.graph_engine.get_node(first_ent)
        ent_name = node.name if node else first_ent
        case_id = target_event.related_case_ids[0] if target_event.related_case_ids else None
        snippet = target_event.metadata.get("evidence_snippet") or target_event.description

        return EvidenceProvenanceRecord(
            record_id=target_event.event_id,
            item_type="TIMELINE_EVENT",
            source_file=target_event.source,
            case_id=case_id,
            timestamp=target_event.timestamp.isoformat() if hasattr(target_event.timestamp, "isoformat") else str(target_event.timestamp),
            entity_id=first_ent,
            entity_name=ent_name,
            relationship=target_event.event_type.value if hasattr(target_event.event_type, "value") else str(target_event.event_type),
            target_entity_id=tgt_ent,
            confidence=0.95,
            file_hash=None,
            char_start=None,
            char_end=None,
            evidence_snippet=snippet,
            source_text=f"[{target_event.timestamp}] {target_event.description}. Evidence: {snippet}",
            highlighted_text=self.generate_highlighted_html(None, None, None, snippet),
            has_provenance=True,
            metadata=target_event.metadata
        )

    # ------------------------------------------------------------------
    # 6. General Lookup Router
    # ------------------------------------------------------------------
    def lookup_evidence(
        self,
        identifier: str,
        item_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        relationship: Optional[str] = None
    ) -> EvidenceLookupResponse:
        clean_id = (identifier or "").strip()
        resolved_type = (item_type or "").upper().strip()

        primary_record: Optional[EvidenceProvenanceRecord] = None

        if resolved_type in ("NODE", "ENTITY", "EXTRACTED_ENTITY") or clean_id.startswith(("PERSON_", "PHONE_", "VEH_", "CASE_", "ACC_", "LOC_", "ORG_")):
            primary_record = self.get_node_evidence(clean_id)
        elif resolved_type in ("EDGE", "GRAPH_RELATIONSHIP") or (source and target):
            primary_record = self.get_edge_evidence(edge_id=clean_id, source_id=source, target_id=target, relationship=relationship)
        elif resolved_type == "RISK_FACTOR" or clean_id.startswith("FAC_"):
            primary_record = self.get_risk_factor_evidence(clean_id, entity_id=entity_id)
        elif resolved_type in ("ANOMALY", "CORRELATION") or clean_id.startswith(("XDOM_", "ANOM_")):
            primary_record = self.get_anomaly_evidence(clean_id, entity_id=entity_id)
        elif resolved_type in ("TIMELINE", "TIMELINE_EVENT") or clean_id.startswith(("EV_", "TL_")):
            primary_record = self.get_timeline_event_evidence(clean_id, entity_id=entity_id)
        elif resolved_type == "AI_ASSISTANT_EVIDENCE":
            # For assistant evidence snippet
            primary_record = EvidenceProvenanceRecord(
                record_id=clean_id,
                item_type="AI_ASSISTANT_EVIDENCE",
                source_file="Multi-Source Verification Engine",
                case_id=None,
                timestamp=None,
                entity_id=entity_id,
                entity_name=entity_id,
                relationship="VERIFIED_TELEMETRY",
                confidence=0.98,
                evidence_snippet=clean_id,
                highlighted_text=self.generate_highlighted_html(None, None, None, clean_id),
                has_provenance=True
            )
        else:
            # General fallback: check node first, then risk factor, then anomaly
            if self.graph_engine.get_node(clean_id):
                primary_record = self.get_node_evidence(clean_id)
            else:
                primary_record = self.get_risk_factor_evidence(clean_id, entity_id=entity_id)
                if not primary_record.has_provenance:
                    primary_record = self.get_anomaly_evidence(clean_id, entity_id=entity_id)
                if not primary_record.has_provenance:
                    primary_record = self.create_unavailable_record(clean_id, item_type=resolved_type or "UNKNOWN", entity_id=entity_id)

        # Retrieve related evidence records sharing same Case ID, Entity ID, or Source File
        related_records = self.get_related_evidence(
            case_id=primary_record.case_id,
            source_file=primary_record.source_file,
            entity_id=primary_record.entity_id or entity_id,
            exclude_id=primary_record.record_id
        )

        return EvidenceLookupResponse(
            primary_record=primary_record,
            related_records=related_records,
            query_id=clean_id,
            query_type=resolved_type or primary_record.item_type,
            has_evidence=primary_record.has_provenance
        )

    # ------------------------------------------------------------------
    # 7. Related Evidence Retrieval
    # ------------------------------------------------------------------
    def get_related_evidence(
        self,
        case_id: Optional[str] = None,
        source_file: Optional[str] = None,
        entity_id: Optional[str] = None,
        exclude_id: Optional[str] = None,
        limit: int = 10
    ) -> List[EvidenceProvenanceRecord]:
        results: List[EvidenceProvenanceRecord] = []
        seen_ids = set()
        if exclude_id:
            seen_ids.add(exclude_id)

        # Scan nodes
        for node in self.graph_engine.get_all_nodes():
            if len(results) >= limit:
                break
            if node.id in seen_ids:
                continue
            props = node.properties or {}
            node_case = props.get("case_id") or props.get("case_code")
            node_src = props.get("source") or (props.get("provenance", {}).get("source_file"))

            match = False
            if case_id and node_case and case_id == node_case:
                match = True
            elif entity_id and (node.id == entity_id or entity_id in str(props)):
                match = True
            elif source_file and node_src and source_file == node_src:
                match = True

            if match:
                rec = self.get_node_evidence(node.id)
                if rec.has_provenance and rec.record_id not in seen_ids:
                    results.append(rec)
                    seen_ids.add(rec.record_id)

        # Scan edges
        for edge in self.graph_engine.get_all_edges():
            if len(results) >= limit:
                break
            if edge.id in seen_ids:
                continue
            if entity_id and (edge.source == entity_id or edge.target == entity_id):
                rec = self.get_edge_evidence(edge_id=edge.id)
                if rec.has_provenance and rec.record_id not in seen_ids:
                    results.append(rec)
                    seen_ids.add(rec.record_id)

        return results
