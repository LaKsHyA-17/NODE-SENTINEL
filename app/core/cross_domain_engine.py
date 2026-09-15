# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Cross-Domain Telemetry Fusion Engine
Correlates multi-source temporal occurrences between Call Detail Records (CDR),
Financial Ledger Transfers, FIR Police Case Filings, and Geographic Surveillance.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.cdr_analytics import get_cdr_storage
from app.core.face_storage import get_face_storage
from app.core.financial_analytics import get_financial_storage
from app.core.graph_engine import BaseGraphEngine, get_graph_engine
from app.models.cross_domain_models import (
    CrossDomainCorrelationItem,
    CrossDomainSummaryResponse,
    EntityCrossDomainProfile,
)
from app.models.graph_models import EdgeType, NodeType

logger = logging.getLogger(__name__)


def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    s = str(val).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


class CrossDomainFusionEngine:
    """
    Fuses telemetry across telephony, banking, location, and case registries.
    """

    def __init__(self, graph_engine: Optional[BaseGraphEngine] = None):
        self.graph = graph_engine or get_graph_engine()
        self.cdr_storage = get_cdr_storage()
        self.fin_storage = get_financial_storage()
        self.face_storage = get_face_storage()

    def _resolve_entity_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Build mapping of Person ID -> { 'phones': set(), 'accounts': set(), 'name': str, 'cases': set() }
        """
        mapping: Dict[str, Dict[str, Any]] = {}
        all_nodes = self.graph.get_all_nodes()

        for n in all_nodes:
            lbl = n.label.value if hasattr(n.label, "value") else str(n.label)
            if lbl == NodeType.PERSON.value or n.id.startswith("PERSON_"):
                mapping[n.id] = {
                    "name": n.name or n.id,
                    "phones": set(),
                    "accounts": set(),
                    "cases": set(),
                }
                if "phone_number" in n.properties:
                    mapping[n.id]["phones"].add(str(n.properties["phone_number"]))
                if "account_number" in n.properties:
                    mapping[n.id]["accounts"].add(str(n.properties["account_number"]))

        all_edges = self.graph.get_all_edges()
        for e in all_edges:
            rel = e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship)
            src_node = self.graph.get_node(e.source)
            tgt_node = self.graph.get_node(e.target)
            if not src_node or not tgt_node:
                continue

            if e.source in mapping:
                t_lbl = tgt_node.label.value if hasattr(tgt_node.label, "value") else str(tgt_node.label)
                if t_lbl == NodeType.PHONE.value or tgt_node.id.startswith("PHONE_"):
                    p_num = tgt_node.properties.get("phone_number") or tgt_node.name or tgt_node.id
                    mapping[e.source]["phones"].add(str(p_num))
                elif t_lbl == NodeType.BANK_ACCOUNT.value or tgt_node.id.startswith("ACC_"):
                    a_num = tgt_node.properties.get("account_number") or tgt_node.name or tgt_node.id
                    mapping[e.source]["accounts"].add(str(a_num))
                elif t_lbl == NodeType.CASE.value or tgt_node.id.startswith("CASE_"):
                    mapping[e.source]["cases"].add(tgt_node.id)

        return mapping

    def detect_call_to_transfer_correlations(
        self, max_latency_minutes: float = 60.0
    ) -> List[CrossDomainCorrelationItem]:
        """
        Identify coordinated events where Subject A calls Subject B, and within `max_latency_minutes`,
        a financial transfer occurs between their accounts.
        """
        entity_map = self._resolve_entity_mappings()
        cdr_records = self.cdr_storage.get_all_records()
        fin_records = self.fin_storage.get_all_records()

        if not cdr_records or not fin_records:
            return []

        # Invert phone->person and account->person
        phone_to_person = {}
        account_to_person = {}
        for pid, data in entity_map.items():
            for p in data["phones"]:
                digits = "".join(c for c in p if c.isdigit())[-10:]
                phone_to_person[digits] = pid
                phone_to_person[p] = pid
            for a in data["accounts"]:
                account_to_person[a] = pid
                account_to_person[a.replace("ACC_", "")] = pid

        correlations: List[CrossDomainCorrelationItem] = []
        corr_idx = 1

        for call in cdr_records:
            c_time = _parse_ts(call.timestamp)
            if not c_time:
                continue

            c_caller_digits = "".join(c for c in call.caller if c.isdigit())[-10:]
            c_rec_digits = "".join(c for c in call.receiver if c.isdigit())[-10:]

            p1 = phone_to_person.get(c_caller_digits, phone_to_person.get(call.caller))
            p2 = phone_to_person.get(c_rec_digits, phone_to_person.get(call.receiver))

            if not p1 or not p2 or p1 == p2:
                continue

            p1_name = entity_map[p1]["name"]
            p2_name = entity_map[p2]["name"]

            # Search financial records in window [c_time - 10m, c_time + max_latency]
            for tx in fin_records:
                tx_time = _parse_ts(tx.timestamp)
                if not tx_time:
                    continue

                diff_seconds = (tx_time - c_time).total_seconds()
                diff_minutes = diff_seconds / 60.0

                if 0 <= diff_minutes <= max_latency_minutes:
                    tx_sender_pid = account_to_person.get(tx.sender, account_to_person.get(tx.sender.replace("ACC_", "")))
                    tx_rec_pid = account_to_person.get(tx.receiver, account_to_person.get(tx.receiver.replace("ACC_", "")))

                    if {tx_sender_pid, tx_rec_pid} == {p1, p2}:
                        amt = getattr(tx, "amount", 0.0)
                        sev = "CRITICAL" if amt >= 500000.0 or diff_minutes <= 15.0 else "HIGH"
                        conf = 0.96 if diff_minutes <= 15.0 else 0.90

                        correlations.append(
                            CrossDomainCorrelationItem(
                                correlation_id=f"CORR_CALL_TX_{corr_idx}",
                                correlation_type="CALL_TO_TRANSFER",
                                severity=sev,
                                title="Coordinated Call-to-Transfer Telemetry Spike",
                                description=(
                                    f"Telephony call between {p1_name} and {p2_name} was followed {diff_minutes:.1f} minutes later "
                                    f"by a bank transfer of ₹{amt:,.2f} between their linked accounts."
                                ),
                                entities_involved=[p1, p2, tx.sender, tx.receiver],
                                time_window_minutes=round(diff_minutes, 1),
                                first_timestamp=c_time.isoformat(),
                                second_timestamp=tx_time.isoformat(),
                                evidence_channels=["CDR_TELEPHONY", "BANK_LEDGER"],
                                confidence_score=conf,
                                details={
                                    "caller": call.caller,
                                    "receiver": call.receiver,
                                    "call_duration_seconds": call.duration_seconds,
                                    "transaction_id": tx.transaction_id,
                                    "amount": amt,
                                    "transfer_sender": tx.sender,
                                    "transfer_receiver": tx.receiver,
                                },
                                recommended_action="Inspect linked bank accounts and request cell tower triangulation.",
                            )
                        )
                        corr_idx += 1

        return correlations

    def detect_cotemporal_colocation_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify multi-subject co-presence at surveillance locations correlated with synchronous calls.
        """
        all_edges = self.graph.get_all_edges()
        loc_edges = [
            e for e in all_edges
            if (e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship)) == EdgeType.LOCATED_AT.value
        ]

        loc_to_entities: Dict[str, List[str]] = {}
        for e in loc_edges:
            loc_to_entities.setdefault(e.target, []).append(e.source)

        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1

        for loc_id, subjects in loc_to_entities.items():
            unique_sub = list(set(subjects))
            if len(unique_sub) >= 2:
                loc_node = self.graph.get_node(loc_id)
                loc_name = loc_node.name if loc_node else loc_id

                names = []
                for s in unique_sub:
                    sn = self.graph.get_node(s)
                    names.append(sn.name if sn else s)

                correlations.append(
                    CrossDomainCorrelationItem(
                        correlation_id=f"CORR_COLOC_{c_idx}",
                        correlation_type="CO_TEMPORAL_PRESENCE",
                        severity="HIGH" if len(unique_sub) >= 3 else "ELEVATED",
                        title="Multi-Subject Hotspot Co-Presence",
                        description=f"{len(unique_sub)} key subjects ({', '.join(names[:3])}) co-located at surveillance hotspot '{loc_name}'.",
                        entities_involved=[loc_id] + unique_sub,
                        time_window_minutes=0.0,
                        first_timestamp=datetime.now().isoformat(),
                        second_timestamp=None,
                        evidence_channels=["LOCATION_SURVEILLANCE", "KNOWLEDGE_GRAPH"],
                        confidence_score=0.92,
                        details={"location_id": loc_id, "location_name": loc_name, "subjects": names},
                        recommended_action="Cross-reference CCTV stills with Face Search Registry.",
                    )
                )
                c_idx += 1

        return correlations

    def get_all_cross_domain_correlations(self) -> CrossDomainSummaryResponse:
        """Aggregate all fused cross-domain correlations across system."""
        c2t = self.detect_call_to_transfer_correlations()
        coloc = self.detect_cotemporal_colocation_correlations()
        all_corrs = c2t + coloc

        crit_count = sum(1 for c in all_corrs if c.severity == "CRITICAL")
        high_count = sum(1 for c in all_corrs if c.severity in ("HIGH", "CRITICAL"))

        return CrossDomainSummaryResponse(
            total_correlations=len(all_corrs),
            critical_count=crit_count,
            high_count=high_count,
            correlations=all_corrs,
            channels_monitored=[
                "CDR_TELEPHONY",
                "BANK_LEDGER",
                "LOCATION_SURVEILLANCE",
                "POLICE_FIR_REGISTRY",
                "BIOMETRIC_FACE_DATABASE",
            ],
        )

    def get_entity_cross_domain_profile(self, entity_id: str) -> EntityCrossDomainProfile:
        """Get cross-domain correlation profile for a specific entity."""
        clean_id = entity_id.strip()
        all_corrs = self.get_all_cross_domain_correlations().correlations
        node = self.graph.get_node(clean_id)

        matching = [c for c in all_corrs if clean_id in c.entities_involved]

        channels = set()
        for c in matching:
            for ch in c.evidence_channels:
                channels.add(ch)

        highest_sev = "LOW"
        if any(c.severity == "CRITICAL" for c in matching):
            highest_sev = "CRITICAL"
        elif any(c.severity == "HIGH" for c in matching):
            highest_sev = "HIGH"
        elif any(c.severity == "ELEVATED" for c in matching):
            highest_sev = "ELEVATED"
        elif matching:
            highest_sev = "NOTICE"

        risk_index = min(100.0, float(len(matching) * 25.0 + (30.0 if highest_sev == "CRITICAL" else 0.0)))

        return EntityCrossDomainProfile(
            entity_id=clean_id,
            entity_name=node.name if node else clean_id,
            entity_type=node.label.value if (node and hasattr(node.label, "value")) else "Unknown",
            total_fused_events=len(matching),
            highest_severity=highest_sev,
            correlations=matching,
            associated_channels=list(channels),
            multi_channel_risk_index=round(risk_index, 1),
        )


# Singleton
_fusion_instance: Optional[CrossDomainFusionEngine] = None


def get_cross_domain_fusion_engine() -> CrossDomainFusionEngine:
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = CrossDomainFusionEngine()
    return _fusion_instance
