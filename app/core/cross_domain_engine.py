# -*- coding: utf-8 -*-
"""
NODE SENTINEL - Cross-Domain Telemetry Fusion Engine
Correlates multi-source temporal occurrences between:
- Police FIR / Case Registries
- Call Detail Records (CDR Telephony)
- Financial Ledger Transactions
- Knowledge Graph Topological Pathways
- Geographic Surveillance Locations & Cell Towers
- Chronological Timelines
- Biometric Facial Identity Registry

Adheres to strict neutral decision-support terminology ("correlated activity", "observed relationship", "potentially relevant pattern").
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
    CaseCrossDomainProfile,
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
        return val.replace(tzinfo=None) if val.tzinfo is not None else val
    s = str(val).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
        except ValueError:
            continue
    return None


class CrossDomainFusionEngine:
    """
    Fuses multi-source telemetry across telephony, banking, location, case registries,
    knowledge graph topology, and biometric face matches.
    """

    def __init__(self, graph_engine: Optional[BaseGraphEngine] = None):
        self.graph = graph_engine or get_graph_engine()
        self.cdr_storage = get_cdr_storage()
        self.fin_storage = get_financial_storage()
        self.face_storage = get_face_storage()

    def _resolve_entity_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Build comprehensive mapping of:
        Person ID -> { 'phones': set(), 'accounts': set(), 'name': str, 'cases': set(), 'vehicles': set() }
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
                    "vehicles": set(),
                }
                if "phone_number" in n.properties:
                    mapping[n.id]["phones"].add(str(n.properties["phone_number"]))
                if "account_number" in n.properties:
                    mapping[n.id]["accounts"].add(str(n.properties["account_number"]))
                if "case_id" in n.properties:
                    mapping[n.id]["cases"].add(str(n.properties["case_id"]))
                if "registration" in n.properties:
                    mapping[n.id]["vehicles"].add(str(n.properties["registration"]))

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
                elif t_lbl == NodeType.VEHICLE.value or tgt_node.id.startswith("VEH_"):
                    mapping[e.source]["vehicles"].add(tgt_node.id)

        return mapping

    # -----------------------------------------------------------------------
    # 1. CDR + Financial Coordination Detector
    # -----------------------------------------------------------------------

    def detect_call_to_transfer_correlations(
        self, max_latency_minutes: float = 60.0
    ) -> List[CrossDomainCorrelationItem]:
        """
        Identify coordinated multi-channel events where Subject A calls Subject B, and within `max_latency_minutes`,
        a financial transfer occurs between their linked accounts.
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
                if digits:
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

            # Search financial records in window [c_time - 5m, c_time + max_latency]
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

                        snippet = (
                            f"Telephony call from {call.caller} ({p1_name}) to {call.receiver} ({p2_name}) "
                            f"[Duration: {call.duration_seconds}s] at {c_time.strftime('%Y-%m-%d %H:%M:%S')} was followed "
                            f"{diff_minutes:.1f} minutes later by bank transfer of ₹{amt:,.2f} from {tx.sender} to {tx.receiver} (TxID: {tx.transaction_id})."
                        )

                        correlations.append(
                            CrossDomainCorrelationItem(
                                correlation_id=f"CORR_CALL_TX_{corr_idx}",
                                correlation_type="CALL_TO_TRANSFER",
                                signal_type="CDR + Financial Coordination",
                                severity=sev,
                                title="Coordinated Call-to-Transfer Telemetry Pattern",
                                description=(
                                    f"Correlated activity: Observed telephony call between {p1_name} and {p2_name} "
                                    f"followed {diff_minutes:.1f} minutes later by financial transfer of ₹{amt:,.2f} between linked accounts."
                                ),
                                entities_involved=[p1, p2, tx.sender, tx.receiver, call.caller, call.receiver],
                                time_window_minutes=round(diff_minutes, 1),
                                first_timestamp=c_time.isoformat(),
                                second_timestamp=tx_time.isoformat(),
                                timestamp=c_time.isoformat(),
                                source="CDR Telephony Logs · Bank Ledger",
                                evidence_channels=["CDR_TELEPHONY", "BANK_LEDGER"],
                                confidence_score=conf,
                                confidence=conf,
                                evidence_snippet=snippet,
                                graph_relationship="CALLS → TRANSFERRED_MONEY",
                                timeline_link={
                                    "entity_id": p1,
                                    "event_type": "CALL",
                                    "timestamp": c_time.isoformat(),
                                    "date_filter": c_time.strftime("%Y-%m-%d"),
                                },
                                source_evidence_link={
                                    "entity_id": p1,
                                    "channel": "CDR_AND_BANKING",
                                    "transaction_id": tx.transaction_id,
                                    "call_id": getattr(call, "call_id", "CDR_LOG"),
                                },
                                details={
                                    "caller": call.caller,
                                    "receiver": call.receiver,
                                    "call_duration_seconds": call.duration_seconds,
                                    "transaction_id": tx.transaction_id,
                                    "amount": amt,
                                    "transfer_sender": tx.sender,
                                    "transfer_receiver": tx.receiver,
                                },
                                recommended_action="Cross-reference linked bank statements with cell tower subscriber registration.",
                            )
                        )
                        corr_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 2. Location Surveillance Hotspot & Co-Presence Detector
    # -----------------------------------------------------------------------

    def detect_cotemporal_colocation_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify multi-subject co-presence at surveillance locations correlated with synchronous activity.
        """
        all_edges = self.graph.get_all_edges()
        loc_edges = [
            e
            for e in all_edges
            if (e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship))
            == EdgeType.LOCATED_AT.value
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

                now_iso = datetime.now().isoformat()
                snippet = (
                    f"Surveillance location '{loc_name}' ({loc_id}) documents co-presence of "
                    f"{len(unique_sub)} active entities: {', '.join(names[:4])}."
                )

                correlations.append(
                    CrossDomainCorrelationItem(
                        correlation_id=f"CORR_COLOC_{c_idx}",
                        correlation_type="CO_TEMPORAL_PRESENCE",
                        signal_type="Location Surveillance Hotspot",
                        severity="HIGH" if len(unique_sub) >= 3 else "ELEVATED",
                        title="Multi-Subject Hotspot Co-Presence",
                        description=f"Observed relationship: {len(unique_sub)} key subjects ({', '.join(names[:3])}) share documented presence at surveillance hotspot '{loc_name}'.",
                        entities_involved=[loc_id] + unique_sub,
                        time_window_minutes=0.0,
                        first_timestamp=now_iso,
                        second_timestamp=None,
                        timestamp=now_iso,
                        source="Location Surveillance · Knowledge Graph",
                        evidence_channels=["LOCATION_SURVEILLANCE", "KNOWLEDGE_GRAPH"],
                        confidence_score=0.92,
                        confidence=0.92,
                        evidence_snippet=snippet,
                        graph_relationship="LOCATED_AT",
                        timeline_link={
                            "entity_id": unique_sub[0],
                            "event_type": "LOCATION",
                            "location": loc_name,
                        },
                        source_evidence_link={
                            "entity_id": loc_id,
                            "channel": "LOCATION_SURVEILLANCE",
                        },
                        details={"location_id": loc_id, "location_name": loc_name, "subjects": names},
                        recommended_action="Cross-reference CCTV surveillance captures with Face Search Registry.",
                    )
                )
                c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 3. Police FIR & Case Record Association Detector
    # -----------------------------------------------------------------------

    def detect_fir_case_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify multi-entity co-involvement and operational asset linkages documented in Police FIR / Case records.
        """
        all_nodes = self.graph.get_all_nodes()
        case_nodes = [
            n for n in all_nodes
            if (n.label.value if hasattr(n.label, "value") else str(n.label)) in ("CASE", NodeType.CASE.value)
            or n.id.startswith("CASE_")
        ]

        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1

        for c_node in case_nodes:
            nbrs = self.graph.get_neighbors(c_node.id, depth=1)
            nbr_nodes = nbrs.get("nodes", [])

            linked_persons = []
            linked_vehicles = []
            linked_accounts = []
            linked_phones = []

            for n in nbr_nodes:
                if n.id == c_node.id:
                    continue
                lbl = n.label.value if hasattr(n.label, "value") else str(n.label)
                if lbl in ("PERSON", NodeType.PERSON.value) or n.id.startswith("PERSON_"):
                    linked_persons.append(n)
                elif lbl in ("VEHICLE", NodeType.VEHICLE.value) or n.id.startswith("VEH_"):
                    linked_vehicles.append(n)
                elif lbl in ("BANK_ACCOUNT", "BANKACCOUNT", NodeType.BANK_ACCOUNT.value) or n.id.startswith("ACC_"):
                    linked_accounts.append(n)
                elif lbl in ("PHONE", NodeType.PHONE.value) or n.id.startswith("PHONE_"):
                    linked_phones.append(n)

            if linked_persons:
                case_code = c_node.name or c_node.id
                case_props = c_node.properties or {}
                raw_snippet = case_props.get("evidence", case_props.get("description", f"Documented case record {case_code}"))

                involved_ids = [c_node.id] + [p.id for p in linked_persons] + [v.id for v in linked_vehicles] + [a.id for a in linked_accounts]
                p_names = [p.name for p in linked_persons]

                ts = case_props.get("filing_date", case_props.get("date", datetime.now().isoformat()))
                ts_parsed = _parse_ts(ts)
                ts_str = ts_parsed.isoformat() if ts_parsed else datetime.now().isoformat()

                snippet = (
                    f"Police FIR / Case filing {case_code} documents co-involvement of {len(linked_persons)} subject(s) "
                    f"({', '.join(p_names[:3])})"
                    + (f" with linked vehicle(s) {', '.join([v.name for v in linked_vehicles])}" if linked_vehicles else "")
                    + (f" and account(s) {', '.join([a.name for a in linked_accounts])}" if linked_accounts else "")
                    + f". Evidentiary note: '{raw_snippet}'."
                )

                sev = "CRITICAL" if len(linked_persons) >= 2 or len(linked_vehicles) > 0 else "HIGH"

                correlations.append(
                    CrossDomainCorrelationItem(
                        correlation_id=f"CORR_CASE_{c_idx}",
                        correlation_type="FIR_CASE_ASSOCIATION",
                        signal_type="FIR Case Association",
                        severity=sev,
                        title=f"Case Association & Asset Co-Involvement ({case_code})",
                        description=(
                            f"Potentially relevant pattern: Police record {case_code} establishes multi-entity co-involvement "
                            f"across {len(linked_persons)} subject(s) and {len(linked_vehicles) + len(linked_accounts)} linked operational asset(s)."
                        ),
                        entities_involved=involved_ids,
                        time_window_minutes=None,
                        first_timestamp=ts_str,
                        second_timestamp=None,
                        timestamp=ts_str,
                        source=f"Police FIR Registry ({case_code}) · Knowledge Graph",
                        evidence_channels=["POLICE_FIR_REGISTRY", "KNOWLEDGE_GRAPH"],
                        confidence_score=0.95,
                        confidence=0.95,
                        evidence_snippet=snippet,
                        graph_relationship="INVOLVED_IN",
                        timeline_link={
                            "entity_id": linked_persons[0].id,
                            "event_type": "CASE",
                            "case_id": case_code,
                        },
                        source_evidence_link={
                            "entity_id": c_node.id,
                            "channel": "POLICE_FIR_REGISTRY",
                            "case_id": case_code,
                        },
                        details={
                            "case_code": case_code,
                            "suspects": p_names,
                            "vehicles": [v.name for v in linked_vehicles],
                            "accounts": [a.name for a in linked_accounts],
                        },
                        recommended_action="Review primary FIR narrative snippet and verify witness/suspect depositions.",
                    )
                )
                c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 4. Biometric Face Match & Surveillance Detector
    # -----------------------------------------------------------------------

    def detect_biometric_face_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify biometric facial enrollments and surveillance identity verifications linked to case subjects.
        """
        identities = self.face_storage.list_identities()
        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1

        for face_rec in identities:
            person_id = face_rec.person_id or face_rec.identity_id
            if not person_id:
                continue
            node = self.graph.get_node(person_id)
            p_name = node.name if node else face_rec.name

            case_str = face_rec.case_id or "General Surveillance Registry"
            enrolled_ts = getattr(face_rec, "registration_timestamp", None) or getattr(face_rec, "enrolled_at", None) or datetime.now().isoformat()
            enrolled_dt = _parse_ts(enrolled_ts)
            ts_str = enrolled_dt.isoformat() if enrolled_dt else datetime.now().isoformat()

            snippet = (
                f"Biometric Facial Registry entry verified for subject '{p_name}' ({person_id}) "
                f"enrolled under reference {case_str} [Vector Embedding: 512D ResNet-50]."
            )

            correlations.append(
                CrossDomainCorrelationItem(
                    correlation_id=f"CORR_FACE_{c_idx}",
                    correlation_type="BIOMETRIC_ALIGNMENT",
                    signal_type="Biometric Identity Alignment",
                    severity="ELEVATED",
                    title=f"Biometric Facial Profile Alignment ({p_name})",
                    description=(
                        f"Observed relationship: Facial recognition registry record aligns with subject profile "
                        f"'{p_name}' linked to {case_str} (investigator verification required)."
                    ),
                    entities_involved=[person_id] + ([face_rec.case_id] if face_rec.case_id else []),
                    time_window_minutes=None,
                    first_timestamp=ts_str,
                    second_timestamp=None,
                    timestamp=ts_str,
                    source="Biometric Face Registry · Surveillance Database",
                    evidence_channels=["BIOMETRIC_FACE_DATABASE", "KNOWLEDGE_GRAPH"],
                    confidence_score=0.90,
                    confidence=0.90,
                    evidence_snippet=snippet,
                    graph_relationship="BIOMETRIC_PROFILE",
                    timeline_link={
                        "entity_id": person_id,
                        "event_type": "FACE",
                    },
                    source_evidence_link={
                        "entity_id": person_id,
                        "channel": "BIOMETRIC_FACE_DATABASE",
                        "case_id": face_rec.case_id,
                    },
                    details={
                        "person_id": person_id,
                        "name": p_name,
                        "case_id": face_rec.case_id,
                        "embedding_dim": len(face_rec.embedding) if face_rec.embedding else 0,
                    },
                    recommended_action="Validate biometric similarity score against secondary photo gallery capture.",
                )
            )
            c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 5. Knowledge Graph Topology & Operative Clusters Detector
    # -----------------------------------------------------------------------

    def detect_graph_topology_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify structural organizational clusters where multiple subjects operate or share key nodes.
        """
        all_nodes = self.graph.get_all_nodes()
        org_nodes = [
            n for n in all_nodes
            if (n.label.value if hasattr(n.label, "value") else str(n.label)) in ("ORGANIZATION", NodeType.ORGANIZATION.value)
            or n.id.startswith("ORG_")
        ]

        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1

        for o_node in org_nodes:
            nbrs = self.graph.get_neighbors(o_node.id, depth=1)
            operatives = [
                n for n in nbrs.get("nodes", [])
                if n.id != o_node.id
                and ((n.label.value if hasattr(n.label, "value") else str(n.label)) in ("PERSON", NodeType.PERSON.value) or n.id.startswith("PERSON_"))
            ]

            if len(operatives) >= 2:
                org_name = o_node.name or o_node.id
                op_names = [op.name for op in operatives]
                now_iso = datetime.now().isoformat()

                snippet = (
                    f"Organizational entity '{org_name}' ({o_node.id}) is shared across "
                    f"{len(operatives)} documented operative(s): {', '.join(op_names)}."
                )

                correlations.append(
                    CrossDomainCorrelationItem(
                        correlation_id=f"CORR_ORG_{c_idx}",
                        correlation_type="GRAPH_TOPOLOGY",
                        signal_type="Knowledge Graph Topology",
                        severity="HIGH" if len(operatives) >= 3 else "ELEVATED",
                        title=f"Shared Organization Operative Cluster ({org_name})",
                        description=(
                            f"Correlated activity: Relational graph clustering connects {len(operatives)} subject(s) "
                            f"({', '.join(op_names[:3])}) to organizational entity '{org_name}'."
                        ),
                        entities_involved=[o_node.id] + [op.id for op in operatives],
                        time_window_minutes=None,
                        first_timestamp=now_iso,
                        second_timestamp=None,
                        timestamp=now_iso,
                        source="Knowledge Graph Engine · Entity Network",
                        evidence_channels=["KNOWLEDGE_GRAPH"],
                        confidence_score=0.91,
                        confidence=0.91,
                        evidence_snippet=snippet,
                        graph_relationship="OPERATES",
                        timeline_link={
                            "entity_id": operatives[0].id,
                            "event_type": "GRAPH",
                        },
                        source_evidence_link={
                            "entity_id": o_node.id,
                            "channel": "KNOWLEDGE_GRAPH",
                        },
                        details={"organization": org_name, "operatives": op_names},
                        recommended_action="Investigate corporate registrar records and corporate bank accounts.",
                    )
                )
                c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 6. Hidden Link Discovery: Shared Phone Numbers
    # -----------------------------------------------------------------------
    def detect_shared_phone_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify distinct subjects who share or operate the same telephone identifier.
        """
        all_nodes = self.graph.get_all_nodes()
        phone_nodes = [
            n for n in all_nodes
            if (n.label.value if hasattr(n.label, "value") else str(n.label)) in ("PHONE", NodeType.PHONE.value)
            or n.id.startswith("PHONE_")
        ]

        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1
        now_iso = datetime.now().isoformat()

        for p_node in phone_nodes:
            nbrs = self.graph.get_neighbors(p_node.id, depth=1)
            linked_persons = [
                n for n in nbrs.get("nodes", [])
                if n.id != p_node.id
                and ((n.label.value if hasattr(n.label, "value") else str(n.label)) in ("PERSON", NodeType.PERSON.value) or n.id.startswith("PERSON_"))
            ]

            if len(linked_persons) >= 2:
                phone_num = p_node.properties.get("phone_number") or p_node.name or p_node.id
                p_names = [p.name for p in linked_persons]
                p_ids = [p.id for p in linked_persons]

                snippet = (
                    f"Telephony record documents shared phone {phone_num} ({p_node.id}) "
                    f"associated with {len(linked_persons)} distinct subjects: {', '.join(p_names)}."
                )

                correlations.append(
                    CrossDomainCorrelationItem(
                        correlation_id=f"CORR_SHARED_PHONE_{c_idx}",
                        correlation_type="SHARED_PHONE",
                        signal_type="Hidden Link: Shared Phone Lead",
                        severity="HIGH",
                        title=f"Potential Connection: Shared Telephony Identifier ({phone_num})",
                        description=(
                            f"Investigative lead: Shared telephony identifier {phone_num} links "
                            f"{len(linked_persons)} subjects ({', '.join(p_names[:3])})."
                        ),
                        entities_involved=[p_node.id] + p_ids,
                        time_window_minutes=None,
                        first_timestamp=now_iso,
                        second_timestamp=None,
                        timestamp=now_iso,
                        source="CDR Telephony Logs · Knowledge Graph",
                        evidence_channels=["CDR_TELEPHONY", "KNOWLEDGE_GRAPH"],
                        confidence_score=0.94,
                        confidence=0.94,
                        evidence_snippet=snippet,
                        graph_relationship="SHARED_PHONE",
                        timeline_link={
                            "entity_id": p_ids[0],
                            "event_type": "HIDDEN_LINK",
                            "phone_number": phone_num,
                        },
                        source_evidence_link={
                            "entity_id": p_node.id,
                            "channel": "CDR_TELEPHONY",
                        },
                        details={
                            "provenance_nature": "INFERRED_CONNECTION",
                            "derivation_basis": f"Shared phone identifier {phone_num} observed across {len(linked_persons)} linked subjects",
                            "intermediary_nodes": [p_node.id],
                            "source_domain": "CDR_TELEPHONY",
                            "phone_number": phone_num,
                            "subjects": p_names,
                        },
                        recommended_action="Cross-reference subscriber CDR call logs and SIM registration records.",
                    )
                )
                c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 7. Hidden Link Discovery: Shared Bank Accounts
    # -----------------------------------------------------------------------
    def detect_shared_account_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify distinct subjects connected to or transacting through a shared bank account.
        """
        all_nodes = self.graph.get_all_nodes()
        acc_nodes = [
            n for n in all_nodes
            if (n.label.value if hasattr(n.label, "value") else str(n.label)) in ("BANK_ACCOUNT", "BANKACCOUNT", NodeType.BANK_ACCOUNT.value)
            or n.id.startswith("ACC_")
        ]

        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1
        now_iso = datetime.now().isoformat()

        for a_node in acc_nodes:
            nbrs = self.graph.get_neighbors(a_node.id, depth=1)
            linked_persons = [
                n for n in nbrs.get("nodes", [])
                if n.id != a_node.id
                and ((n.label.value if hasattr(n.label, "value") else str(n.label)) in ("PERSON", NodeType.PERSON.value) or n.id.startswith("PERSON_"))
            ]

            if len(linked_persons) >= 2:
                acc_num = a_node.properties.get("account_number") or a_node.name or a_node.id
                p_names = [p.name for p in linked_persons]
                p_ids = [p.id for p in linked_persons]

                snippet = (
                    f"Financial ledger documents shared banking account {acc_num} ({a_node.id}) "
                    f"utilized across {len(linked_persons)} subjects: {', '.join(p_names)}."
                )

                correlations.append(
                    CrossDomainCorrelationItem(
                        correlation_id=f"CORR_SHARED_ACC_{c_idx}",
                        correlation_type="SHARED_ACCOUNT",
                        signal_type="Hidden Link: Shared Bank Account Lead",
                        severity="CRITICAL",
                        title=f"Potential Connection: Common Banking Identifier ({acc_num})",
                        description=(
                            f"Investigative lead: Shared financial account {acc_num} links "
                            f"{len(linked_persons)} subjects ({', '.join(p_names[:3])})."
                        ),
                        entities_involved=[a_node.id] + p_ids,
                        time_window_minutes=None,
                        first_timestamp=now_iso,
                        second_timestamp=None,
                        timestamp=now_iso,
                        source="Bank Ledger · Financial Analytics",
                        evidence_channels=["BANK_LEDGER", "KNOWLEDGE_GRAPH"],
                        confidence_score=0.96,
                        confidence=0.96,
                        evidence_snippet=snippet,
                        graph_relationship="SHARED_ACCOUNT",
                        timeline_link={
                            "entity_id": p_ids[0],
                            "event_type": "HIDDEN_LINK",
                            "account_number": acc_num,
                        },
                        source_evidence_link={
                            "entity_id": a_node.id,
                            "channel": "BANK_LEDGER",
                        },
                        details={
                            "provenance_nature": "INFERRED_CONNECTION",
                            "derivation_basis": f"Common bank account {acc_num} observed across {len(linked_persons)} subjects",
                            "intermediary_nodes": [a_node.id],
                            "source_domain": "FINANCIAL",
                            "account_number": acc_num,
                            "subjects": p_names,
                        },
                        recommended_action="Audit KYC account authorization records and transaction counterparties.",
                    )
                )
                c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 8. Hidden Link Discovery: Shared Vehicles / License Plates
    # -----------------------------------------------------------------------
    def detect_shared_vehicle_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify distinct subjects associated with the same vehicle registration plate.
        """
        all_nodes = self.graph.get_all_nodes()
        veh_nodes = [
            n for n in all_nodes
            if (n.label.value if hasattr(n.label, "value") else str(n.label)) in ("VEHICLE", NodeType.VEHICLE.value)
            or n.id.startswith("VEH_")
        ]

        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1
        now_iso = datetime.now().isoformat()

        for v_node in veh_nodes:
            nbrs = self.graph.get_neighbors(v_node.id, depth=1)
            linked_persons = [
                n for n in nbrs.get("nodes", [])
                if n.id != v_node.id
                and ((n.label.value if hasattr(n.label, "value") else str(n.label)) in ("PERSON", NodeType.PERSON.value) or n.id.startswith("PERSON_"))
            ]

            if len(linked_persons) >= 2:
                plate = v_node.properties.get("registration") or v_node.name or v_node.id
                p_names = [p.name for p in linked_persons]
                p_ids = [p.id for p in linked_persons]

                snippet = (
                    f"Police case records document shared vehicle {plate} ({v_node.id}) "
                    f"associated with {len(linked_persons)} distinct subjects: {', '.join(p_names)}."
                )

                correlations.append(
                    CrossDomainCorrelationItem(
                        correlation_id=f"CORR_SHARED_VEH_{c_idx}",
                        correlation_type="SHARED_VEHICLE",
                        signal_type="Hidden Link: Shared Vehicle Lead",
                        severity="HIGH",
                        title=f"Potential Connection: Shared Vehicle Registration ({plate})",
                        description=(
                            f"Investigative lead: Vehicle registration {plate} associated with "
                            f"{len(linked_persons)} subjects ({', '.join(p_names[:3])})."
                        ),
                        entities_involved=[v_node.id] + p_ids,
                        time_window_minutes=None,
                        first_timestamp=now_iso,
                        second_timestamp=None,
                        timestamp=now_iso,
                        source="Police FIR Records · Vehicle Registry",
                        evidence_channels=["POLICE_FIR_REGISTRY", "KNOWLEDGE_GRAPH"],
                        confidence_score=0.93,
                        confidence=0.93,
                        evidence_snippet=snippet,
                        graph_relationship="SHARED_VEHICLE",
                        timeline_link={
                            "entity_id": p_ids[0],
                            "event_type": "HIDDEN_LINK",
                            "vehicle": plate,
                        },
                        source_evidence_link={
                            "entity_id": v_node.id,
                            "channel": "POLICE_FIR_REGISTRY",
                        },
                        details={
                            "provenance_nature": "INFERRED_CONNECTION",
                            "derivation_basis": f"Vehicle registration {plate} linked to {len(linked_persons)} subjects",
                            "intermediary_nodes": [v_node.id],
                            "source_domain": "FIR",
                            "registration": plate,
                            "subjects": p_names,
                        },
                        recommended_action="Verify RTO vehicle ownership transfer and toll-plaza surveillance logs.",
                    )
                )
                c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 9. Hidden Link Discovery: Cross-Case Associations
    # -----------------------------------------------------------------------
    def detect_cross_case_link_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify entities (persons, vehicles, phones, accounts) that appear across 2 or more distinct Case / FIR nodes.
        """
        all_nodes = self.graph.get_all_nodes()
        all_edges = self.graph.get_all_edges()

        # Map entity -> set of connected case nodes
        entity_cases: Dict[str, Set[str]] = {}
        for e in all_edges:
            src_node = self.graph.get_node(e.source)
            tgt_node = self.graph.get_node(e.target)
            if not src_node or not tgt_node:
                continue

            t_lbl = tgt_node.label.value if hasattr(tgt_node.label, "value") else str(tgt_node.label)
            s_lbl = src_node.label.value if hasattr(src_node.label, "value") else str(src_node.label)

            if t_lbl in ("CASE", NodeType.CASE.value) or tgt_node.id.startswith("CASE_"):
                entity_cases.setdefault(e.source, set()).add(tgt_node.id)
            if s_lbl in ("CASE", NodeType.CASE.value) or src_node.id.startswith("CASE_"):
                entity_cases.setdefault(e.target, set()).add(src_node.id)

        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1
        now_iso = datetime.now().isoformat()

        for ent_id, cases in entity_cases.items():
            if len(cases) >= 2:
                node = self.graph.get_node(ent_id)
                ent_name = node.name if node else ent_id
                case_names = []
                for cid in sorted(list(cases)):
                    cn = self.graph.get_node(cid)
                    case_names.append(cn.name if cn else cid)

                snippet = (
                    f"Police FIR registry documents multi-case co-involvement: Entity '{ent_name}' ({ent_id}) is linked "
                    f"across {len(cases)} distinct FIR/Case filings: {', '.join(case_names)}."
                )

                correlations.append(
                    CrossDomainCorrelationItem(
                        correlation_id=f"CORR_CROSS_CASE_{c_idx}",
                        correlation_type="CROSS_CASE_LINK",
                        signal_type="Hidden Link: Cross-Case Association",
                        severity="CRITICAL",
                        title=f"Cross-Case Link: Multi-Case Co-Occurrence ({ent_name})",
                        description=(
                            f"Observed relationship: Entity '{ent_name}' is linked across multiple "
                            f"independent case records: {', '.join(case_names[:3])}."
                        ),
                        entities_involved=[ent_id] + list(cases),
                        time_window_minutes=None,
                        first_timestamp=now_iso,
                        second_timestamp=None,
                        timestamp=now_iso,
                        source="Police FIR Registry · Knowledge Graph",
                        evidence_channels=["POLICE_FIR_REGISTRY", "KNOWLEDGE_GRAPH"],
                        confidence_score=0.98,
                        confidence=0.98,
                        evidence_snippet=snippet,
                        graph_relationship="CROSS_CASE_LINK",
                        timeline_link={
                            "entity_id": ent_id,
                            "event_type": "CASE",
                        },
                        source_evidence_link={
                            "entity_id": ent_id,
                            "channel": "POLICE_FIR_REGISTRY",
                        },
                        details={
                            "provenance_nature": "DIRECT_EVIDENCE",
                            "derivation_basis": f"Direct co-occurrence of {ent_name} across cases {', '.join(case_names)}",
                            "source_domain": "FIR",
                            "case_ids": list(cases),
                            "case_names": case_names,
                        },
                        recommended_action="Consolidate cross-case investigation briefs and coordinate with respective IOs.",
                    )
                )
                c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # 10. Hidden Link Discovery: Intermediary Bridge Nodes
    # -----------------------------------------------------------------------
    def detect_intermediary_bridge_correlations(self) -> List[CrossDomainCorrelationItem]:
        """
        Identify bridge entities connecting two or more subjects who have no direct edge between them.
        """
        all_nodes = self.graph.get_all_nodes()
        correlations: List[CrossDomainCorrelationItem] = []
        c_idx = 1
        now_iso = datetime.now().isoformat()

        for node in all_nodes:
            lbl = node.label.value if hasattr(node.label, "value") else str(node.label)
            if lbl in ("CASE", NodeType.CASE.value) or node.id.startswith("CASE_"):
                continue

            nbrs = self.graph.get_neighbors(node.id, depth=1)
            connected_persons = [
                n for n in nbrs.get("nodes", [])
                if n.id != node.id
                and ((n.label.value if hasattr(n.label, "value") else str(n.label)) in ("PERSON", NodeType.PERSON.value) or n.id.startswith("PERSON_"))
            ]

            if len(connected_persons) >= 2:
                # Check if there is an unlinked pair
                p1, p2 = connected_persons[0], connected_persons[1]
                edge_direct = self.graph.get_edge_between(p1.id, p2.id) or self.graph.get_edge_between(p2.id, p1.id)

                if not edge_direct:
                    bridge_name = node.name or node.id
                    snippet = (
                        f"Graph topology documents observed relationship: Intermediary entity '{bridge_name}' ({node.id}) "
                        f"is linked between otherwise unconnected subjects '{p1.name}' and '{p2.name}'."
                    )

                    correlations.append(
                        CrossDomainCorrelationItem(
                            correlation_id=f"CORR_BRIDGE_{c_idx}",
                            correlation_type="INTERMEDIARY_BRIDGE",
                            signal_type="Hidden Link: Intermediary Bridge Node",
                            severity="HIGH",
                            title=f"Potential Connection: Intermediary Bridge Node ({bridge_name})",
                            description=(
                                f"Observed relationship: Intermediary entity '{bridge_name}' is linked as a bridge "
                                f"connecting '{p1.name}' and '{p2.name}' without a direct connection."
                            ),
                            entities_involved=[node.id, p1.id, p2.id],
                            time_window_minutes=None,
                            first_timestamp=now_iso,
                            second_timestamp=None,
                            timestamp=now_iso,
                            source="Knowledge Graph Engine · Topology",
                            evidence_channels=["KNOWLEDGE_GRAPH"],
                            confidence_score=0.91,
                            confidence=0.91,
                            evidence_snippet=snippet,
                            graph_relationship="INTERMEDIARY_BRIDGE",
                            timeline_link={
                                "entity_id": node.id,
                                "event_type": "GRAPH",
                            },
                            source_evidence_link={
                                "entity_id": node.id,
                                "channel": "KNOWLEDGE_GRAPH",
                            },
                            details={
                                "provenance_nature": "INFERRED_CONNECTION",
                                "derivation_basis": f"2-hop topological bridge through {bridge_name} between {p1.name} and {p2.name}",
                                "intermediary_nodes": [node.id],
                                "source_domain": "CROSS_DOMAIN",
                                "bridge_node": node.id,
                                "bridge_name": bridge_name,
                                "subject_1": p1.name,
                                "subject_2": p2.name,
                            },
                            recommended_action="Examine financial flows and communication logs passing through this intermediary.",
                        )
                    )
                    c_idx += 1

        return correlations

    # -----------------------------------------------------------------------
    # System-Wide & Target-Specific Retrieval
    # -----------------------------------------------------------------------

    def get_all_cross_domain_correlations(self) -> CrossDomainSummaryResponse:
        """Aggregate all fused cross-domain correlations across all monitored channels."""
        c2t = self.detect_call_to_transfer_correlations()
        coloc = self.detect_cotemporal_colocation_correlations()
        firs = self.detect_fir_case_correlations()
        faces = self.detect_biometric_face_correlations()
        topos = self.detect_graph_topology_correlations()
        shared_phones = self.detect_shared_phone_correlations()
        shared_accs = self.detect_shared_account_correlations()
        shared_vehs = self.detect_shared_vehicle_correlations()
        cross_cases = self.detect_cross_case_link_correlations()
        bridges = self.detect_intermediary_bridge_correlations()

        all_corrs = c2t + coloc + firs + faces + topos + shared_phones + shared_accs + shared_vehs + cross_cases + bridges

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
                "KNOWLEDGE_GRAPH",
            ],
        )

    def get_entity_cross_domain_profile(self, entity_id: str) -> EntityCrossDomainProfile:
        """Get cross-domain correlation profile for a specific person or identifier."""
        clean_id = entity_id.strip()
        all_corrs = self.get_all_cross_domain_correlations().correlations
        node = self.graph.get_node(clean_id)

        # Collect entity ID variants (clean_id, phones, accounts, cases)
        entity_map = self._resolve_entity_mappings()
        associated_ids = {clean_id, clean_id.lower(), clean_id.upper()}
        if clean_id in entity_map:
            associated_ids.update(entity_map[clean_id]["phones"])
            associated_ids.update(entity_map[clean_id]["accounts"])
            associated_ids.update(entity_map[clean_id]["cases"])
            associated_ids.update(entity_map[clean_id]["vehicles"])

        matching = [
            c
            for c in all_corrs
            if any(
                str(ent_inv).lower() in [a.lower() for a in associated_ids]
                or any(a.lower() in str(ent_inv).lower() for a in associated_ids)
                for ent_inv in c.entities_involved
            )
        ]

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

        risk_index = min(100.0, float(len(matching) * 20.0 + (30.0 if highest_sev == "CRITICAL" else 15.0 if highest_sev == "HIGH" else 0.0)))

        return EntityCrossDomainProfile(
            entity_id=clean_id,
            entity_name=node.name if node else clean_id,
            entity_type=node.label.value if (node and hasattr(node.label, "value")) else "Unknown",
            total_fused_events=len(matching),
            highest_severity=highest_sev,
            correlations=matching,
            associated_channels=sorted(list(channels)),
            multi_channel_risk_index=round(risk_index, 1),
        )

    def get_case_cross_domain_profile(self, case_id: str) -> CaseCrossDomainProfile:
        """Get cross-domain correlation profile for an entire case / FIR."""
        clean_case = case_id.strip()
        all_corrs = self.get_all_cross_domain_correlations().correlations
        case_node = self.graph.get_node(clean_case)

        # Collect entities in case neighborhood
        nbrs = self.graph.get_neighbors(clean_case, depth=1) if case_node else {"nodes": []}
        case_entity_ids = {clean_case, clean_case.lower(), clean_case.upper()}
        for n in nbrs.get("nodes", []):
            case_entity_ids.add(n.id)
            case_entity_ids.add(n.name)

        matching = [
            c
            for c in all_corrs
            if any(
                str(ent_inv).lower() in [ce.lower() for ce in case_entity_ids]
                or any(ce.lower() in str(ent_inv).lower() for ce in case_entity_ids)
                for ent_inv in c.entities_involved
            )
        ]

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

        return CaseCrossDomainProfile(
            case_id=clean_case,
            case_title=case_node.name if case_node else clean_case,
            total_fused_events=len(matching),
            highest_severity=highest_sev,
            involved_entities=sorted(list(case_entity_ids)),
            correlations=matching,
            associated_channels=sorted(list(channels)),
        )


# Singleton
_fusion_instance: Optional[CrossDomainFusionEngine] = None


def get_cross_domain_fusion_engine() -> CrossDomainFusionEngine:
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = CrossDomainFusionEngine()
    return _fusion_instance
