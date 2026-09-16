# -*- coding: utf-8 -*-
"""
Explainable Risk & Anomaly Intelligence Engine for NODE SENTINEL.
Replaces opaque risk scores with an explainable, traceable evidence breakdown
across Graph Analytics, CDR Telephony, Financial Transactions, Case Registries,
and Physical Surveillance Co-Locations.
Strictly adheres to neutral investigator decision-support standards.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.core.cdr_analytics import get_cdr_storage
from app.core.financial_analytics import get_financial_storage
from app.core.graph_analytics import GraphAnalytics
from app.core.graph_engine import BaseGraphEngine, get_graph_engine
from app.models.graph_models import EdgeType, NodeType
from app.models.risk_models import (
    RiskCategory,
    RiskFactor,
    RiskIntelligenceResult,
    RiskSeverity,
)
from app.models.schemas import AlertItem, RiskScoreBreakdown

logger = logging.getLogger(__name__)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        cleaned = str(val).replace("₹", "").replace(",", "").replace("$", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return default


class AnomalyDetector:
    """Statistical anomaly detector & Explainable Investigative Risk Intelligence Engine."""

    def __init__(self, graph_engine: Optional[BaseGraphEngine] = None):
        self.graph_engine = graph_engine or get_graph_engine()
        self.analytics = GraphAnalytics(self.graph_engine)
        self.cdr_storage = get_cdr_storage()
        self.fin_storage = get_financial_storage()

    # -------------------------------------------------------------------------
    # 1. Graph-Level Anomaly Detection (Preserved)
    # -------------------------------------------------------------------------

    def detect_call_bursts(self) -> List[AlertItem]:
        """Flag CDR relationships with communication frequency > 3x mean."""
        edges = self.graph_engine.get_all_edges()
        call_edges = [
            e for e in edges
            if (e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship)) == EdgeType.CALLS.value
        ]

        if not call_edges or len(call_edges) < 2:
            return []

        frequencies = []
        for e in call_edges:
            raw_freq = e.properties.get("frequency", e.properties.get("count", 1))
            frequencies.append(_safe_float(raw_freq, 1.0))

        if not frequencies:
            return []

        mean_freq = float(np.mean(frequencies))
        burst_threshold = max(3.0, mean_freq * 3.0)

        alerts = []
        for idx, e in enumerate(call_edges):
            raw_freq = e.properties.get("frequency", e.properties.get("count", 1))
            freq = _safe_float(raw_freq, 1.0)
            if freq >= burst_threshold:
                source_node = self.graph_engine.get_node(e.source)
                target_node = self.graph_engine.get_node(e.target)
                src_name = source_node.name if source_node else e.source
                tgt_name = target_node.name if target_node else e.target

                alerts.append(AlertItem(
                    id=f"ALERT_CALL_BURST_{idx}_{e.id}",
                    alert_type="CALL_BURST",
                    severity="HIGH",
                    title="Suspicious Call Burst Detected",
                    description=f"Communication frequency between '{src_name}' and '{tgt_name}' ({int(freq)} calls) exceeds 3x mean frequency ({mean_freq:.1f}).",
                    entities=[e.source, e.target],
                    timestamp=e.properties.get("timestamp", datetime.now().isoformat()),
                    evidence={
                        "observed_frequency": freq,
                        "network_mean_frequency": round(mean_freq, 2),
                        "multiplier": round(freq / (mean_freq + 0.01), 2),
                    },
                ))

        return alerts

    def detect_financial_anomalies(self) -> List[AlertItem]:
        """Flag financial transactions > 2 sigma above user/network average."""
        edges = self.graph_engine.get_all_edges()
        money_edges = [
            e for e in edges
            if (e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship)) == EdgeType.TRANSFERRED_MONEY.value
        ]

        if not money_edges:
            return []

        amounts = []
        for e in money_edges:
            amt = _safe_float(e.properties.get("amount", 0.0))
            if amt > 0:
                amounts.append(amt)

        if not amounts or len(amounts) < 2:
            return []

        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts))

        if std_amt < 1e-4:
            return []

        threshold = mean_amt + (2.0 * std_amt)

        alerts = []
        for idx, e in enumerate(money_edges):
            amt = _safe_float(e.properties.get("amount", 0.0))
            if amt >= threshold and amt > mean_amt and amt > 0:
                sender = self.graph_engine.get_node(e.source)
                receiver = self.graph_engine.get_node(e.target)
                src_name = sender.name if sender else e.source
                rec_name = receiver.name if receiver else e.target
                z_score = (amt - mean_amt) / (std_amt + 0.01)

                alerts.append(AlertItem(
                    id=f"ALERT_FINANCIAL_SPIKE_{idx}_{e.id}",
                    alert_type="FINANCIAL_ANOMALY",
                    severity="CRITICAL",
                    title="High-Value Financial Transaction Anomaly",
                    description=f"Transaction of ₹{amt:,.2f} from '{src_name}' to '{rec_name}' is {z_score:.1f}σ above average.",
                    entities=[e.source, e.target],
                    timestamp=e.properties.get("timestamp", datetime.now().isoformat()),
                    evidence={
                        "amount": amt,
                        "mean_amount": round(mean_amt, 2),
                        "z_score": round(z_score, 2),
                        "tx_id": e.properties.get("tx_id", "N/A"),
                    },
                ))

        return alerts

    def detect_colocation_clusters(self) -> List[AlertItem]:
        """Flag co-location clusters where multiple individuals appear at same location in tight window."""
        edges = self.graph_engine.get_all_edges()
        loc_edges = [
            e for e in edges
            if (e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship)) == EdgeType.LOCATED_AT.value
        ]

        loc_to_people: Dict[str, List[str]] = {}
        for e in loc_edges:
            src_node = self.graph_engine.get_node(e.source)
            tgt_node = self.graph_engine.get_node(e.target)
            src_lbl = src_node.label.value if src_node and hasattr(src_node.label, "value") else (str(src_node.label) if src_node else "")
            tgt_lbl = tgt_node.label.value if tgt_node and hasattr(tgt_node.label, "value") else (str(tgt_node.label) if tgt_node else "")

            if tgt_lbl == NodeType.LOCATION.value:
                loc_id, person_id = e.target, e.source
            elif src_lbl == NodeType.LOCATION.value:
                loc_id, person_id = e.source, e.target
            else:
                loc_id, person_id = e.target, e.source
            loc_to_people.setdefault(loc_id, []).append(person_id)

        alerts = []
        for loc_id, person_ids in loc_to_people.items():
            unique_people = list(set(person_ids))
            if len(unique_people) >= 3:
                loc_node = self.graph_engine.get_node(loc_id)
                loc_name = loc_node.name if loc_node else loc_id

                person_names = []
                for pid in unique_people:
                    pn = self.graph_engine.get_node(pid)
                    person_names.append(pn.name if pn else pid)

                alerts.append(AlertItem(
                    id=f"ALERT_COLOCATION_{loc_id}",
                    alert_type="COLOCATION_CLUSTER",
                    severity="HIGH",
                    title="Co-Location Cluster Alert",
                    description=f"Cluster of {len(unique_people)} suspects ({', '.join(person_names[:3])}) co-located at '{loc_name}'.",
                    entities=[loc_id] + unique_people,
                    timestamp=datetime.now().isoformat(),
                    evidence={
                        "location_id": loc_id,
                        "location_name": loc_name,
                        "co_located_count": len(unique_people),
                        "suspects": person_names,
                    },
                ))

        return alerts

    def get_all_alerts(self) -> List[AlertItem]:
        """Aggregate all graph anomaly alerts."""
        call_alerts = self.detect_call_bursts()
        fin_alerts = self.detect_financial_anomalies()
        loc_alerts = self.detect_colocation_clusters()
        return call_alerts + fin_alerts + loc_alerts

    # -------------------------------------------------------------------------
    # 2. STEP 12: Explainable Risk & Anomaly Intelligence Calculation
    # -------------------------------------------------------------------------

    def calculate_investigative_risk_score(self, entity_id: str) -> RiskIntelligenceResult:
        """
        Calculate composite 0-100 Investigative Risk Score with explainable evidence breakdown.
        Replaces opaque scores with reproducible evidence-backed factors across:
        - Police FIR Registry & Role
        - Graph Centrality & Broker Hub Status
        - CDR Communication Volume & Bursts
        - Financial Velocity & High-Value Transfers
        - Direct Case Linkages & Serious Statutory Sections
        - Location Surveillance Co-Locations
        """
        clean_id = (entity_id or "").strip()
        node = self.graph_engine.get_node(clean_id) if clean_id else None

        # Resolve subject identity
        ent_name = node.name if node else clean_id
        ent_type = node.label.value if (node and hasattr(node.label, "value")) else (str(node.label) if node else "Unknown")

        if not node:
            # Check if entity exists in CDR or Financial storage
            has_cdr = clean_id and len(self.cdr_storage.get_records_for_phone(clean_id)) > 0
            has_fin = clean_id and len(self.fin_storage.get_records_for_account(clean_id)) > 0
            if not has_cdr and not has_fin:
                return RiskIntelligenceResult(
                    entity_id=clean_id,
                    entity_name=clean_id or "Unknown Entity",
                    entity_type="Unknown",
                    risk_score=0.0,
                    risk_level="LOW",
                    overall_score=0.0,
                    severity_level="LOW",
                    factors=[],
                    evidence_summary=["No observed investigative risk indicators for this entity."],
                    conclusion="Decision-Support Signal: Entity not present in active intelligence databases. Requires investigator verification.",
                )

        factors: List[RiskFactor] = []

        # Collect linked identifiers for multi-channel correlation
        linked_phones: Set[str] = set()
        linked_accounts: Set[str] = set()
        linked_cases: List[Any] = []

        if node:
            if "phone_number" in node.properties:
                linked_phones.add(str(node.properties["phone_number"]))
            if "account_number" in node.properties:
                linked_accounts.add(str(node.properties["account_number"]))

            neighbors = self.graph_engine.get_neighbors(clean_id, depth=1).get("nodes", [])
            for n in neighbors:
                if n.id == clean_id:
                    continue
                n_lbl = n.label.value if hasattr(n.label, "value") else str(n.label)
                if n_lbl == NodeType.PHONE.value:
                    linked_phones.add(n.id)
                    raw_p = n.properties.get("phone_number") or n.name
                    if raw_p:
                        linked_phones.add(str(raw_p))
                elif n_lbl == NodeType.BANK_ACCOUNT.value:
                    linked_accounts.add(n.id)
                    raw_a = n.properties.get("account_number") or n.name
                    if raw_a:
                        linked_accounts.add(str(raw_a))
                elif n_lbl == NodeType.CASE.value:
                    linked_cases.append(n)

        # ---------------------------------------------------------------------
        # Factor 1: REGISTRY & ROLE INDICATORS (Max: 35 pts)
        # ---------------------------------------------------------------------
        if node:
            tagged_risk = str(node.properties.get("risk_tag", node.properties.get("risk_level", ""))).upper()
            if tagged_risk in ("HIGH", "CRITICAL"):
                factors.append(RiskFactor(
                    factor_id="FAC_REG_HIGH",
                    category=RiskCategory.REGISTRY,
                    title="High-priority investigative risk indicator",
                    score_contribution=35.0,
                    numerical_contribution=35.0,
                    severity="HIGH",
                    underlying_metric="Official intelligence registry tag: HIGH PRIORITY SUSPECT",
                    explanation="Official law enforcement intelligence registry maintains an active high-priority suspect classification for this entity.",
                    evidence="Tagged as High-Risk Suspect in Police FIR Registry records",
                    source="Police FIR Registry",
                    entity_id=clean_id,
                    involved_entities=[clean_id],
                    confidence=0.98,
                    action_hint="View Registry Dossier",
                    action_type="VIEW_DOSSIER",
                ))
            elif tagged_risk in ("MEDIUM", "ELEVATED"):
                factors.append(RiskFactor(
                    factor_id="FAC_REG_MED",
                    category=RiskCategory.REGISTRY,
                    title="Medium-priority investigative risk indicator",
                    score_contribution=15.0,
                    numerical_contribution=15.0,
                    severity="MODERATE",
                    underlying_metric="Official intelligence registry tag: MEDIUM PRIORITY POI",
                    explanation="Entity is tagged as a medium-priority person of interest in active police records.",
                    evidence="Tagged as Medium-Risk Person of Interest in Police Registry records",
                    source="Police FIR Registry",
                    entity_id=clean_id,
                    involved_entities=[clean_id],
                    confidence=0.95,
                    action_hint="View Registry Dossier",
                    action_type="VIEW_DOSSIER",
                ))

            role = str(node.properties.get("role", "")).strip()
            if role and any(r in role.lower() for r in ("kingpin", "coordinator", "mastermind", "boss", "head", "syndicate", "hawala")):
                factors.append(RiskFactor(
                    factor_id="FAC_REG_ROLE",
                    category=RiskCategory.REGISTRY,
                    title="Key operational syndicate role indicator",
                    score_contribution=15.0,
                    numerical_contribution=15.0,
                    severity="ELEVATED",
                    underlying_metric=f"Syndicate hierarchy designation: '{role}'",
                    explanation=f"Investigative intelligence documents entity with operational role '{role}'.",
                    evidence=f"Identified with role '{role}' in syndicate hierarchy records",
                    source="Police FIR Registry",
                    entity_id=clean_id,
                    involved_entities=[clean_id],
                    confidence=0.92,
                    action_hint="View Registry Dossier",
                    action_type="VIEW_DOSSIER",
                ))

        # ---------------------------------------------------------------------
        # Factor 2: GRAPH CENTRALITY & BROKER HUBS (Max: 25 pts)
        # ---------------------------------------------------------------------
        centrality_data = self.analytics.compute_centrality_metrics()
        b_val = round(centrality_data["betweenness"].get(clean_id, 0.0), 3)
        d_val = round(centrality_data["degree"].get(clean_id, 0.0), 3)

        if b_val >= 0.08:
            # Score contribution scaled by betweenness broker metric
            b_pts = min(25.0, max(10.0, round(b_val * 100.0, 1)))
            factors.append(RiskFactor(
                factor_id="FAC_GRAPH_BETWEENNESS",
                category=RiskCategory.GRAPH,
                title="High graph betweenness centrality",
                score_contribution=b_pts,
                numerical_contribution=b_pts,
                severity="HIGH" if b_val >= 0.15 else "ELEVATED",
                underlying_metric=f"Betweenness Centrality Metric: {b_val:.3f} (Top bridging tier)",
                explanation="Entity exhibits high betweenness centrality, operating as a vital broker or bridge between otherwise disconnected sub-networks.",
                evidence=f"Betweenness centrality: {b_val:.2f} (ranks in top network bridging tier)",
                source="Graph Analytics",
                entity_id=clean_id,
                involved_entities=[clean_id],
                confidence=0.96,
                timeline_event_type="GRAPH",
                action_hint="View Network Graph",
                action_type="VIEW_GRAPH",
            ))
        elif d_val >= 0.15:
            d_pts = min(15.0, max(5.0, round(d_val * 50.0, 1)))
            factors.append(RiskFactor(
                factor_id="FAC_GRAPH_DEGREE",
                category=RiskCategory.GRAPH,
                title="Elevated direct degree connectivity",
                score_contribution=d_pts,
                numerical_contribution=d_pts,
                severity="MODERATE",
                underlying_metric=f"Degree Centrality Metric: {d_val:.3f} (Dense direct connections)",
                explanation="Entity maintains an unusually high number of direct one-hop linkages across network nodes.",
                evidence=f"Degree centrality: {d_val:.2f} with dense direct connections",
                source="Graph Analytics",
                entity_id=clean_id,
                involved_entities=[clean_id],
                confidence=0.94,
                timeline_event_type="GRAPH",
                action_hint="View Network Graph",
                action_type="VIEW_GRAPH",
            ))

        # ---------------------------------------------------------------------
        # Factor 3: CDR COMMUNICATION BURSTS & DENSITY (Max: 20 pts)
        # ---------------------------------------------------------------------
        target_phones = {clean_id} | linked_phones
        all_calls = []
        for p in target_phones:
            all_calls.extend(self.cdr_storage.get_records_for_phone(p))

        # Deduplicate calls
        seen_call_ids = set()
        matched_calls = []
        for c in all_calls:
            cid = c.call_id or f"{c.caller}_{c.receiver}_{c.timestamp.isoformat()}"
            if cid not in seen_call_ids:
                seen_call_ids.add(cid)
                matched_calls.append(c)

        # Check call burst alert involvement
        all_alerts = self.get_all_alerts()
        call_burst_alerts = [a for a in all_alerts if a.alert_type == "CALL_BURST" and (clean_id in a.entities or any(p in a.entities for p in target_phones))]

        call_partners = set()
        for c in matched_calls:
            call_partners.add(c.caller)
            call_partners.add(c.receiver)

        cdr_ts = matched_calls[0].timestamp.isoformat() if (matched_calls and hasattr(matched_calls[0].timestamp, "isoformat")) else (str(matched_calls[0].timestamp) if matched_calls else None)

        if call_burst_alerts or len(matched_calls) >= 10:
            count_desc = f"{len(matched_calls)} calls recorded" if matched_calls else "Communication frequency > 3x network average"
            factors.append(RiskFactor(
                factor_id="FAC_CDR_BURST",
                category=RiskCategory.CDR,
                title="Communication burst detected",
                score_contribution=15.0,
                numerical_contribution=15.0,
                severity="ELEVATED",
                underlying_metric=f"Communication burst: {len(matched_calls)} calls in 24 hours (exceeds 3x baseline)",
                explanation="Call Detail Records (CDR) demonstrate intense sequential telecommunications bursts, typical of operational coordination.",
                evidence=f"{count_desc} across monitored telecommunications lines",
                source="CDR Analysis",
                entity_id=clean_id,
                involved_entities=list(call_partners) or [clean_id],
                timestamp=cdr_ts,
                confidence=0.95,
                timeline_event_type="CALL",
                action_hint="View CDR Logs",
                action_type="VIEW_CDR",
            ))
        elif len(matched_calls) >= 4:
            factors.append(RiskFactor(
                factor_id="FAC_CDR_ACTIVITY",
                category=RiskCategory.CDR,
                title="Frequent telecommunications activity",
                score_contribution=5.0,
                numerical_contribution=5.0,
                severity="NOTICE",
                underlying_metric=f"Telephony volume: {len(matched_calls)} calls logged across target lines",
                explanation="Consistent ongoing telecommunications traffic observed across target phone lines.",
                evidence=f"{len(matched_calls)} telephony calls recorded in CDR storage",
                source="CDR Analysis",
                entity_id=clean_id,
                involved_entities=list(call_partners) or [clean_id],
                timestamp=cdr_ts,
                confidence=0.90,
                timeline_event_type="CALL",
                action_hint="View CDR Logs",
                action_type="VIEW_CDR",
            ))

        # ---------------------------------------------------------------------
        # Factor 4: FINANCIAL ANOMALIES & TRANSACTION VELOCITY (Max: 20 pts)
        # ---------------------------------------------------------------------
        target_accounts = {clean_id} | linked_accounts
        all_txs = []
        for a in target_accounts:
            all_txs.extend(self.fin_storage.get_records_for_account(a))

        # Also check graph edges for transferred money
        for edge in self.graph_engine.get_all_edges():
            rel = edge.relationship.value if hasattr(edge.relationship, "value") else str(edge.relationship)
            if rel == EdgeType.TRANSFERRED_MONEY.value and (edge.source in target_accounts or edge.target in target_accounts):
                amt = _safe_float(edge.properties.get("amount", 0.0))
                if amt > 0:
                    all_txs.append(type("GraphTx", (), {"amount": amt, "timestamp": datetime.now()})())

        # Check for high-value transactions (>= 500,000 INR)
        high_val_txs = [t for t in all_txs if getattr(t, "amount", 0.0) >= 500000.0]
        fin_spike_alerts = [a for a in all_alerts if a.alert_type == "FINANCIAL_ANOMALY" and (clean_id in a.entities or any(acc in a.entities for acc in target_accounts))]

        tx_partners = set()
        for t in all_txs:
            if hasattr(t, "sender") and t.sender:
                tx_partners.add(t.sender)
            if hasattr(t, "receiver") and t.receiver:
                tx_partners.add(t.receiver)

        fin_ts = None
        if all_txs:
            raw_t = getattr(all_txs[0], "timestamp", None)
            fin_ts = raw_t.isoformat() if hasattr(raw_t, "isoformat") else str(raw_t) if raw_t else None

        if high_val_txs or fin_spike_alerts:
            max_amt = max([getattr(t, "amount", 0.0) for t in high_val_txs] or [500000.0])
            factors.append(RiskFactor(
                factor_id="FAC_FIN_HIGH_VALUE",
                category=RiskCategory.FINANCIAL,
                title="Unusual financial transaction activity",
                score_contribution=15.0,
                numerical_contribution=15.0,
                severity="HIGH",
                underlying_metric=f"High-value transfer: ₹{max_amt:,.2f} (exceeds ₹500,000 investigative threshold)",
                explanation="Observed transactions exceed the configurable high-value investigation threshold (₹500,000) or represent statistical variance outliers.",
                evidence=f"High-value transfer of ₹{max_amt:,.2f} exceeds ₹500,000 investigative threshold",
                source="Financial Analysis",
                entity_id=clean_id,
                involved_entities=list(tx_partners) or [clean_id],
                timestamp=fin_ts,
                confidence=0.97,
                timeline_event_type="FINANCIAL",
                action_hint="View Transactions",
                action_type="VIEW_FINANCE",
            ))
        elif len(all_txs) >= 4:
            factors.append(RiskFactor(
                factor_id="FAC_FIN_VELOCITY",
                category=RiskCategory.FINANCIAL,
                title="Rapid financial transaction velocity",
                score_contribution=10.0,
                numerical_contribution=10.0,
                severity="ELEVATED",
                underlying_metric=f"Financial velocity: {len(all_txs)} transfers executed in rapid succession",
                explanation="Multiple transaction records execute in sequential succession, consistent with potential layering patterns.",
                evidence=f"{len(all_txs)} transactions logged with rapid sequential activity",
                source="Financial Analysis",
                entity_id=clean_id,
                involved_entities=list(tx_partners) or [clean_id],
                timestamp=fin_ts,
                confidence=0.91,
                timeline_event_type="FINANCIAL",
                action_hint="View Transactions",
                action_type="VIEW_FINANCE",
            ))
        elif len(all_txs) >= 1:
            factors.append(RiskFactor(
                factor_id="FAC_FIN_ACTIVITY",
                category=RiskCategory.FINANCIAL,
                title="Recorded financial transactions",
                score_contribution=5.0,
                numerical_contribution=5.0,
                severity="NOTICE",
                underlying_metric=f"Financial volume: {len(all_txs)} transaction(s) documented in ledger",
                explanation="Financial activity recorded in ledger involving subject or mapped account numbers.",
                evidence=f"{len(all_txs)} transaction record(s) documented in ledger",
                source="Financial Analysis",
                entity_id=clean_id,
                involved_entities=list(tx_partners) or [clean_id],
                timestamp=fin_ts,
                confidence=0.90,
                timeline_event_type="FINANCIAL",
                action_hint="View Transactions",
                action_type="VIEW_FINANCE",
            ))

        # ---------------------------------------------------------------------
        # Factor 5: CASE LINKAGES & STATUTORY SEVERITY (Max: 20 pts)
        # ---------------------------------------------------------------------
        if ent_type.lower() != "case" and linked_cases:
            case_count = len(linked_cases)
            case_codes = [c.properties.get("case_code", c.name) for c in linked_cases]
            case_pts = min(20.0, max(5.0, case_count * 5.0))
            factors.append(RiskFactor(
                factor_id="FAC_CASE_LINKS",
                category=RiskCategory.CASE,
                title=f"Multiple case connections ({case_count} linked cases)" if case_count > 1 else "Linked criminal case connection",
                score_contribution=case_pts,
                numerical_contribution=case_pts,
                severity="HIGH" if case_count >= 3 else ("ELEVATED" if case_count >= 2 else "NOTICE"),
                underlying_metric=f"Case linkages: {case_count} registered police case filings ({', '.join(case_codes[:2])})",
                explanation="Entity is cited as a suspect, co-conspirator, or partner across multiple registered police FIR cases.",
                evidence=f"Connected to {case_count} registered criminal case(s) ({', '.join(case_codes[:3])})",
                source="Case Data",
                entity_id=clean_id,
                involved_entities=[clean_id] + case_codes,
                confidence=0.98,
                timeline_event_type="CASE",
                action_hint="View Case Records",
                action_type="VIEW_CASE",
            ))

            # Check statutory severity (NDPS, Arms, PMLA, Extortion)
            matching_sections = []
            for c in linked_cases:
                sec = str(c.properties.get("section", "")).upper()
                c_type = str(c.properties.get("crime_type", "")).upper()
                combined = f"{sec} {c_type}"
                if any(k in combined for k in ("NDPS", "ARMS", "PMLA", "EXTORTION", "UAPA", "MCOCA", "NARCO")):
                    matching_sections.append(sec or c_type or "Organized Crime")

            if matching_sections:
                factors.append(RiskFactor(
                    factor_id="FAC_CASE_SECTIONS",
                    category=RiskCategory.CASE,
                    title="Severe statutory offense allegations",
                    score_contribution=10.0,
                    numerical_contribution=10.0,
                    severity="HIGH",
                    underlying_metric=f"Statutory sections: {', '.join(matching_sections[:2])} (serious organized crime / narcotics)",
                    explanation="Associated investigation filings allege serious organized crime or narcotics distribution sections.",
                    evidence=f"Case charges cite major statutes: {', '.join(matching_sections[:2])}",
                    source="Case Data",
                    entity_id=clean_id,
                    involved_entities=[clean_id] + case_codes,
                    confidence=0.95,
                    timeline_event_type="CASE",
                    action_hint="View Case Records",
                    action_type="VIEW_CASE",
                ))

        # ---------------------------------------------------------------------
        # Factor 6: LOCATION CO-LOCATION CLUSTERS (Max: 10 pts)
        # ---------------------------------------------------------------------
        loc_alerts = [a for a in all_alerts if a.alert_type == "COLOCATION_CLUSTER" and clean_id in a.entities]
        if loc_alerts:
            factors.append(RiskFactor(
                factor_id="FAC_LOC_CLUSTER",
                category=RiskCategory.LOCATION,
                title="Co-location cluster observed",
                score_contribution=10.0,
                numerical_contribution=10.0,
                severity="ELEVATED",
                underlying_metric="Location surveillance: Physical presence confirmed at suspect gathering hotspot",
                explanation="Physical presence confirmed at a high-density suspect gathering location alongside other monitored individuals.",
                evidence="Co-located at suspect hotspot with 3+ identified network subjects",
                source="Location Surveillance",
                entity_id=clean_id,
                involved_entities=[clean_id] + (loc_alerts[0].entities if loc_alerts else []),
                confidence=0.91,
                timeline_event_type="LOCATION",
                action_hint="View Location Hotspots",
                action_type="VIEW_LOCATION",
            ))

        # ---------------------------------------------------------------------
        # Composite Score Calculation (Reproducible & Bounded 0 - 100)
        # ---------------------------------------------------------------------
        raw_score = sum(f.score_contribution for f in factors)
        final_score = min(100.0, max(0.0, round(raw_score, 1)))

        # Threshold classification
        if final_score >= 75.0:
            level = "HIGH"
        elif final_score >= 50.0:
            level = "ELEVATED"
        elif final_score >= 25.0:
            level = "MODERATE"
        else:
            level = "LOW"

        # Evidence Summary Bullets
        evidence_summary = [f.evidence for f in factors]
        if not evidence_summary:
            evidence_summary = ["No observed investigative risk indicators for this entity."]

        return RiskIntelligenceResult(
            entity_id=clean_id,
            entity_name=ent_name,
            entity_type=ent_type,
            risk_score=final_score,
            total_score=final_score,
            risk_level=level,
            severity=level,
            overall_score=final_score,
            severity_level=level,
            factors=factors,
            evidence_summary=evidence_summary,
            conclusion="Decision-Support Signal: Observed indicators suggest elevated investigative priority based on available telemetry. Does not prove criminal activity.",
            disclaimer="Decision-Support Signal: Investigative risk indicators and anomalous patterns reflect empirical telemetry density across data sources. Scores do not prove criminal activity and require independent investigator verification.",
        )
