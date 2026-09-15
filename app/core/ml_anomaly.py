# -*- coding: utf-8 -*-
"""
NODE SENTINEL - ML Anomaly Detection Engine
Provides multi-dimensional unsupervised anomaly scoring and feature attribution
across graph topology, telephony bursts, financial flow velocity, and co-locations.
Strictly adheres to neutral investigator decision-support standards.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from app.core.cdr_analytics import get_cdr_storage
from app.core.financial_analytics import get_financial_storage
from app.core.graph_analytics import GraphAnalytics
from app.core.graph_engine import BaseGraphEngine, get_graph_engine
from app.models.graph_models import EdgeType, NodeType

logger = logging.getLogger(__name__)


class MLAnomalyDetector:
    """
    Multi-dimensional unsupervised anomaly scoring engine for criminal intelligence.
    Extracts high-dimensional behavioral feature vectors and evaluates statistical &
    structural outlier scores with explainable component attributions.
    """

    FEATURE_NAMES = [
        "betweenness_centrality",
        "degree_centrality",
        "pagerank",
        "cdr_call_density",
        "cdr_unique_interlocutors",
        "fin_velocity_spikes",
        "fin_high_value_ratio",
        "case_involvement_density",
        "colocation_frequency",
    ]

    def __init__(self, graph_engine: Optional[BaseGraphEngine] = None):
        self.graph = graph_engine or get_graph_engine()
        self.analytics = GraphAnalytics(self.graph)
        self.cdr_storage = get_cdr_storage()
        self.fin_storage = get_financial_storage()

    def extract_entity_feature_vector(self, entity_id: str) -> Dict[str, float]:
        """Extract multi-dimensional telemetry features for a given entity."""
        clean_id = entity_id.strip()
        node = self.graph.get_node(clean_id)

        # 1. Graph Centrality
        metrics = self.analytics.compute_centrality_metrics()
        b_val = float(metrics["betweenness"].get(clean_id, 0.0))
        d_val = float(metrics["degree"].get(clean_id, 0.0))
        pr_val = float(metrics["pagerank"].get(clean_id, 0.0))

        # 2. Linked phone & account collection
        linked_phones: Set[str] = {clean_id}
        linked_accounts: Set[str] = {clean_id}
        linked_cases_count = 0
        colocation_count = 0

        if node:
            if "phone_number" in node.properties:
                linked_phones.add(str(node.properties["phone_number"]))
            if "account_number" in node.properties:
                linked_accounts.add(str(node.properties["account_number"]))

            neighborhood = self.graph.get_neighbors(clean_id, depth=1).get("nodes", [])
            for n in neighborhood:
                if n.id == clean_id:
                    continue
                lbl = n.label.value if hasattr(n.label, "value") else str(n.label)
                if lbl == NodeType.PHONE.value:
                    linked_phones.add(n.id)
                    p = n.properties.get("phone_number") or n.name
                    if p:
                        linked_phones.add(str(p))
                elif lbl == NodeType.BANK_ACCOUNT.value:
                    linked_accounts.add(n.id)
                    a = n.properties.get("account_number") or n.name
                    if a:
                        linked_accounts.add(str(a))
                elif lbl == NodeType.CASE.value:
                    linked_cases_count += 1
                elif lbl == NodeType.LOCATION.value:
                    colocation_count += 1

        # 3. CDR Metrics
        all_calls = []
        for p in linked_phones:
            all_calls.extend(self.cdr_storage.get_records_for_phone(p))
        unique_call_ids = set()
        dedup_calls = []
        contacts = set()
        for c in all_calls:
            cid = c.call_id or f"{c.caller}_{c.receiver}_{c.timestamp.isoformat()}"
            if cid not in unique_call_ids:
                unique_call_ids.add(cid)
                dedup_calls.append(c)
                contacts.add(c.caller)
                contacts.add(c.receiver)

        cdr_density = float(len(dedup_calls))
        cdr_interlocutors = float(max(0, len(contacts) - len(linked_phones)))

        # 4. Financial Metrics
        all_txs = []
        for a in linked_accounts:
            all_txs.extend(self.fin_storage.get_records_for_account(a))

        fin_velocity = float(len(all_txs))
        high_val_count = sum(1 for t in all_txs if getattr(t, "amount", 0.0) >= 500000.0)
        high_val_ratio = float(high_val_count / max(1, len(all_txs))) if all_txs else 0.0

        return {
            "betweenness_centrality": b_val,
            "degree_centrality": d_val,
            "pagerank": pr_val,
            "cdr_call_density": cdr_density,
            "cdr_unique_interlocutors": cdr_interlocutors,
            "fin_velocity_spikes": fin_velocity,
            "fin_high_value_ratio": high_val_ratio,
            "case_involvement_density": float(linked_cases_count),
            "colocation_frequency": float(colocation_count),
        }

    def compute_population_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Compute feature matrix across all active entities in graph."""
        nodes = self.graph.get_all_nodes()
        if not nodes:
            return [], np.zeros((0, len(self.FEATURE_NAMES)))

        entity_ids = [n.id for n in nodes]
        rows = []
        for eid in entity_ids:
            feats = self.extract_entity_feature_vector(eid)
            rows.append([feats[k] for k in self.FEATURE_NAMES])

        return entity_ids, np.array(rows, dtype=np.float64)

    def evaluate_entity_anomaly(self, entity_id: str) -> Dict[str, Any]:
        """
        Evaluate multi-dimensional anomaly score (0.0 to 100.0) with feature attributions.
        """
        entity_ids, X = self.compute_population_matrix()
        clean_id = entity_id.strip()

        raw_feats = self.extract_entity_feature_vector(clean_id)
        x_vec = np.array([raw_feats[k] for k in self.FEATURE_NAMES], dtype=np.float64)

        if X.shape[0] < 2:
            # Baseline when population is small
            score = min(100.0, float(np.sum(x_vec) * 10.0))
            return {
                "entity_id": clean_id,
                "ml_anomaly_score": round(score, 1),
                "is_anomalous": score >= 50.0,
                "severity": "HIGH" if score >= 75.0 else ("ELEVATED" if score >= 50.0 else "LOW"),
                "features": raw_feats,
                "feature_attributions": {k: 1.0 / len(self.FEATURE_NAMES) for k in self.FEATURE_NAMES},
                "explanation": "Evaluated against baseline heuristic profile. Requires Investigator Verification.",
            }

        # Calculate population means and standard deviations
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds < 1e-5] = 1.0  # prevent div by zero

        # Normalized z-score vector
        z_scores = np.abs((x_vec - means) / stds)

        # Multi-dimensional Mahalanobis-style distance proxy
        # Weighted aggregation of normalized outlier components
        weights = np.array([2.5, 1.5, 1.2, 2.0, 1.5, 2.2, 2.8, 2.0, 1.8], dtype=np.float64)
        weighted_z = z_scores * weights

        composite_distance = float(np.sum(weighted_z))
        # Logistic sigmoid scaling to [0, 100]
        scaled_score = float(100.0 / (1.0 + np.exp(-0.35 * (composite_distance - 5.0))))
        scaled_score = min(100.0, max(0.0, round(scaled_score, 1)))

        # Feature attributions (proportional contribution to outlier score)
        tot_contrib = float(np.sum(weighted_z))
        if tot_contrib > 1e-5:
            attributions = {
                self.FEATURE_NAMES[i]: round(float(weighted_z[i] / tot_contrib) * 100.0, 1)
                for i in range(len(self.FEATURE_NAMES))
            }
        else:
            attributions = {k: 0.0 for k in self.FEATURE_NAMES}

        # Sort top drivers
        top_drivers = sorted(attributions.items(), key=lambda x: x[1], reverse=True)[:3]
        top_driver_str = ", ".join([f"{k.replace('_', ' ').title()} ({v}%)" for k, v in top_drivers if v > 0])

        sev = "HIGH" if scaled_score >= 75.0 else ("ELEVATED" if scaled_score >= 50.0 else ("MODERATE" if scaled_score >= 25.0 else "LOW"))

        return {
            "entity_id": clean_id,
            "ml_anomaly_score": scaled_score,
            "is_anomalous": scaled_score >= 50.0,
            "severity": sev,
            "features": {k: round(v, 4) for k, v in raw_feats.items()},
            "feature_attributions": attributions,
            "top_drivers": top_driver_str or "Evenly distributed baseline indicators",
            "explanation": f"ML anomaly detector measured {scaled_score}/100 statistical outlier score. Primary drivers: {top_driver_str or 'Baseline'}. Requires Investigator Verification.",
        }

    def detect_population_anomalies(self, threshold: float = 50.0) -> List[Dict[str, Any]]:
        """Rank and return all entities exceeding the anomaly threshold."""
        nodes = self.graph.get_all_nodes()
        results = []
        for n in nodes:
            res = self.evaluate_entity_anomaly(n.id)
            if res["ml_anomaly_score"] >= threshold:
                res["name"] = n.name
                res["type"] = n.label.value if hasattr(n.label, "value") else str(n.label)
                results.append(res)

        results.sort(key=lambda x: x["ml_anomaly_score"], reverse=True)
        return results


# Singleton
_ml_anomaly_instance: Optional[MLAnomalyDetector] = None


def get_ml_anomaly_detector() -> MLAnomalyDetector:
    global _ml_anomaly_instance
    if _ml_anomaly_instance is None:
        _ml_anomaly_instance = MLAnomalyDetector()
    return _ml_anomaly_instance
