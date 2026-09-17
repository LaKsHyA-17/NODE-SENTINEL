from typing import Optional
from fastapi import APIRouter, HTTPException, Query


from app.core.anomaly_detector import AnomalyDetector
from app.core.graph_analytics import GraphAnalytics
from app.core.graph_engine import get_graph_engine
from app.models.graph_models import NodeType

router = APIRouter(tags=["network"])


def _dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


def graph_payload(nodes, edges):
    return {
        "nodes": [
            {
                "id": n.id,
                "label": n.name,
                "name": n.name,
                "group": n.label.value if hasattr(n.label, "value") else str(n.label),
                "node_type": n.label.value if hasattr(n.label, "value") else str(n.label),
                "title": f"{(n.label.value if hasattr(n.label, 'value') else str(n.label))}: {n.name}",
                "properties": n.properties
            }
            for n in nodes
        ],
        "edges": [
            {
                "id": e.id,
                "from": e.source,
                "to": e.target,
                "label": e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship),
                "arrows": "to",
                "properties": e.properties
            }
            for e in edges
        ],
        "stats": {"nodes": len(nodes), "edges": len(edges)}
    }


@router.get("/network/graph")
def get_graph():
    graph = get_graph_engine()
    return graph_payload(graph.get_all_nodes(), graph.get_all_edges())


@router.get("/network/overview")
def network_overview():
    """Presentation-friendly summary derived from the current live graph."""
    graph = get_graph_engine()
    nodes = graph.get_all_nodes()
    by_type = {node_type.value: 0 for node_type in NodeType}
    for node in nodes:
        lbl = node.label.value if hasattr(node.label, "value") else str(node.label)
        by_type[lbl] = by_type.get(lbl, 0) + 1
    alerts = AnomalyDetector(graph).get_all_alerts()
    high_risk = [node for node in nodes if str(node.properties.get("risk_tag", "")).upper() in {"HIGH", "CRITICAL"}]
    return {"entity_count": len(nodes), "relationship_count": len(graph.get_all_edges()), "high_risk_count": len(high_risk), "alert_count": len(alerts), "by_type": by_type}


def _normalize_phone_digits(val: str) -> str:
    digits = "".join(c for c in str(val) if c.isdigit())
    if digits.startswith("91") and len(digits) == 12:
        return digits[2:]
    return digits


def _normalize_alnum(val: str) -> str:
    return "".join(c for c in str(val) if c.isalnum()).upper()


@router.get("/network/search")
def search_network(query: str = Query(min_length=1), depth: int = Query(1, ge=0, le=4)):
    """
    Upgraded graph search across all entity types:
    - Person (ID, name, alias)
    - Phone (raw or normalized digits)
    - Vehicle (raw or normalized plate alphanumeric)
    - Case / FIR (case code, FIR number)
    - Bank Account (account number)
    - Location / Address
    - Organization / Document
    Exact ID matches are prioritized first.
    Never invents entities. Returns clean empty payload if zero matches.
    """
    graph = get_graph_engine()
    q = query.strip()
    if not q:
        return graph_payload([], [])

    q_lower = q.lower()
    q_digits = _normalize_phone_digits(q)
    q_alnum = _normalize_alnum(q)

    exact_matches = []
    high_matches = []
    partial_matches = []

    for node in graph.get_all_nodes():
        nid_lower = node.id.lower()
        name_lower = node.name.lower()
        props_str = " ".join(str(v).lower() for v in node.properties.values()) if node.properties else ""

        # 1. Exact ID match
        if nid_lower == q_lower:
            exact_matches.append((100, node.id))
            continue

        # 2. Exact Name match
        if name_lower == q_lower:
            exact_matches.append((95, node.id))
            continue

        # 3. Normalized Phone match
        matched_phone = False
        if len(q_digits) >= 6:
            for pk in ["phone", "phone_number", "mobile", "contact"]:
                pv = node.properties.get(pk) if node.properties else None
                if pv and q_digits in _normalize_phone_digits(str(pv)):
                    high_matches.append((85, node.id))
                    matched_phone = True
                    break
            if not matched_phone and "phone" in nid_lower and q_digits in _normalize_phone_digits(nid_lower):
                high_matches.append((85, node.id))
                matched_phone = True

        if matched_phone:
            continue

        # 4. Normalized Vehicle plate match
        matched_veh = False
        if len(q_alnum) >= 4:
            for vk in ["registration", "plate", "vehicle_number", "vehicle_plate"]:
                vv = node.properties.get(vk) if node.properties else None
                if vv and q_alnum in _normalize_alnum(str(vv)):
                    high_matches.append((80, node.id))
                    matched_veh = True
                    break
            if not matched_veh and ("veh" in nid_lower or "dl" in nid_lower or "up" in nid_lower) and q_alnum in _normalize_alnum(nid_lower):
                high_matches.append((80, node.id))
                matched_veh = True

        if matched_veh:
            continue

        # 5. Case ID / Case Code match
        matched_case = False
        for ck in ["case_id", "case_code", "fir_number"]:
            cv = node.properties.get(ck) if node.properties else None
            if cv and (q_lower in str(cv).lower() or q_alnum in _normalize_alnum(str(cv))):
                high_matches.append((78, node.id))
                matched_case = True
                break
        if matched_case:
            continue

        # 6. Bank Account match
        matched_bank = False
        for bk in ["account_number", "account", "bank_account"]:
            bv = node.properties.get(bk) if node.properties else None
            if bv and (q_lower in str(bv).lower() or q_alnum in _normalize_alnum(str(bv))):
                high_matches.append((75, node.id))
                matched_bank = True
                break
        if matched_bank:
            continue

        # 7. Substring in Name or ID
        if q_lower in name_lower or q_lower in nid_lower:
            partial_matches.append((60, node.id))
            continue

        # 8. Substring in properties (address, location, role, alias)
        if q_lower in props_str:
            partial_matches.append((50, node.id))

    # Order matches: exact first, then high, then partial
    sorted_candidates = sorted(exact_matches + high_matches + partial_matches, key=lambda x: x[0], reverse=True)
    seen_ids = set()
    match_ids = []
    for _, nid in sorted_candidates:
        if nid not in seen_ids:
            seen_ids.add(nid)
            match_ids.append(nid)

    if not match_ids:
        payload = graph_payload([], [])
        payload["query"] = query
        payload["match_count"] = 0
        return payload

    selected_nodes, selected_edges = {}, {}
    for node_id in match_ids:
        neighborhood = graph.get_neighbors(node_id, depth)
        for n in neighborhood["nodes"]:
            selected_nodes[n.id] = n
        for e in neighborhood["edges"]:
            selected_edges[e.id] = e

    payload = graph_payload(list(selected_nodes.values()), list(selected_edges.values()))
    payload["query"] = query
    payload["match_count"] = len(match_ids)
    payload["matched_node_ids"] = match_ids
    return payload


@router.get("/entity/{entity_id}/dossier")
def dossier(entity_id: str):
    graph = get_graph_engine()
    node = graph.get_node(entity_id)
    if not node:
        raise HTTPException(status_code=404, detail="Entity not found")
    analytics = GraphAnalytics(graph)
    metrics = analytics.compute_centrality_metrics()
    communities = analytics.detect_communities()
    detector = AnomalyDetector(graph)
    neighborhood = graph.get_neighbors(entity_id)
    lbl = node.label.value if hasattr(node.label, "value") else str(node.label)
    all_alerts = detector.get_all_alerts()
    active_alerts = [_dump(a) for a in all_alerts if entity_id in a.entities]
    return {
        "entity_id": node.id,
        "name": node.name,
        "type": lbl,
        "properties": node.properties,
        "risk": _dump(detector.calculate_investigative_risk_score(entity_id)),
        "centrality": {key: round(values.get(entity_id, 0.0), 4) for key, values in metrics.items()},
        "community_id": communities.get(entity_id),
        "connected_nodes": [n.to_dict() for n in neighborhood["nodes"] if n.id != entity_id],
        "active_alerts": active_alerts
    }


@router.get("/network/meta")
def network_meta():
    """Metadata regarding currently active entity types and relationship types in the knowledge graph."""
    graph = get_graph_engine()
    nodes = graph.get_all_nodes()
    edges = graph.get_all_edges()

    entity_types = sorted(list({
        n.label.value if hasattr(n.label, "value") else str(n.label)
        for n in nodes
    }))
    relationship_types = sorted(list({
        e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship)
        for e in edges
    }))

    return {
        "node_types": entity_types,
        "entity_types": entity_types,
        "relationship_types": relationship_types,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def _resolve_node_id(graph, identifier: str) -> Optional[str]:
    """Resolve an entity ID allowing for common variations (e.g. VEHICLE_ vs VEH_, case differences, exact names)."""
    if not identifier:
        return None
    raw = identifier.strip()
    if graph.get_node(raw):
        return raw

    raw_upper = raw.upper()
    for n in graph.get_all_nodes():
        if n.id.upper() == raw_upper:
            return n.id

    # Handle VEHICLE_ <-> VEH_ prefix mapping
    if raw_upper.startswith("VEHICLE_"):
        alt = "VEH_" + raw_upper[8:]
        if graph.get_node(alt):
            return alt
    elif raw_upper.startswith("VEH_"):
        alt = "VEHICLE_" + raw_upper[4:]
        if graph.get_node(alt):
            return alt

    # Handle name or property exact matches
    for n in graph.get_all_nodes():
        if n.name.upper() == raw_upper:
            return n.id
        if n.properties:
            for k in ["registration", "phone_number", "case_code", "account_number", "case_id"]:
                if str(n.properties.get(k, "")).upper() == raw_upper:
                    return n.id

    return None


@router.get("/network/node/{node_id}/details")
def node_details(node_id: str):
    """Investigation details for an entity (cases, phones, vehicles, direct/indirect connections)."""
    graph = get_graph_engine()
    resolved_id = _resolve_node_id(graph, node_id)
    if not resolved_id:
        raise HTTPException(status_code=404, detail=f"Entity '{node_id}' not found in knowledge graph.")
    try:
        return graph.get_node_investigative_details(resolved_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving details for '{node_id}': {str(e)}")


@router.get("/network/node/{node_id}/expand")
def expand_node(node_id: str, depth: int = Query(1, ge=1, le=3)):
    """Reveal connected entities and relationships around a specific node on demand."""
    graph = get_graph_engine()
    resolved_id = _resolve_node_id(graph, node_id)
    if not resolved_id:
        raise HTTPException(status_code=404, detail=f"Entity '{node_id}' not found in knowledge graph.")
    neighborhood = graph.get_neighbors(resolved_id, depth=depth)
    payload = graph_payload(neighborhood["nodes"], neighborhood["edges"])
    payload["center_node_id"] = resolved_id
    payload["depth"] = depth
    return payload


@router.get("/network/path")
@router.get("/network/shortest-path")
def shortest_path(source: str = Query(..., description="Source entity ID"), target: str = Query(..., description="Target entity ID")):
    """
    Find and trace the shortest investigative path between two entities in the knowledge graph.
    Exposes both /network/path and /network/shortest-path.
    Gracefully handles unknown source/target and disconnected entities without 404/500 errors.
    """
    graph = get_graph_engine()
    source_clean = source.strip()
    target_clean = target.strip()

    resolved_source = _resolve_node_id(graph, source_clean)
    resolved_target = _resolve_node_id(graph, target_clean)

    if not resolved_source:
        return {
            "path_found": False,
            "message": f"Source entity '{source_clean}' not found in the graph.",
            "path_ids": [],
            "nodes": [],
            "steps": [],
            "edges": [],
        }
    if not resolved_target:
        return {
            "path_found": False,
            "message": f"Target entity '{target_clean}' not found in the graph.",
            "path_ids": [],
            "nodes": [],
            "steps": [],
            "edges": [],
        }
    
    res = graph.find_shortest_path(resolved_source, resolved_target)
    if res.get("path_found") and res.get("path_ids"):
        if source_clean != resolved_source and res["path_ids"][0] == resolved_source:
            res["path_ids"][0] = source_clean
        if target_clean != resolved_target and res["path_ids"][-1] == resolved_target:
            res["path_ids"][-1] = target_clean
    return res

