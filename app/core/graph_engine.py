"""Small, dependency-light graph storage used when Neo4j is not configured."""
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional

import networkx as nx

from app.models.graph_models import Edge, Node, NodeType


class BaseGraphEngine(ABC):
    @abstractmethod
    def add_node(self, node: Node) -> Node: ...
    @abstractmethod
    def add_edge(self, edge: Edge) -> Edge: ...
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]: ...
    @abstractmethod
    def get_edge(self, edge_id: str) -> Optional[Edge]: ...
    @abstractmethod
    def get_edge_between(self, source_id: str, target_id: str, relationship: Optional[str] = None) -> Optional[Edge]: ...
    @abstractmethod
    def get_all_nodes(self) -> List[Node]: ...
    @abstractmethod
    def get_all_edges(self) -> List[Edge]: ...
    @abstractmethod
    def get_neighbors(self, node_id: str, depth: int = 1) -> Dict[str, List]: ...
    @abstractmethod
    def find_shortest_path(self, source_id: str, target_id: str) -> Dict[str, Any]: ...
    @abstractmethod
    def get_node_investigative_details(self, node_id: str) -> Dict[str, Any]: ...
    @abstractmethod
    def clear(self) -> None: ...


class NetworkXGraphEngine(BaseGraphEngine):
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}

    def add_node(self, node: Node) -> Node:
        existing = self._nodes.get(node.id)
        if existing:
            existing.name = node.name or existing.name
            existing.label = node.label or existing.label
            existing.properties.update(node.properties or {})
            return existing
        self._nodes[node.id] = node
        self.graph.add_node(node.id)
        return node

    def add_edge(self, edge: Edge) -> Edge:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise ValueError("Both edge endpoints must exist before adding an edge")
        self._edges[edge.id] = edge
        self.graph.add_edge(edge.source, edge.target, key=edge.id)
        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return self._edges.get(edge_id)

    def get_edge_between(self, source_id: str, target_id: str, relationship: Optional[str] = None) -> Optional[Edge]:
        for edge in self._edges.values():
            if edge.source == source_id and edge.target == target_id:
                if relationship is None:
                    return edge
                rel_str = edge.relationship.value if hasattr(edge.relationship, "value") else str(edge.relationship)
                target_rel = relationship.value if hasattr(relationship, "value") else str(relationship)
                if rel_str.upper() == target_rel.upper():
                    return edge
        return None

    def get_all_nodes(self) -> List[Node]:
        return list(self._nodes.values())

    def get_all_edges(self) -> List[Edge]:
        return list(self._edges.values())

    def get_neighbors(self, node_id: str, depth: int = 1) -> Dict[str, List]:
        if node_id not in self.graph or depth < 0:
            return {"nodes": [], "edges": []}
        seen = {node_id}
        queue = deque([(node_id, 0)])
        while queue:
            current, level = queue.popleft()
            if level >= depth:
                continue
            for neighbor in set(self.graph.successors(current)) | set(self.graph.predecessors(current)):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, level + 1))
        return {
            "nodes": [self._nodes[n] for n in seen],
            "edges": [e for e in self._edges.values() if e.source in seen and e.target in seen],
        }

    def find_shortest_path(self, source_id: str, target_id: str) -> Dict[str, Any]:
        if source_id not in self._nodes:
            return {
                "path_found": False,
                "message": f"Source entity '{source_id}' not found in graph",
                "path_ids": [],
                "nodes": [],
                "steps": [],
                "edges": [],
            }
        if target_id not in self._nodes:
            return {
                "path_found": False,
                "message": f"Target entity '{target_id}' not found in graph",
                "path_ids": [],
                "nodes": [],
                "steps": [],
                "edges": [],
            }
        if source_id == target_id:
            node = self._nodes[source_id]
            return {
                "path_found": True,
                "length": 0,
                "path_ids": [source_id],
                "nodes": [node.to_dict()],
                "steps": [],
                "edges": [],
            }

        undir = self.graph.to_undirected(as_view=True)
        try:
            path_ids = nx.shortest_path(undir, source=source_id, target=target_id)
        except nx.NetworkXNoPath:
            return {
                "path_found": False,
                "message": f"No connection path found between '{self._nodes[source_id].name}' and '{self._nodes[target_id].name}'",
                "path_ids": [],
                "nodes": [],
                "steps": [],
                "edges": [],
            }
        except Exception as e:
            return {
                "path_found": False,
                "message": str(e),
                "path_ids": [],
                "nodes": [],
                "steps": [],
                "edges": [],
            }

        steps = []
        path_edges = []
        for i in range(len(path_ids) - 1):
            u = path_ids[i]
            v = path_ids[i + 1]
            u_node = self._nodes[u]
            v_node = self._nodes[v]

            matching_edge = None
            forward = True
            for edge in self._edges.values():
                if edge.source == u and edge.target == v:
                    matching_edge = edge
                    forward = True
                    break
                elif edge.source == v and edge.target == u:
                    matching_edge = edge
                    forward = False
                    break

            rel_name = (
                matching_edge.relationship.value
                if (matching_edge and hasattr(matching_edge.relationship, "value"))
                else (str(matching_edge.relationship) if matching_edge else "CONNECTED_TO")
            )

            steps.append({
                "step_index": i + 1,
                "from_id": u,
                "from_name": u_node.name,
                "from_type": u_node.label.value if hasattr(u_node.label, "value") else str(u_node.label),
                "to_id": v,
                "to_name": v_node.name,
                "to_type": v_node.label.value if hasattr(v_node.label, "value") else str(v_node.label),
                "relationship": rel_name,
                "direction": "forward" if forward else "reverse",
                "edge_id": matching_edge.id if matching_edge else None,
            })
            if matching_edge:
                path_edges.append(matching_edge.to_dict())

        return {
            "path_found": True,
            "length": len(path_ids) - 1,
            "path_ids": path_ids,
            "nodes": [self._nodes[nid].to_dict() for nid in path_ids],
            "steps": steps,
            "edges": path_edges,
        }

    def get_node_investigative_details(self, node_id: str) -> Dict[str, Any]:
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Entity '{node_id}' not found in graph")

        dir_neighbors = set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id))
        dir_neighbors.discard(node_id)

        ext_neighbors = set()
        for dn in dir_neighbors:
            for nn in set(self.graph.successors(dn)) | set(self.graph.predecessors(dn)):
                if nn != node_id and nn not in dir_neighbors:
                    ext_neighbors.add(nn)

        cases: List[str] = []
        phones: List[str] = []
        vehicles: List[str] = []
        locations: List[str] = []

        all_related_ids = [node_id] + list(dir_neighbors) + list(ext_neighbors)
        seen_c, seen_p, seen_v, seen_l = set(), set(), set(), set()

        for nid in all_related_ids:
            n = self._nodes.get(nid)
            if not n:
                continue
            lbl = n.label.value if hasattr(n.label, "value") else str(n.label)
            lbl_lower = lbl.lower()

            if "case" in lbl_lower or nid.startswith("CASE_"):
                val = n.name or n.properties.get("case_code") or n.id
                if val not in seen_c:
                    seen_c.add(val)
                    cases.append(val)
            elif "case_id" in n.properties:
                val = str(n.properties["case_id"])
                if val not in seen_c:
                    seen_c.add(val)
                    cases.append(val)
            elif "case_code" in n.properties:
                val = str(n.properties["case_code"])
                if val not in seen_c:
                    seen_c.add(val)
                    cases.append(val)

            if "phone" in lbl_lower or nid.startswith("PHONE_"):
                val = n.properties.get("phone_number") or n.name or n.id
                if val not in seen_p:
                    seen_p.add(val)
                    phones.append(val)
            elif "phone_number" in n.properties:
                val = str(n.properties["phone_number"])
                if val not in seen_p:
                    seen_p.add(val)
                    phones.append(val)

            if "vehicle" in lbl_lower or nid.startswith("VEH_"):
                val = n.properties.get("registration") or n.name or n.id
                if val not in seen_v:
                    seen_v.add(val)
                    vehicles.append(val)
            elif "registration" in n.properties:
                val = str(n.properties["registration"])
                if val not in seen_v:
                    seen_v.add(val)
                    vehicles.append(val)

            if "location" in lbl_lower or nid.startswith("LOC_"):
                val = n.properties.get("address") or n.name or n.id
                if val not in seen_l:
                    seen_l.add(val)
                    locations.append(val)
            elif "location" in n.properties or "address" in n.properties:
                val = str(n.properties.get("address") or n.properties.get("location"))
                if val not in seen_l:
                    seen_l.add(val)
                    locations.append(val)

        # Check face registration status
        face_registered = False
        try:
            from app.core.face_storage import get_face_storage
            f_storage = get_face_storage()
            face_registered = node.id in f_storage.list_identities()
        except Exception:
            face_registered = False

        # Extract risk indicators
        risk_score = 0
        risk_level = "LOW"
        if node.properties:
            if "risk_score" in node.properties:
                try:
                    risk_score = int(node.properties["risk_score"])
                except (ValueError, TypeError):
                    pass
            if "risk_tag" in node.properties:
                risk_level = str(node.properties["risk_tag"]).upper()
            elif "risk_level" in node.properties:
                risk_level = str(node.properties["risk_level"]).upper()

        type_str = node.label.value if hasattr(node.label, "value") else str(node.label)

        return {
            "id": node.id,
            "node_id": node.id,
            "name": node.name,
            "type": type_str,
            "node_type": type_str,
            "properties": node.properties,
            "cases": cases,
            "phones": phones,
            "vehicles": vehicles,
            "locations": locations,
            "direct_count": len(dir_neighbors),
            "indirect_count": len(ext_neighbors),
            "direct_ids": list(dir_neighbors),
            "indirect_ids": list(ext_neighbors),
            "face_registered": face_registered,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

    def clear(self) -> None:
        self.graph.clear()
        self._nodes.clear()
        self._edges.clear()


_engine: Optional[NetworkXGraphEngine] = None


def get_graph_engine() -> NetworkXGraphEngine:
    global _engine
    if _engine is None:
        _engine = NetworkXGraphEngine()
    return _engine
