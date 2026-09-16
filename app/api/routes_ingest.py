import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.core.graph_engine import BaseGraphEngine, get_graph_engine
from app.core.nlp_extractor import NLPExtractor
from app.core.document_parser import DocumentParser
from app.core.entity_resolver import EntityResolver
from app.models.graph_models import Edge, EdgeType, Node, NodeType
from app.models.schemas import IngestTextRequest, IngestTextResponse, IngestFileResponse

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _label(value: str) -> NodeType:
    try:
        return NodeType(value)
    except (ValueError, TypeError):
        return NodeType.PERSON


def _edge_type(value: str) -> EdgeType:
    try:
        return EdgeType(value)
    except (ValueError, TypeError):
        return EdgeType.CALLS


def _build_summary_by_type(entities: List[Dict[str, Any]]) -> Dict[str, int]:
    """Helper to aggregate entity counts by category."""
    summary: Dict[str, int] = {}
    for ent in entities:
        lbl = ent.get("label", "Unknown")
        summary[lbl] = summary.get(lbl, 0) + 1
    return summary


def _process_graph_ingestion(
    graph: BaseGraphEngine,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    source_file: str,
    file_hash: str,
    case_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Ingest extracted entities and relationships into Knowledge Graph,
    tracking created nodes vs merged existing entities, newly created edges,
    and collecting actionable decision-support warnings for low-confidence data.
    """
    created_nodes: List[Dict[str, Any]] = []
    merged_entities: List[Dict[str, Any]] = []
    created_edges: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # 1. Collect low-confidence / unverified extraction warnings
    for ent in entities:
        prov = ent.get("provenance", {})
        conf = prov.get("confidence", 1.0)
        is_ver = prov.get("is_verified", True)
        if not is_ver or conf < 0.80:
            warnings.append(
                f"Low-confidence entity detected: '{ent['name']}' ({ent.get('label', 'Entity')}) with score {conf}. Requires manual verification."
            )

    for rel in relationships:
        prov = rel.get("provenance", {})
        conf = prov.get("confidence", 1.0)
        is_ver = prov.get("is_verified", True)
        if not is_ver or conf < 0.80:
            warnings.append(
                f"Low-confidence relationship detected: {rel.get('source')} -> {rel.get('relationship')} -> {rel.get('target')} with score {conf}."
            )

    # 2. Node Ingestion & Deduplication Tracking
    for item in entities:
        props = dict(item.get("properties", {}))
        props["source_file"] = source_file
        props["file_hash"] = file_hash
        if case_id:
            props["case_id"] = case_id

        node_id = item["id"]
        existing_node = graph.get_node(node_id)

        if existing_node:
            EntityResolver.merge_node_properties(existing_node, props)
            node_dict = existing_node.to_dict() if hasattr(existing_node, "to_dict") else {
                "id": node_id, "label": item["label"], "name": existing_node.name, "properties": existing_node.properties
            }
            merged_entities.append(node_dict)
        else:
            node_obj = Node(node_id, _label(item["label"]), item["name"], props)
            EntityResolver.merge_node_properties(node_obj, props)
            graph.add_node(node_obj)
            node_dict = node_obj.to_dict() if hasattr(node_obj, "to_dict") else {
                "id": node_id, "label": item["label"], "name": item["name"], "properties": node_obj.properties
            }
            created_nodes.append(node_dict)

    # 3. Relationship Ingestion Tracking
    existing_edges = graph.get_all_edges()
    existing_edges_set = {
        (e.source, e.target, e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship))
        for e in existing_edges
    }

    for item in relationships:
        src_id = item["source"]
        tgt_id = item["target"]
        rel_type = _edge_type(item["relationship"])
        rel_str = rel_type.value if hasattr(rel_type, "value") else str(rel_type)

        if graph.get_node(src_id) and graph.get_node(tgt_id):
            props = dict(item.get("properties", {}))
            props["source_file"] = source_file
            props["file_hash"] = file_hash
            if case_id:
                props["case_id"] = case_id

            edge_obj = Edge(src_id, tgt_id, rel_type, props)
            is_new = (src_id, tgt_id, rel_str) not in existing_edges_set

            graph.add_edge(edge_obj)
            existing_edges_set.add((src_id, tgt_id, rel_str))

            if is_new:
                created_edges.append(edge_obj.to_dict() if hasattr(edge_obj, "to_dict") else {
                    "source": src_id, "target": tgt_id, "relationship": rel_str, "properties": props
                })

    # 4. Associate Case Node if case_id provided
    if case_id:
        case_node_id = f"CASE_{case_id.upper()}"
        existing_case = graph.get_node(case_node_id)
        case_props = {"case_code": case_id.upper(), "source_file": source_file, "file_hash": file_hash}
        case_node = Node(case_node_id, NodeType.CASE, case_id.upper(), case_props)
        graph.add_node(case_node)
        if not existing_case:
            created_nodes.append(case_node.to_dict() if hasattr(case_node, "to_dict") else {
                "id": case_node_id, "label": NodeType.CASE.value, "name": case_id.upper(), "properties": case_props
            })

        for ent in entities:
            if ent["label"] == NodeType.PERSON.value:
                edge_key = (ent["id"], case_node_id, EdgeType.INVOLVED_IN.value)
                if edge_key not in existing_edges_set:
                    c_edge = Edge(
                        ent["id"],
                        case_node_id,
                        EdgeType.INVOLVED_IN,
                        {"source": "fir_ingest", "source_file": source_file, "file_hash": file_hash, "case_id": case_id}
                    )
                    graph.add_edge(c_edge)
                    existing_edges_set.add(edge_key)
                    created_edges.append(c_edge.to_dict() if hasattr(c_edge, "to_dict") else {
                        "source": ent["id"], "target": case_node_id, "relationship": EdgeType.INVOLVED_IN.value, "properties": c_edge.properties
                    })

    return created_nodes, created_edges, merged_entities, warnings


def load_dataset_by_name(graph: BaseGraphEngine, dataset_name: str = "demo_graph"):
    """Load a named dataset from the sample_data directory."""
    clean_name = dataset_name if dataset_name.endswith(".json") else f"{dataset_name}.json"
    dataset_path = Path(__file__).resolve().parents[2] / "sample_data" / clean_name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    with dataset_path.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    for item in dataset["nodes"]:
        graph.add_node(Node(item["id"], _label(item["label"]), item["name"], item.get("properties", {})))
    for item in dataset["edges"]:
        graph.add_edge(Edge(item["source"], item["target"], _edge_type(item["relationship"]), item.get("properties", {})))
    return {
        "status": "success",
        "dataset": clean_name,
        "total_graph_nodes": len(graph.get_all_nodes()),
        "total_graph_edges": len(graph.get_all_edges())
    }


def ingest_sample_batch_data(graph: BaseGraphEngine):
    """Load default demonstration graph dataset."""
    if graph.get_all_nodes():
        return {"status": "already_loaded", "total_graph_nodes": len(graph.get_all_nodes()), "total_graph_edges": len(graph.get_all_edges())}
    return load_dataset_by_name(graph, "demo_graph.json")


@router.get("/datasets")
def list_available_datasets():
    """List all available graph datasets in sample_data directory."""
    data_dir = Path(__file__).resolve().parents[2] / "sample_data"
    files = [f.stem for f in sorted(data_dir.glob("*.json"))]
    return {"datasets": files}


@router.post("/dataset/{dataset_name}")
def switch_dataset(dataset_name: str):
    """Clear and load a specified dataset (e.g. demo_graph or syndicate_network)."""
    graph = get_graph_engine()
    graph.clear()
    return load_dataset_by_name(graph, dataset_name)


@router.post("/batch")
def ingest_batch():
    return ingest_sample_batch_data(get_graph_engine())


@router.post("/demo/reset")
def reset_demo_data():
    """Restore the predictable demo graph after a presentation or experiment."""
    graph = get_graph_engine()
    graph.clear()
    return load_dataset_by_name(graph, "demo_graph.json")


@router.post("/text", response_model=IngestTextResponse)
def ingest_text(request: IngestTextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")

    text_hash = hashlib.sha256(request.text.encode("utf-8")).hexdigest()[:16]

    extractor = NLPExtractor()
    source_meta = {
        "filename": "unstructured_fir_text.txt",
        "source": "text_narrative",
        "file_hash": text_hash,
        "case_id": request.source_case_id
    }
    entities = extractor.extract_entities_with_provenance(request.text, source_metadata=source_meta)
    relationships = extractor.extract_triplets_with_provenance(request.text, entities, source_metadata=source_meta)
    graph = get_graph_engine()

    created_nodes, created_edges, merged_entities, warnings = _process_graph_ingestion(
        graph=graph,
        entities=entities,
        relationships=relationships,
        source_file="unstructured_fir_text.txt",
        file_hash=text_hash,
        case_id=request.source_case_id
    )

    summary_by_type = _build_summary_by_type(entities)

    return {
        "status": "success",
        "extracted_entities_count": len(entities),
        "extracted_relations_count": len(relationships),
        "entities": entities,
        "relationships": relationships,
        "created_nodes": created_nodes,
        "created_edges": created_edges,
        "merged_entities": merged_entities,
        "warnings": warnings,
        "summary_by_type": summary_by_type,
        "file_hash": text_hash,
        "case_id": request.source_case_id,
    }


@router.post("/file", response_model=IngestFileResponse)
async def ingest_file(
    file: UploadFile = File(..., description="FIR report file (TXT, PDF or Image format: png, jpg, jpeg, webp, bmp)"),
    source_case_id: Optional[str] = Form(None)
):
    """Upload a copy of an FIR (TXT, PDF or Image), extract text, parse 8 entity types with NLP, and merge into graph."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

    parser = DocumentParser()
    try:
        extracted_text = await parser.extract_text_from_file(file_bytes, file.filename)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=422, detail=str(re))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(ex)}")

    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No readable text could be extracted from the uploaded document.")

    extractor = NLPExtractor()
    source_meta = {
        "filename": file.filename,
        "source": "file_upload",
        "file_hash": file_hash,
        "case_id": source_case_id
    }
    entities = extractor.extract_entities_with_provenance(extracted_text, source_metadata=source_meta)
    relationships = extractor.extract_triplets_with_provenance(extracted_text, entities, source_metadata=source_meta)
    graph = get_graph_engine()

    created_nodes, created_edges, merged_entities, warnings = _process_graph_ingestion(
        graph=graph,
        entities=entities,
        relationships=relationships,
        source_file=file.filename,
        file_hash=file_hash,
        case_id=source_case_id
    )

    summary_by_type = _build_summary_by_type(entities)

    return {
        "status": "success",
        "filename": file.filename,
        "extracted_text": extracted_text,
        "extracted_entities_count": len(entities),
        "extracted_relations_count": len(relationships),
        "entities": entities,
        "relationships": relationships,
        "created_nodes": created_nodes,
        "created_edges": created_edges,
        "merged_entities": merged_entities,
        "warnings": warnings,
        "summary_by_type": summary_by_type,
        "file_hash": file_hash,
        "case_id": source_case_id,
    }
