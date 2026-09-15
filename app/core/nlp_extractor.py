import re
import logging
from typing import List, Dict, Any, Tuple
from app.models.graph_models import Node, Edge, NodeType, EdgeType

logger = logging.getLogger(__name__)

# Try importing spaCy optional model
_nlp_spacy = None
try:
    import spacy
    try:
        _nlp_spacy = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy en_core_web_sm model.")
    except Exception:
        logger.info("spaCy model en_core_web_sm not downloaded. Falling back to rule-based NLP parser.")
except ImportError:
    logger.info("spaCy package not available. Using pure regex rule-based NLP parser.")


class NLPExtractor:
    """Natural Language Processing engine for extracting domain entities & relationships from police reports/FIRs."""

    # Regex Patterns
    VEHICLE_REGEX = r'\b[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{4}\b'
    PHONE_REGEX = r'(?:\+91[\-\s]?)?[6-9]\d{9}\b'
    BANK_ACC_REGEX = r'\b(?:ACC|ACC-|A/C|A/C-)?\d{9,16}\b'
    MONEY_REGEX = r'(?:₹|INR|rs\.?|rupees)\s?([\d,]+)'
    PERSON_TITLE_REGEX = r'\b(?:Mr\.|Mrs\.|Ms\.|Shri|Accused|Suspect|Officer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    CASE_ID_REGEX = r'\b(?:FIR|CASE|CRIM)-?\d{3,6}\b'

    def __init__(self):
        self.vehicle_pattern = re.compile(self.VEHICLE_REGEX)
        self.phone_pattern = re.compile(self.PHONE_REGEX)
        self.bank_acc_pattern = re.compile(self.BANK_ACC_REGEX)
        self.money_pattern = re.compile(self.MONEY_REGEX, re.IGNORECASE)
        self.person_title_pattern = re.compile(self.PERSON_TITLE_REGEX)
        self.case_id_pattern = re.compile(self.CASE_ID_REGEX, re.IGNORECASE)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured entities from raw text."""
        entities = []
        seen_ids = set()

        # 1. Extract Vehicles
        for match in self.vehicle_pattern.finditer(text):
            plate = match.group(0).replace(" ", "").upper()
            if plate not in seen_ids:
                seen_ids.add(plate)
                entities.append({
                    "id": f"VEH_{plate}",
                    "name": plate,
                    "label": NodeType.VEHICLE.value,
                    "properties": {"registration": plate, "source": "nlp_extracted"}
                })

        # 2. Extract Phone Numbers
        for match in self.phone_pattern.finditer(text):
            raw_phone = match.group(0).replace(" ", "").replace("-", "")
            clean_phone = raw_phone if raw_phone.startswith("+91") else f"+91{raw_phone}"
            if clean_phone not in seen_ids:
                seen_ids.add(clean_phone)
                entities.append({
                    "id": f"PHONE_{clean_phone}",
                    "name": clean_phone,
                    "label": NodeType.PHONE.value,
                    "properties": {"phone_number": clean_phone, "source": "nlp_extracted"}
                })

        # 3. Extract Bank Accounts
        for match in self.bank_acc_pattern.finditer(text):
            acc = match.group(0).strip()
            if acc not in seen_ids:
                seen_ids.add(acc)
                entities.append({
                    "id": f"ACC_{acc}",
                    "name": acc,
                    "label": NodeType.BANK_ACCOUNT.value,
                    "properties": {"account_number": acc, "source": "nlp_extracted"}
                })

        # 4. Extract Case IDs
        for match in self.case_id_pattern.finditer(text):
            cid = match.group(0).upper()
            if cid not in seen_ids:
                seen_ids.add(cid)
                entities.append({
                    "id": f"CASE_{cid}",
                    "name": cid,
                    "label": NodeType.CASE.value,
                    "properties": {"case_code": cid, "source": "nlp_extracted"}
                })

        # 5. Extract Persons & Locations via spaCy or Regex
        if _nlp_spacy:
            doc = _nlp_spacy(text)
            for ent in doc.ents:
                clean_name = ent.text.strip()
                if len(clean_name) < 3:
                    continue
                if ent.label_ == "PERSON":
                    pid = f"PERSON_{clean_name.replace(' ', '_')}"
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        entities.append({
                            "id": pid,
                            "name": clean_name,
                            "label": NodeType.PERSON.value,
                            "properties": {"source": "spacy_ner"}
                        })
                elif ent.label_ in ("GPE", "LOC"):
                    loc_id = f"LOC_{clean_name.replace(' ', '_')}"
                    if loc_id not in seen_ids:
                        seen_ids.add(loc_id)
                        entities.append({
                            "id": loc_id,
                            "name": clean_name,
                            "label": NodeType.LOCATION.value,
                            "properties": {"source": "spacy_ner"}
                        })
                elif ent.label_ == "ORG":
                    org_id = f"ORG_{clean_name.replace(' ', '_')}"
                    if org_id not in seen_ids:
                        seen_ids.add(org_id)
                        entities.append({
                            "id": org_id,
                            "name": clean_name,
                            "label": NodeType.ORGANIZATION.value,
                            "properties": {"source": "spacy_ner"}
                        })

        # Fallback / Regex Person & Location heuristic
        for match in self.person_title_pattern.finditer(text):
            pname = match.group(1).strip()
            pid = f"PERSON_{pname.replace(' ', '_')}"
            if pid not in seen_ids:
                seen_ids.add(pid)
                entities.append({
                    "id": pid,
                    "name": pname,
                    "label": NodeType.PERSON.value,
                    "properties": {"source": "regex_title"}
                })

        # Common known location keywords heuristic if not extracted
        known_locations = ["Ghaziabad", "Delhi", "Chandni Chowk", "Noida", "Mumbai", "Location X", "Sector 62", "Cannaught Place"]
        for loc in known_locations:
            if re.search(r'\b' + re.escape(loc) + r'\b', text, re.IGNORECASE):
                loc_id = f"LOC_{loc.replace(' ', '_')}"
                if loc_id not in seen_ids:
                    seen_ids.add(loc_id)
                    entities.append({
                        "id": loc_id,
                        "name": loc,
                        "label": NodeType.LOCATION.value,
                        "properties": {"source": "location_heuristic"}
                    })

        return entities

    def extract_triplets(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract Subject-Predicate-Object triplets to construct network edges."""
        relationships = []
        
        # Helper to map entity string to ID
        person_nodes = [e for e in entities if e["label"] == NodeType.PERSON.value]
        location_nodes = [e for e in entities if e["label"] == NodeType.LOCATION.value]
        vehicle_nodes = [e for e in entities if e["label"] == NodeType.VEHICLE.value]
        phone_nodes = [e for e in entities if e["label"] == NodeType.PHONE.value]
        account_nodes = [e for e in entities if e["label"] == NodeType.BANK_ACCOUNT.value]
        case_nodes = [e for e in entities if e["label"] == NodeType.CASE.value]

        # Keep common honorifics intact (for example, ``Mr. Amit Verma``),
        # otherwise a naive period split separates a name from its evidence.
        sentences = re.split(r'(?<!Mr)(?<!Ms)(?<!Dr)(?<!Mrs)\.(?=\s)|[!?;\n]+', text)

        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            # 1. Money transfer triplets: "X transferred ₹Y to Z" or "sent ₹Y to"
            money_match = self.money_pattern.search(sent_str)
            if money_match:
                amount_val = money_match.group(1).replace(",", "")
                # Find sender (Person) and receiver (Account or Person) in sentence
                p_in_sent = [p for p in person_nodes if p["name"].lower() in sent_str.lower()]
                acc_in_sent = [a for a in account_nodes if a["name"].lower() in sent_str.lower()]
                if p_in_sent and acc_in_sent:
                    relationships.append({
                        "source": p_in_sent[0]["id"],
                        "target": acc_in_sent[0]["id"],
                        "relationship": EdgeType.TRANSFERRED_MONEY.value,
                        "properties": {"amount": float(amount_val), "evidence": sent_str}
                    })
                elif len(p_in_sent) >= 2:
                    sender = p_in_sent[0]
                    receiver = p_in_sent[1]
                    relationships.append({
                        "source": sender["id"],
                        "target": receiver["id"],
                        "relationship": EdgeType.TRANSFERRED_MONEY.value,
                        "properties": {"amount": float(amount_val), "evidence": sent_str}
                    })

            # 2. Location co-presence: "X met Y at/in L" or "seen at L"
            for loc in location_nodes:
                if loc["name"].lower() in sent_str.lower():
                    for p in person_nodes:
                        if p["name"].lower() in sent_str.lower():
                            relationships.append({
                                "source": p["id"],
                                "target": loc["id"],
                                "relationship": EdgeType.LOCATED_AT.value,
                                "properties": {"evidence": sent_str}
                            })

            # 3. Vehicle operation: "X drove/operated vehicle V"
            for v in vehicle_nodes:
                if v["name"].lower() in sent_str.lower():
                    for p in person_nodes:
                        if p["name"].lower() in sent_str.lower():
                            relationships.append({
                                "source": p["id"],
                                "target": v["id"],
                                "relationship": EdgeType.OPERATES.value,
                                "properties": {"evidence": sent_str}
                            })

            # 4. Calls / Phone ownership: "X called Y" or "X uses phone P"
            for ph in phone_nodes:
                if ph["name"].lower() in sent_str.lower():
                    for p in person_nodes:
                        if p["name"].lower() in sent_str.lower():
                            relationships.append({
                                "source": p["id"],
                                "target": ph["id"],
                                "relationship": EdgeType.OWNS.value,
                                "properties": {"evidence": sent_str}
                            })

            # 5. Case involvement: "X involved in Case C"
            for c in case_nodes:
                if c["name"].lower() in sent_str.lower():
                    for p in person_nodes:
                        if p["name"].lower() in sent_str.lower():
                            relationships.append({
                                "source": p["id"],
                                "target": c["id"],
                                "relationship": EdgeType.INVOLVED_IN.value,
                                "properties": {"evidence": sent_str, "role": "Suspect"}
                            })

            # 6. Direct interaction ("met", "contacted", "associated with") between people
            if any(k in sent_str.lower() for k in ["met", "contacted", "spoke to", "associated", "partnered", "worked with"]):
                p_in_sent = [p for p in person_nodes if p["name"].lower() in sent_str.lower()]
                if len(p_in_sent) >= 2:
                    relationships.append({
                        "source": p_in_sent[0]["id"],
                        "target": p_in_sent[1]["id"],
                        "relationship": EdgeType.CALLS.value,
                        "properties": {"type": "direct_interaction", "evidence": sent_str}
                    })

        return relationships

    def extract_entities_with_provenance(
        self, text: str, source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract entities attaching cryptographic hash, offsets, and provenance metadata."""
        import hashlib
        from datetime import datetime

        meta = source_metadata or {}
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        base_entities = self.extract_entities(text)

        for ent in base_entities:
            name = ent["name"]
            char_idx = text.find(name)
            char_start = char_idx if char_idx != -1 else 0
            char_end = char_start + len(name) if char_idx != -1 else len(name)

            ent["provenance"] = {
                "source_file": meta.get("filename", meta.get("source", "unstructured_text")),
                "file_hash": meta.get("file_hash", text_hash),
                "case_id": meta.get("case_id"),
                "char_start": char_start,
                "char_end": char_end,
                "extraction_method": ent.get("properties", {}).get("source", "nlp_hybrid_ner"),
                "extracted_at": datetime.now().isoformat(),
                "confidence": 0.95 if ent["label"] in (NodeType.PHONE.value, NodeType.VEHICLE.value) else 0.88,
            }
            if "properties" in ent:
                ent["properties"]["provenance"] = ent["provenance"]

        return base_entities

    def extract_triplets_with_provenance(
        self, text: str, entities: List[Dict[str, Any]], source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract triplets attaching sentence span and evidence provenance."""
        import hashlib
        from datetime import datetime

        meta = source_metadata or {}
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        triplets = self.extract_triplets(text, entities)

        for trip in triplets:
            evidence_str = trip.get("properties", {}).get("evidence", "")
            char_idx = text.find(evidence_str) if evidence_str else 0
            trip["provenance"] = {
                "source_file": meta.get("filename", meta.get("source", "unstructured_text")),
                "file_hash": meta.get("file_hash", text_hash),
                "case_id": meta.get("case_id"),
                "evidence_snippet": evidence_str[:120],
                "char_start": char_idx if char_idx != -1 else 0,
                "char_end": (char_idx + len(evidence_str)) if char_idx != -1 else len(evidence_str),
                "extraction_timestamp": datetime.now().isoformat(),
                "verification_status": "UNVERIFIED_EVIDENCE_PROVENANCE",
            }
            if "properties" in trip:
                trip["properties"]["provenance"] = trip["provenance"]

        return triplets
