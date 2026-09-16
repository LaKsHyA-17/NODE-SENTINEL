import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import hashlib
from app.models.graph_models import Node, Edge, NodeType, EdgeType
from app.core.entity_resolver import EntityResolver

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

    # Canonical Regex Patterns
    # Indian Vehicle Registration: 2 letters state code + 1-2 digits district + optional 1-3 letters series + 4 digits
    VEHICLE_REGEX = r'\b([A-Z]{2})[\s\-]?(0?[1-9]|[1-9][0-9])[\s\-]?([A-Z]{1,3})[\s\-]?([0-9]{4})\b'
    # Fallback broader vehicle pattern
    VEHICLE_BROAD_REGEX = r'\b[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{4}\b'

    # Phone Regex: Indian mobile numbers (6-9 followed by 9 digits), optional +91 or 0 prefix, optional hyphens/spaces
    PHONE_REGEX = r'(?:\+?91[\-\s]?|0)?[6-9](?:[\-\s]?\d){9}\b'

    # Bank Account Regex: standard account numbers preceded optionally by ACC / A/C or 9-18 digits with account indicators
    BANK_ACC_REGEX = r'\b(?:ACC|ACC-|A/C|A/C-|Account(?:\s+No\.?|:)?)\s*([0-9]{9,18})\b|\b(?:ACC|ACC-)\d{9,16}\b'
    
    # Currency / Money Regex
    MONEY_REGEX = r'(?:₹|INR|rs\.?|rupees)\s?([\d,]+)'

    # Person with titles / honorifics
    PERSON_TITLE_REGEX = r'\b(?:Mr\.|Mrs\.|Ms\.|Shri|Dr\.|Accused|Suspect|Officer|Inspector|Constable|Operative)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    # Unconfirmed / Casual Person Mentions (named X, associate X, etc.)
    UNCONFIRMED_PERSON_REGEX = r'\b(?:named|alias|associate|unconfirmed|individual|person|contact)\s+([A-Z][a-z]+)\b'

    # Case / FIR Identifier Regex
    CASE_ID_REGEX = r'\b(?:FIR|CASE|CRIM)[-/\s]?[0-9A-Za-z\-]{3,24}\b'

    # Organization Patterns (Commercial firms, banks, gangs, agencies, syndicates)
    ORG_PATTERN_REGEX = r'\b([A-Z][a-zA-Z0-9&]*(?:\s+[A-Z][a-zA-Z0-9&]*)*\s+(?:Pvt\.?\s*Ltd\.?|Ltd\.?|LLC|Corp\.?|Enterprises|Syndicate|Logistics|Trading\s+Co\.?|Bank|Gang|Agency|Trust|Foundation|Builders|Police\s+Station|Department))\b|\b(Axis\s+Bank|HDFC\s+Bank|ICICI\s+Bank|State\s+Bank\s+of\s+India|SBI|Punjab\s+National\s+Bank|PNB)\b'

    # Dates Regex Patterns:
    # 1. ISO format: YYYY-MM-DD
    DATE_ISO_REGEX = r'\b(20\d{2}|19\d{2})[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\b'
    # 2. Indian standard: DD-MM-YYYY or DD/MM/YYYY
    DATE_DMY_REGEX = r'\b(0[1-9]|[12]\d|3[01])[-/.](0[1-9]|1[0-2])[-/.](20\d{2}|19\d{2})\b'
    # 3. Textual dates: 15th August 2024, 12 May 2023, July 16, 2024
    DATE_TEXTUAL_REGEX = r'\b(?:(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(20\d{2}|19\d{2})|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(20\d{2}|19\d{2}))\b'

    # Known Indian Locations & Landmarks
    KNOWN_LOCATIONS = [
        "Ghaziabad", "Delhi", "New Delhi", "Chandni Chowk", "Noida", "Greater Noida",
        "Mumbai", "Sector 62", "Cannaught Place", "Connaught Place", "Okhla", "Okhla Industrial Phase 2",
        "Lajpat Nagar", "Rohini", "Gurugram", "Gurgaon", "Bandra", "Kolkata", "Bengaluru", "Bangalore",
        "Hyderabad", "Pune", "Chennai", "Jaipur", "Lucknow", "Ahmedabad", "Surat", "Kanpur", "Nagpur",
        "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Meerut", "Agra", "Faridabad",
        "Varanasi", "Amritsar", "Navi Mumbai", "Allahabad", "Prayagraj", "Ranchi", "Howrah", "Coimbatore",
        "Jabalpur", "Gwalior", "Vijayawada", "Jodhpur", "Madurai", "Raipur", "Kota", "Guwahati", "Chandigarh"
    ]

    def __init__(self):
        self.vehicle_pattern = re.compile(self.VEHICLE_REGEX, re.IGNORECASE)
        self.vehicle_broad_pattern = re.compile(self.VEHICLE_BROAD_REGEX, re.IGNORECASE)
        self.phone_pattern = re.compile(self.PHONE_REGEX)
        self.bank_acc_pattern = re.compile(self.BANK_ACC_REGEX, re.IGNORECASE)
        self.money_pattern = re.compile(self.MONEY_REGEX, re.IGNORECASE)
        self.person_title_pattern = re.compile(self.PERSON_TITLE_REGEX)
        self.unconfirmed_person_pattern = re.compile(self.UNCONFIRMED_PERSON_REGEX, re.IGNORECASE)
        self.case_id_pattern = re.compile(self.CASE_ID_REGEX, re.IGNORECASE)
        self.org_pattern = re.compile(self.ORG_PATTERN_REGEX, re.IGNORECASE)
        self.date_iso_pattern = re.compile(self.DATE_ISO_REGEX)
        self.date_dmy_pattern = re.compile(self.DATE_DMY_REGEX)
        self.date_textual_pattern = re.compile(self.DATE_TEXTUAL_REGEX, re.IGNORECASE)

    @staticmethod
    def normalize_phone_number(raw_phone: str) -> str:
        """Normalize phone numbers to canonical E.164 (+91XXXXXXXXXX) format via EntityResolver."""
        _, e164, _ = EntityResolver.canonicalize_phone(raw_phone)
        return e164

    @staticmethod
    def normalize_vehicle_registration(raw_vehicle: str) -> str:
        """Normalize vehicle plates to canonical uppercase alphanumeric via EntityResolver."""
        _, clean_plate = EntityResolver.canonicalize_vehicle(raw_vehicle)
        return clean_plate

    @staticmethod
    def normalize_bank_account(raw_acc: str) -> Tuple[str, str]:
        """Extract clean digits from bank account string via EntityResolver."""
        return EntityResolver.canonicalize_account(raw_acc)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract all 8 structured entity types from raw text with canonical normalizations."""
        entities = []
        seen_ids = set()

        # 1. Extract Vehicles
        for match in self.vehicle_pattern.finditer(text):
            raw_v = match.group(0)
            vid, norm_v = EntityResolver.canonicalize_vehicle(raw_v)
            if vid not in seen_ids:
                seen_ids.add(vid)
                entities.append({
                    "id": vid,
                    "name": norm_v,
                    "label": NodeType.VEHICLE.value,
                    "properties": {
                        "registration": norm_v,
                        "raw_match": raw_v,
                        "source": "regex_vehicle_pattern"
                    }
                })

        # Fallback broad vehicle match if not caught
        for match in self.vehicle_broad_pattern.finditer(text):
            raw_v = match.group(0)
            vid, norm_v = EntityResolver.canonicalize_vehicle(raw_v)
            if vid not in seen_ids:
                seen_ids.add(vid)
                entities.append({
                    "id": vid,
                    "name": norm_v,
                    "label": NodeType.VEHICLE.value,
                    "properties": {
                        "registration": norm_v,
                        "raw_match": raw_v,
                        "source": "regex_vehicle_broad"
                    }
                })

        # 2. Extract Phone Numbers (Canonical PHONE_XXXXXXXXXX & E.164 name)
        for match in self.phone_pattern.finditer(text):
            raw_phone = match.group(0).strip()
            pid, canonical_phone, clean_digits = EntityResolver.canonicalize_phone(raw_phone)
            if pid not in seen_ids:
                seen_ids.add(pid)
                entities.append({
                    "id": pid,
                    "name": canonical_phone,
                    "label": NodeType.PHONE.value,
                    "properties": {
                        "phone_number": canonical_phone,
                        "raw_digits": clean_digits,
                        "raw_match": raw_phone,
                        "source": "regex_phone_e164"
                    }
                })

        # 3. Extract Bank Accounts
        for match in self.bank_acc_pattern.finditer(text):
            raw_acc = match.group(0).strip()
            node_id, clean_acc = EntityResolver.canonicalize_account(raw_acc)
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                entities.append({
                    "id": node_id,
                    "name": clean_acc,
                    "label": NodeType.BANK_ACCOUNT.value,
                    "properties": {
                        "account_number": clean_acc,
                        "raw_match": raw_acc,
                        "source": "regex_bank_account"
                    }
                })

        # 4. Extract Case / FIR IDs
        for match in self.case_id_pattern.finditer(text):
            raw_cid = match.group(0).strip()
            cid, norm_cid = EntityResolver.canonicalize_case(raw_cid)
            if cid not in seen_ids:
                seen_ids.add(cid)
                entities.append({
                    "id": cid,
                    "name": norm_cid,
                    "label": NodeType.CASE.value,
                    "properties": {
                        "case_code": norm_cid,
                        "raw_match": raw_cid,
                        "source": "regex_case_id"
                    }
                })

        # 5. Extract Dates
        date_matches = []
        for match in self.date_iso_pattern.finditer(text):
            date_matches.append(match.group(0))
        for match in self.date_dmy_pattern.finditer(text):
            date_matches.append(match.group(0))
        for match in self.date_textual_pattern.finditer(text):
            date_matches.append(match.group(0))

        for raw_date in date_matches:
            clean_date = raw_date.strip()
            did = f"DATE_{clean_date.replace(' ', '_').replace('/', '-').replace(',', '')}"
            if did not in seen_ids:
                seen_ids.add(did)
                entities.append({
                    "id": did,
                    "name": clean_date,
                    "label": NodeType.DATE.value,
                    "properties": {
                        "date_str": clean_date,
                        "raw_match": raw_date,
                        "source": "regex_date_extractor"
                    }
                })

        # 6. Extract Organizations via Regex (Banks, Syndicates, Companies, Police Stations)
        for match in self.org_pattern.finditer(text):
            raw_org = match.group(0).strip()
            # Exclude false positives that are actually locations or titles
            if len(raw_org) >= 4 and not raw_org.startswith("Mr.") and not raw_org.startswith("Suspect"):
                org_clean = " ".join(raw_org.split())
                org_id = f"ORG_{org_clean.replace(' ', '_')}"
                if org_id not in seen_ids:
                    seen_ids.add(org_id)
                    entities.append({
                        "id": org_id,
                        "name": org_clean,
                        "label": NodeType.ORGANIZATION.value,
                        "properties": {
                            "organization_name": org_clean,
                            "raw_match": raw_org,
                            "source": "regex_org_pattern"
                        }
                    })

        # 7. Extract Persons & Locations & Organizations via spaCy (if available)
        if _nlp_spacy:
            doc = _nlp_spacy(text)
            for ent in doc.ents:
                clean_name = ent.text.strip()
                if len(clean_name) < 3:
                    continue

                if ent.label_ == "PERSON" and not any(clean_name.lower() == loc.lower() for loc in self.KNOWN_LOCATIONS):
                    pid = f"PERSON_{clean_name.replace(' ', '_')}"
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        entities.append({
                            "id": pid,
                            "name": clean_name,
                            "label": NodeType.PERSON.value,
                            "properties": {"source": "spacy_ner", "ner_label": "PERSON"}
                        })
                elif ent.label_ in ("GPE", "LOC") or any(clean_name.lower() == loc.lower() for loc in self.KNOWN_LOCATIONS):
                    loc_id = f"LOC_{clean_name.replace(' ', '_')}"
                    if loc_id not in seen_ids:
                        seen_ids.add(loc_id)
                        entities.append({
                            "id": loc_id,
                            "name": clean_name,
                            "label": NodeType.LOCATION.value,
                            "properties": {"source": "spacy_ner", "ner_label": ent.label_}
                        })
                elif ent.label_ == "ORG":
                    org_id = f"ORG_{clean_name.replace(' ', '_')}"
                    if org_id not in seen_ids:
                        seen_ids.add(org_id)
                        entities.append({
                            "id": org_id,
                            "name": clean_name,
                            "label": NodeType.ORGANIZATION.value,
                            "properties": {"source": "spacy_ner", "ner_label": "ORG"}
                        })

        # 8. Person Titles / Honorifics Rule-Based Extractor
        for match in self.person_title_pattern.finditer(text):
            pname = match.group(1).strip()
            full_match = match.group(0).strip()
            pid = f"PERSON_{pname.replace(' ', '_')}"
            if pid not in seen_ids:
                seen_ids.add(pid)
                entities.append({
                    "id": pid,
                    "name": pname,
                    "label": NodeType.PERSON.value,
                    "properties": {
                        "source": "regex_title",
                        "title_context": full_match
                    }
                })

        # 8b. Unconfirmed Person Mentions
        for match in self.unconfirmed_person_pattern.finditer(text):
            pname = match.group(1).strip()
            full_match = match.group(0).strip()
            pid = f"PERSON_{pname.replace(' ', '_')}"
            if pid not in seen_ids:
                seen_ids.add(pid)
                entities.append({
                    "id": pid,
                    "name": pname,
                    "label": NodeType.PERSON.value,
                    "properties": {
                        "source": "unconfirmed_mention",
                        "title_context": full_match
                    }
                })

        # 9. Known Location Keywords Heuristics
        for loc in self.KNOWN_LOCATIONS:
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

        # 10. Canonical Entity Resolution & Deduplication (e.g. merge partial single-token first/last names into canonical full names)
        person_ents = [e for e in entities if e["label"] == NodeType.PERSON.value]
        full_name_ents = [p for p in person_ents if " " in p["name"]]
        if full_name_ents:
            full_names_map = {p["name"]: p for p in full_name_ents}
            resolved_entities = []
            for ent in entities:
                if ent["label"] == NodeType.PERSON.value and " " not in ent["name"]:
                    # If this single-token name matches the first or last name of an existing multi-token person, resolve to canonical
                    canonical_match = next((fn for fn in full_names_map if ent["name"] in fn.split()), None)
                    if canonical_match:
                        # Skip duplicate sub-token entity
                        continue
                resolved_entities.append(ent)
            entities = resolved_entities

        return entities

    def extract_triplets(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract Subject-Predicate-Object triplets to construct network edges.
        Strictly requires supporting textual evidence cues in the sentence/clause.
        Does NOT infer relationships merely because entities appear in the same document.
        """
        relationships = []
        seen_triplets = set()

        # Categorized entity node lookups
        person_nodes = [e for e in entities if e["label"] == NodeType.PERSON.value]
        location_nodes = [e for e in entities if e["label"] == NodeType.LOCATION.value]
        vehicle_nodes = [e for e in entities if e["label"] == NodeType.VEHICLE.value]
        phone_nodes = [e for e in entities if e["label"] == NodeType.PHONE.value]
        account_nodes = [e for e in entities if e["label"] == NodeType.BANK_ACCOUNT.value]
        case_nodes = [e for e in entities if e["label"] == NodeType.CASE.value]
        org_nodes = [e for e in entities if e["label"] == NodeType.ORGANIZATION.value]
        date_nodes = [e for e in entities if e["label"] == NodeType.DATE.value]

        # Relational Evidence Cue Regular Expressions
        re_phone_cue = re.compile(
            r'\b(call(?:ed|ing|s)?|dial(?:ed|ing|s)?|contact(?:ed|ing|s)?|spoke|phoned?|communicat(?:ed|ing|es)?|sms|pinged|used\s+phone|using\s+phone|on\s+phone|on\s+number|mobile|subscriber|sim|phone)\b',
            re.IGNORECASE
        )
        re_vehicle_cue = re.compile(
            r'\b(dr(?:ove|iving|iven|ives?)|own(?:s|ed|ing)?|owner|registered|operat(?:ed|ing|es?)|riding|rode|in\s+vehicle|in\s+car|in\s+bike|vehicle|car|bike|motorcycle|scooter|sedan|suv|truck|plate)\b',
            re.IGNORECASE
        )
        re_case_cue = re.compile(
            r'\b(involved|accused|suspect|booked|named|arrested|wanted|convicted|registered|complaint|fir|case|incident|crime|chargesheet(?:ed)?)\b',
            re.IGNORECASE
        )
        re_location_cue = re.compile(
            r'\b(located|met|spotted|seen|visited|resident|residing|staying|found|at|in|near|hideout|intercepted|arrested|arrived|travel(?:led|ing|ed)?|address)\b',
            re.IGNORECASE
        )
        re_org_cue = re.compile(
            r'\b(operat(?:es|ed|ing)|runs?|running|manage[ds]?|managing|head(?:ed|ing)?|director|works?\s+(?:at|for)|member|associated\s+with|affiliated|syndicate|gang|firm|enterprise|company|bank|agency)\b',
            re.IGNORECASE
        )
        re_money_cue = re.compile(
            r'\b(transfer(?:red|ring|s)?|sent|send|deposit(?:ed|ing|s)?|paid|pay(?:ing)?|wire[ds]?|remit(?:ted|ting|s)?|credited|transacted|money|rupees|inr|₹|rs\.?)\b',
            re.IGNORECASE
        )
        re_person_interact_cue = re.compile(
            r'\b(met|contacted|spoke\s+to|associated\s+with|partnered\s+with|worked\s+with|intercepted\s+with|conspired\s+with|called)\b',
            re.IGNORECASE
        )
        re_date_cue = re.compile(
            r'\b(on|dated|occurred|happened|registered|reported|filed|incident|at)\b',
            re.IGNORECASE
        )

        def _add_rel(src: str, tgt: str, rel: str, props: Dict[str, Any], conf: float = 0.90):
            key = (src, tgt, rel)
            if key not in seen_triplets:
                seen_triplets.add(key)
                props["confidence"] = conf
                relationships.append({
                    "source": src,
                    "target": tgt,
                    "relationship": rel,
                    "properties": props
                })

        # Preserve common honorifics when splitting sentences
        sentences = re.split(r'(?<!Mr)(?<!Ms)(?<!Dr)(?<!Mrs)\.(?=\s)|[!?;\n]+', text)

        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue
            lower_sent = sent_str.lower()

            # Find matching persons in this sentence (by full name, first name, or pronoun if single subject)
            p_in_sent = [
                p for p in person_nodes
                if p["name"].lower() in lower_sent or any(part.lower() in lower_sent.split() for part in p["name"].split() if len(part) >= 3)
            ]
            if not p_in_sent and len(person_nodes) == 1:
                if re.search(r'\b(he|she|they|the accused|the suspect|suspect|accused|subject|operative|him|her)\b', lower_sent):
                    p_in_sent = person_nodes

            # 1. Money transfer: Person → TRANSFERRED_MONEY → BankAccount (Requires money/transfer cues)
            if re_money_cue.search(sent_str):
                money_match = self.money_pattern.search(sent_str)
                amount_val = float(money_match.group(1).replace(",", "")) if money_match else 0.0
                acc_in_sent = [a for a in account_nodes if a["name"].lower() in lower_sent]

                if p_in_sent and acc_in_sent:
                    for p in p_in_sent:
                        for acc in acc_in_sent:
                            _add_rel(p["id"], acc["id"], EdgeType.TRANSFERRED_MONEY.value, {
                                "amount": amount_val,
                                "evidence": sent_str,
                                "predicate": "TRANSFERRED_MONEY"
                            }, conf=0.95 if amount_val > 0 else 0.90)
                elif len(p_in_sent) >= 2 and amount_val > 0:
                    _add_rel(p_in_sent[0]["id"], p_in_sent[1]["id"], EdgeType.TRANSFERRED_MONEY.value, {
                        "amount": amount_val,
                        "evidence": sent_str,
                        "predicate": "TRANSFERRED_MONEY"
                    }, conf=0.92)

            # 2. Vehicle ownership / operation: Person → OWNS → Vehicle (Requires vehicle/driving cues)
            if re_vehicle_cue.search(sent_str) and p_in_sent:
                v_in_sent = [v for v in vehicle_nodes if v["name"].lower() in lower_sent or v.get("properties", {}).get("raw_match", "").lower() in lower_sent]
                for p in p_in_sent:
                    for v in v_in_sent:
                        _add_rel(p["id"], v["id"], EdgeType.OWNS.value, {
                            "evidence": sent_str,
                            "predicate": "OWNS_OR_DRIVES_VEHICLE"
                        }, conf=0.94)

            # 3. Communication / Phone: Person → CALLS → Phone (Requires phone / calling cues)
            if re_phone_cue.search(sent_str) and p_in_sent:
                ph_in_sent = [ph for ph in phone_nodes if ph["name"].lower() in lower_sent or ph.get("properties", {}).get("raw_match", "").lower() in lower_sent]
                for p in p_in_sent:
                    for ph in ph_in_sent:
                        _add_rel(p["id"], ph["id"], EdgeType.CALLS.value, {
                            "evidence": sent_str,
                            "predicate": "CALLS_PHONE"
                        }, conf=0.94)

            # 4. Case involvement: Person → INVOLVED_IN → Case (Requires case / FIR cues)
            if re_case_cue.search(sent_str) and p_in_sent:
                c_in_sent = [c for c in case_nodes if c["name"].lower() in lower_sent or c.get("properties", {}).get("raw_match", "").lower() in lower_sent]
                for p in p_in_sent:
                    for c in c_in_sent:
                        _add_rel(p["id"], c["id"], EdgeType.INVOLVED_IN.value, {
                            "evidence": sent_str,
                            "role": "Suspect",
                            "predicate": "INVOLVED_IN_CASE"
                        }, conf=0.95)

            # 5. Location: Person → LOCATED_AT → Location (Requires location / presence cues)
            if re_location_cue.search(sent_str) and p_in_sent:
                loc_in_sent = [loc for loc in location_nodes if loc["name"].lower() in lower_sent]
                for p in p_in_sent:
                    for loc in loc_in_sent:
                        _add_rel(p["id"], loc["id"], EdgeType.LOCATED_AT.value, {
                            "evidence": sent_str,
                            "predicate": "LOCATED_AT"
                        }, conf=0.90)

            # 6. Organization: Person → OPERATES → Organization (Requires org / management cues)
            if re_org_cue.search(sent_str) and p_in_sent:
                org_in_sent = [org for org in org_nodes if org["name"].lower() in lower_sent or org.get("properties", {}).get("raw_match", "").lower() in lower_sent]
                for p in p_in_sent:
                    for org in org_in_sent:
                        _add_rel(p["id"], org["id"], EdgeType.OPERATES.value, {
                            "evidence": sent_str,
                            "predicate": "OPERATES_ORGANIZATION"
                        }, conf=0.90)

            # 7. Direct interaction between persons (Requires interaction verbs)
            if re_person_interact_cue.search(sent_str) and len(p_in_sent) >= 2:
                _add_rel(p_in_sent[0]["id"], p_in_sent[1]["id"], EdgeType.CALLS.value, {
                    "type": "direct_interaction",
                    "evidence": sent_str,
                    "predicate": "DIRECT_INTERACTION"
                }, conf=0.91)

            # 8. Date occurrence: Case → OCCURRED_ON → Date
            if re_date_cue.search(sent_str):
                d_in_sent = [d for d in date_nodes if d["name"].lower() in lower_sent]
                c_in_sent = [c for c in case_nodes if c["name"].lower() in lower_sent or c.get("properties", {}).get("raw_match", "").lower() in lower_sent]
                for c in c_in_sent:
                    for d in d_in_sent:
                        _add_rel(c["id"], d["id"], EdgeType.OCCURRED_ON.value, {
                            "evidence": sent_str,
                            "predicate": "OCCURRED_ON_DATE"
                        }, conf=0.93)

        return relationships

    def extract_entities_with_provenance(
        self, text: str, source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities attaching cryptographic hash, offsets, calibrated confidence,
        decision-support flags, and surrounding evidence provenance snippets.
        """
        meta = source_metadata or {}
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        base_entities = self.extract_entities(text)

        for ent in base_entities:
            name = ent["name"]
            raw_match = ent.get("properties", {}).get("raw_match", name)
            char_idx = text.find(raw_match)
            if char_idx == -1:
                char_idx = text.find(name)
            
            char_start = max(0, char_idx) if char_idx != -1 else 0
            char_end = char_start + len(raw_match if char_idx != -1 else name)

            # Generate contextual evidence snippet (up to 120 chars around match)
            snippet_start = max(0, char_start - 30)
            snippet_end = min(len(text), char_end + 60)
            evidence_snippet = text[snippet_start:snippet_end].strip()

            # Calibrate confidence score based on entity type and extraction method
            label = ent["label"]
            source_method = ent.get("properties", {}).get("source", "nlp_hybrid_ner")

            if label in (NodeType.PHONE.value, NodeType.VEHICLE.value):
                confidence = 0.98 if "e164" in source_method or "vehicle_pattern" in source_method else 0.94
            elif label == NodeType.BANK_ACCOUNT.value:
                confidence = 0.95
            elif label == NodeType.CASE.value:
                confidence = 0.96
            elif label == NodeType.DATE.value:
                confidence = 0.94
            elif label == NodeType.ORGANIZATION.value:
                confidence = 0.88 if "regex" in source_method else 0.82
            elif label == NodeType.PERSON.value:
                confidence = 0.92 if "title" in source_method else (0.86 if "spacy" in source_method else 0.75)
            elif label == NodeType.LOCATION.value:
                confidence = 0.90 if "heuristic" in source_method else 0.85
            else:
                confidence = 0.80

            # Decision-Support: Flag low confidence (<0.80) or ambiguous items as requiring verification
            is_verified = confidence >= 0.80
            if confidence >= 0.90:
                confidence_tier = "HIGH"
                verification_status = "CONFIRMED_PATTERN"
            elif confidence >= 0.80:
                confidence_tier = "MEDIUM"
                verification_status = "VERIFIED_EXTRACTION"
            else:
                confidence_tier = "LOW_CONFIDENCE_UNVERIFIED"
                verification_status = "REQUIRES_VERIFICATION"

            provenance_obj = {
                "source_file": meta.get("filename", meta.get("source", "unstructured_text")),
                "file_hash": meta.get("file_hash", text_hash),
                "case_id": meta.get("case_id"),
                "char_start": char_start,
                "char_end": char_end,
                "confidence": round(confidence, 2),
                "confidence_tier": confidence_tier,
                "is_verified": is_verified,
                "verification_status": verification_status,
                "evidence_snippet": evidence_snippet,
                "extraction_method": source_method,
                "extracted_at": datetime.now().isoformat(),
            }

            ent["provenance"] = provenance_obj
            if "properties" in ent:
                ent["properties"]["provenance"] = provenance_obj
                ent["properties"]["confidence"] = provenance_obj["confidence"]
                ent["properties"]["confidence_tier"] = confidence_tier
                ent["properties"]["is_verified"] = is_verified
                ent["properties"]["verification_status"] = verification_status

        return base_entities

    def extract_triplets_with_provenance(
        self, text: str, entities: List[Dict[str, Any]], source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract triplets attaching sentence span and evidence provenance."""
        meta = source_metadata or {}
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        triplets = self.extract_triplets(text, entities)

        for trip in triplets:
            evidence_str = trip.get("properties", {}).get("evidence", "")
            char_idx = text.find(evidence_str) if evidence_str else 0
            char_start = char_idx if char_idx != -1 else 0
            char_end = (char_idx + len(evidence_str)) if char_idx != -1 else len(evidence_str)

            trip_conf = trip.get("properties", {}).get("confidence", 0.90)
            is_verified = trip_conf >= 0.80
            conf_tier = "HIGH" if trip_conf >= 0.90 else ("MEDIUM" if trip_conf >= 0.80 else "LOW_CONFIDENCE_UNVERIFIED")
            ver_status = "VERIFIED_EVIDENCE_PROVENANCE" if is_verified else "REQUIRES_VERIFICATION"

            prov = {
                "source_file": meta.get("filename", meta.get("source", "unstructured_text")),
                "file_hash": meta.get("file_hash", text_hash),
                "case_id": meta.get("case_id"),
                "evidence_snippet": evidence_str[:120],
                "char_start": char_start,
                "char_end": char_end,
                "confidence": round(trip_conf, 2),
                "confidence_tier": conf_tier,
                "is_verified": is_verified,
                "extraction_timestamp": datetime.now().isoformat(),
                "verification_status": ver_status,
            }
            trip["provenance"] = prov
            if "properties" in trip:
                trip["properties"]["provenance"] = prov
                trip["properties"]["confidence"] = prov["confidence"]

        return triplets
