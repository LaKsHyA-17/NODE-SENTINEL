# -*- coding: utf-8 -*-
"""
Unified Entity Resolver and Identifier Canonicalization Engine for NODE SENTINEL.

Provides deterministic canonical identifier resolution, conservative person name
candidate matching, decision-support ambiguity flags, and multi-source provenance &
alias preservation.

Core Capabilities:
1. Deterministic Identifier Canonicalization:
   - Phone: '+919811223344', '9811223344', '9811-223-344' -> PHONE_9811223344, '+919811223344'
   - Vehicle: 'DL01AB9988', 'dl 01 ab 9988', 'DL-01-AB-9988' -> VEH_DL01AB9988, 'DL01AB9988'
   - Bank Account: 'ACC987654321', 'A/C 987654321', '987654321' -> ACC_987654321, '987654321'
   - Case ID: 'FIR-2024-311', 'CASE_FIR_2024_311', 'fir/2024/311' -> CASE_FIR-2024-311, 'FIR-2024-311'

2. Responsible AI Person Resolution:
   - Exact match -> Confirmed Match (Score: 100)
   - Conservative Fuzzy Candidate -> "Possible Match" (Score: 80-94, is_ambiguous=True, requires verification)
   - Never auto-merges person entities solely on name similarity.

3. Observability & Audit Trail:
   - Preserves all observed raw values in properties["observed_values"]
   - Preserves name aliases in properties["aliases"]
   - Preserves complete provenance history in properties["provenance_history"]
"""
from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.models.graph_models import Node, NodeType

logger = logging.getLogger(__name__)


class EntityResolver:
    """Unified Canonicalization and Entity Resolution Service."""

    # -----------------------------------------------------------------------
    # 1. Identifier Canonicalization
    # -----------------------------------------------------------------------

    @staticmethod
    def canonicalize_phone(raw_phone: str) -> Tuple[str, str, str]:
        """
        Normalize raw phone string into (canonical_id, e164_formatted, clean_10_digits).
        
        Examples:
            '+919811223344' -> ('PHONE_9811223344', '+919811223344', '9811223344')
            '9811223344'    -> ('PHONE_9811223344', '+919811223344', '9811223344')
            '9811-223-344'  -> ('PHONE_9811223344', '+919811223344', '9811223344')
            '09811223344'   -> ('PHONE_9811223344', '+919811223344', '9811223344')
        """
        digits = re.sub(r"\D", "", raw_phone)
        if not digits or len(digits) < 5:
            return "", "", ""

        if len(digits) == 10:
            clean_digits = digits
            e164 = f"+91{digits}"
        elif len(digits) == 12 and digits.startswith("91"):
            clean_digits = digits[2:]
            e164 = f"+{digits}"
        elif len(digits) == 11 and digits.startswith("0"):
            clean_digits = digits[1:]
            e164 = f"+91{clean_digits}"
        elif len(digits) > 10 and "91" in digits[:4]:
            clean_digits = digits[-10:]
            e164 = f"+91{clean_digits}"
        else:
            clean_digits = digits[-10:] if len(digits) >= 10 else digits
            e164 = f"+91{clean_digits}"

        canonical_id = f"PHONE_{clean_digits}"
        return canonical_id, e164, clean_digits

    @staticmethod
    def canonicalize_vehicle(raw_vehicle: str) -> Tuple[str, str]:
        """
        Normalize vehicle plate into (canonical_id, clean_plate).
        
        Examples:
            'DL01AB9988'    -> ('VEH_DL01AB9988', 'DL01AB9988')
            'dl 01 ab 9988' -> ('VEH_DL01AB9988', 'DL01AB9988')
            'DL-01-AB-9988' -> ('VEH_DL01AB9988', 'DL01AB9988')
            'dl.01.ab.9988' -> ('VEH_DL01AB9988', 'DL01AB9988')
        """
        cleaned = re.sub(r"[\s\-\.]", "", raw_vehicle).upper()
        if not cleaned:
            return "", ""
        canonical_id = f"VEH_{cleaned}"
        return canonical_id, cleaned

    @staticmethod
    def canonicalize_account(raw_acc: str) -> Tuple[str, str]:
        """
        Normalize bank account string into (canonical_id, clean_account_number).
        
        Examples:
            'ACC987654321'    -> ('ACC_987654321', '987654321')
            'A/C 987654321'   -> ('ACC_987654321', '987654321')
            '987654321'       -> ('ACC_987654321', '987654321')
            'ACC_987654321'   -> ('ACC_987654321', '987654321')
        """
        upper = raw_acc.strip().upper()
        digits_only = re.sub(r"\D", "", upper)
        clean_num = digits_only if digits_only else re.sub(r"^(?:ACC|A\/C|AC)[\s\-_]*", "", upper)
        if not clean_num:
            return "", ""
        canonical_id = f"ACC_{clean_num}"
        return canonical_id, clean_num

    @staticmethod
    def canonicalize_case(raw_case: str) -> Tuple[str, str]:
        """
        Normalize case / FIR code into (canonical_id, clean_case_code).
        
        Examples:
            'FIR-2024-311'      -> ('CASE_FIR-2024-311', 'FIR-2024-311')
            'CASE_FIR_2024_311' -> ('CASE_FIR-2024-311', 'FIR-2024-311')
            'fir 2024 311'      -> ('CASE_FIR-2024-311', 'FIR-2024-311')
            'FIR/2024/311'      -> ('CASE_FIR-2024-311', 'FIR-2024-311')
        """
        upper = raw_case.strip().upper()
        without_prefix = re.sub(r"^CASE[\s_\-]*", "", upper)
        # Standardize separators to hyphens
        normalized_code = re.sub(r"[\s_/\\]+", "-", without_prefix)
        if not normalized_code:
            return "", ""
        canonical_id = f"CASE_{normalized_code}"
        return canonical_id, normalized_code

    @staticmethod
    def canonicalize_person(raw_name: str) -> Tuple[str, str]:
        """
        Normalize person name into (canonical_id, clean_title_cased_name).
        
        Examples:
            'Tariq Ahmad' -> ('PERSON_TARIQ_AHMAD', 'Tariq Ahmad')
            'tariq ahmad' -> ('PERSON_TARIQ_AHMAD', 'Tariq Ahmad')
        """
        cleaned_words = [w.capitalize() for w in raw_name.strip().split() if w]
        clean_name = " ".join(cleaned_words)
        canonical_id = f"PERSON_{clean_name.replace(' ', '_').upper()}"
        return canonical_id, clean_name

    # -----------------------------------------------------------------------
    # 2. Person Name Matching & Ambiguity Handling
    # -----------------------------------------------------------------------

    @classmethod
    def match_person_name(cls, query: str, candidate_name: str) -> Dict[str, Any]:
        """
        Compare query string against a person candidate name.
        
        Returns:
            Dict with match_score (0-100), match_type ('exact', 'possible_match', 'none'),
            is_ambiguous (bool), and verification_status.
        """
        q_clean = " ".join(query.strip().lower().split())
        c_clean = " ".join(candidate_name.strip().lower().split())

        if not q_clean or not c_clean:
            return {
                "match_score": 0,
                "match_type": "none",
                "is_ambiguous": False,
                "is_exact_match": False,
                "verification_status": "NO_MATCH",
            }

        # 1. Exact Name Match
        if q_clean == c_clean:
            return {
                "match_score": 100,
                "match_type": "exact",
                "is_ambiguous": False,
                "is_exact_match": True,
                "verification_status": "CONFIRMED_MATCH",
                "label": "Confirmed Exact Match",
            }

        # 2. Name Prefix / Token Subset Match
        q_tokens = q_clean.split()
        c_tokens = c_clean.split()

        # If query is single token matching first or last name
        if len(q_tokens) == 1 and q_tokens[0] in c_tokens:
            return {
                "match_score": 90,
                "match_type": "possible_match",
                "is_ambiguous": True,
                "is_exact_match": False,
                "verification_status": "REQUIRES_INVESTIGATOR_VERIFICATION",
                "label": "Possible Match (Name Token Match)",
            }

        # 3. Conservative Fuzzy Similarity (SequenceMatcher / Levenshtein ratio)
        similarity = difflib.SequenceMatcher(None, q_clean, c_clean).ratio()

        if similarity >= 0.85:
            score = int(round(similarity * 100))
            return {
                "match_score": score,
                "match_type": "possible_match",
                "is_ambiguous": True,
                "is_exact_match": False,
                "verification_status": "REQUIRES_INVESTIGATOR_VERIFICATION",
                "label": f"Possible Match ({score}% Fuzzy Similarity)",
            }

        # 4. Token Set Overlap
        overlap = set(q_tokens) & set(c_tokens)
        if overlap and len(q_tokens) > 1 and len(c_tokens) > 1:
            overlap_ratio = len(overlap) / max(len(q_tokens), len(c_tokens))
            if overlap_ratio >= 0.5:
                score = int(round(overlap_ratio * 85))
                return {
                    "match_score": score,
                    "match_type": "possible_match",
                    "is_ambiguous": True,
                    "is_exact_match": False,
                    "verification_status": "REQUIRES_INVESTIGATOR_VERIFICATION",
                    "label": f"Possible Match ({score}% Token Overlap)",
                }

        return {
            "match_score": int(round(similarity * 60)),
            "match_type": "none",
            "is_ambiguous": False,
            "is_exact_match": False,
            "verification_status": "NO_MATCH",
        }

    # -----------------------------------------------------------------------
    # 3. Multi-Source Provenance & Alias Preservation
    # -----------------------------------------------------------------------

    @classmethod
    def merge_node_properties(cls, existing_node: Node, new_props: Dict[str, Any]) -> Node:
        """
        Merge new properties into existing graph node, preserving:
        1. All observed raw values in properties['observed_values']
        2. All aliases in properties['aliases']
        3. Multi-source provenance history in properties['provenance_history']
        """
        props = existing_node.properties

        # 1. Preserve Observed Values (e.g. raw phone formats, raw vehicle plates)
        if "observed_values" not in props or not isinstance(props["observed_values"], list):
            props["observed_values"] = []

        raw_match = new_props.get("raw_match") or new_props.get("name") or new_props.get("registration")
        if raw_match and raw_match not in props["observed_values"]:
            props["observed_values"].append(raw_match)

        # 2. Preserve Aliases
        if "aliases" not in props or not isinstance(props["aliases"], list):
            props["aliases"] = []

        new_aliases = new_props.get("aliases", [])
        if isinstance(new_aliases, list):
            for a in new_aliases:
                if a and a not in props["aliases"]:
                    props["aliases"].append(a)
        elif isinstance(new_aliases, str) and new_aliases:
            if new_aliases not in props["aliases"]:
                props["aliases"].append(new_aliases)

        # 3. Preserve Provenance History
        if "provenance_history" not in props or not isinstance(props["provenance_history"], list):
            props["provenance_history"] = []

        new_prov = new_props.get("provenance")
        if new_prov and isinstance(new_prov, dict):
            # Check if identical record already recorded
            existing_hashes = {p.get("file_hash") for p in props["provenance_history"] if isinstance(p, dict)}
            if new_prov.get("file_hash") not in existing_hashes:
                props["provenance_history"].append(new_prov)
        elif new_props.get("source_file"):
            prov_stub = {
                "source_file": new_props.get("source_file"),
                "file_hash": new_props.get("file_hash"),
                "case_id": new_props.get("case_id"),
            }
            existing_sources = {p.get("source_file") for p in props["provenance_history"] if isinstance(p, dict)}
            if prov_stub["source_file"] not in existing_sources:
                props["provenance_history"].append(prov_stub)

        # Merge standard scalar properties without overwriting lists
        for k, v in new_props.items():
            if k not in ("observed_values", "aliases", "provenance_history"):
                props[k] = v

        return existing_node


# Singleton instance
_entity_resolver_instance: Optional[EntityResolver] = None


def get_entity_resolver() -> EntityResolver:
    """Return singleton instance of EntityResolver."""
    global _entity_resolver_instance
    if _entity_resolver_instance is None:
        _entity_resolver_instance = EntityResolver()
    return _entity_resolver_instance
