from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import uuid

class NodeType(str, Enum):
    PERSON = "Person"
    LOCATION = "Location"
    VEHICLE = "Vehicle"
    PHONE = "Phone"
    BANK_ACCOUNT = "BankAccount"
    CASE = "Case"
    ORGANIZATION = "Organization"
    DATE = "Date"

class EdgeType(str, Enum):
    CALLS = "CALLS"
    TRANSFERRED_MONEY = "TRANSFERRED_MONEY"
    LOCATED_AT = "LOCATED_AT"
    OWNS = "OWNS"
    OPERATES = "OPERATES"
    INVOLVED_IN = "INVOLVED_IN"
    OCCURRED_ON = "OCCURRED_ON"

@dataclass
class Node:
    id: str
    label: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label.value if isinstance(self.label, NodeType) else str(self.label),
            "name": self.name,
            "properties": self.properties
        }

@dataclass
class Edge:
    source: str
    target: str
    relationship: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            rel_str = self.relationship.value if isinstance(self.relationship, EdgeType) else str(self.relationship)
            self.id = f"{self.source}_{rel_str}_{self.target}_{uuid.uuid4().hex[:6]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship.value if isinstance(self.relationship, EdgeType) else str(self.relationship),
            "properties": self.properties
        }
