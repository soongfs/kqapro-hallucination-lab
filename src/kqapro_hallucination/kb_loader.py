"""
Clean DataForSPARQL + ValueClass loader without module-level side effects.
Adapted from the current KGB workspace implementation.
"""

import json
from datetime import date
from queue import Queue


class ValueClass:
    def __init__(self, type, value, unit=None):
        self.type = type
        self.value = value
        self.unit = unit

    def __str__(self):
        if self.type == "string":
            return self.value
        if self.type == "quantity":
            v = int(self.value) if self.value - int(self.value) < 1e-5 else self.value
            return f"{v} {self.unit}" if self.unit != "1" else str(v)
        if self.type == "year":
            return str(self.value)
        if self.type == "date":
            return self.value.isoformat()
        return str(self.value)


class DataForSPARQL:
    def __init__(self, kb_path):
        kb = json.load(open(kb_path))
        self.concepts = kb["concepts"]
        self.entities = kb["entities"]

        for con_id, con_info in self.concepts.items():
            con_info["name"] = " ".join(con_info["name"].split())
        for ent_id, ent_info in self.entities.items():
            ent_info["name"] = " ".join(ent_info["name"].split())

        self.attribute_keys = set()
        self.predicates = set()
        self.key_type = {}

        for ent_info in self.entities.values():
            for attr_info in ent_info["attributes"]:
                self.attribute_keys.add(attr_info["key"])
                self.key_type[attr_info["key"]] = attr_info["value"]["type"]
                for qk in attr_info["qualifiers"]:
                    self.attribute_keys.add(qk)
                    for qv in attr_info["qualifiers"][qk]:
                        self.key_type[qk] = qv["type"]
        for ent_info in self.entities.values():
            for rel_info in ent_info["relations"]:
                self.predicates.add(rel_info["predicate"])
                for qk in rel_info["qualifiers"]:
                    self.attribute_keys.add(qk)
                    for qv in rel_info["qualifiers"][qk]:
                        self.key_type[qk] = qv["type"]

        self.attribute_keys = list(self.attribute_keys)
        self.predicates = list(self.predicates)
        self.key_type = {
            key: value if value != "year" else "date"
            for key, value in self.key_type.items()
        }

        for ent_info in self.entities.values():
            for attr_info in ent_info["attributes"]:
                attr_info["value"] = self._parse_value(attr_info["value"])
                for qk, qvs in attr_info["qualifiers"].items():
                    attr_info["qualifiers"][qk] = [self._parse_value(qv) for qv in qvs]
        for ent_info in self.entities.values():
            for rel_info in ent_info["relations"]:
                for qk, qvs in rel_info["qualifiers"].items():
                    rel_info["qualifiers"][qk] = [self._parse_value(qv) for qv in qvs]

    def _parse_value(self, value):
        if value["type"] == "date":
            raw = value["value"]
            p1, p2 = raw.find("/"), raw.rfind("/")
            y, m, d = int(raw[:p1]), int(raw[p1 + 1 : p2]), int(raw[p2 + 1 :])
            return ValueClass("date", date(y, m, d))
        if value["type"] == "year":
            return ValueClass("year", value["value"])
        if value["type"] == "string":
            return ValueClass("string", value["value"])
        if value["type"] == "quantity":
            return ValueClass("quantity", value["value"], value["unit"])
        raise ValueError(f"unsupported value type: {value['type']}")

    def get_direct_concepts(self, ent_id):
        if ent_id in self.entities:
            return self.entities[ent_id]["instanceOf"]
        if ent_id in self.concepts:
            return self.concepts[ent_id]["instanceOf"]
        return []

    def get_all_concepts(self, ent_id):
        ancestors = []
        queue = Queue()
        for concept_id in self.get_direct_concepts(ent_id):
            queue.put(concept_id)
        while not queue.empty():
            concept_id = queue.get()
            ancestors.append(concept_id)
            for concept_parent in self.concepts[concept_id]["instanceOf"]:
                queue.put(concept_parent)
        return ancestors

    def get_name(self, ent_id):
        if ent_id in self.entities:
            return self.entities[ent_id]["name"]
        if ent_id in self.concepts:
            return self.concepts[ent_id]["name"]
        return ent_id

    def is_concept(self, ent_id):
        return ent_id in self.concepts

    def get_attribute_facts(self, ent_id, key=None, unit=None):
        if key:
            facts = []
            for attr_info in self.entities[ent_id]["attributes"]:
                if attr_info["key"] == key:
                    if unit:
                        if attr_info["value"].unit == unit:
                            facts.append(attr_info)
                    else:
                        facts.append(attr_info)
        else:
            facts = self.entities[ent_id]["attributes"]
        return [(fact["key"], fact["value"], fact["qualifiers"]) for fact in facts]

    def get_relation_facts(self, ent_id):
        facts = self.entities[ent_id]["relations"]
        return [
            (fact["predicate"], fact["object"], fact["direction"], fact["qualifiers"])
            for fact in facts
        ]
