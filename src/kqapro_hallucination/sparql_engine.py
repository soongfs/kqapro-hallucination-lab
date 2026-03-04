"""
Local SPARQL engine built on pyoxigraph.
Adapted from the current KGB workspace implementation.
"""

import re
import shutil
from itertools import chain
from pathlib import Path

import pyoxigraph as ox
from tqdm import tqdm


KQA_NS = "urn:kqa/"

_XSD_DOUBLE = ox.NamedNode("http://www.w3.org/2001/XMLSchema#double")
_XSD_INTEGER = ox.NamedNode("http://www.w3.org/2001/XMLSchema#integer")
_XSD_DATE = ox.NamedNode("http://www.w3.org/2001/XMLSchema#date")


def _legal(text: str) -> str:
    return text.replace(" ", "_")


def _nn(raw: str) -> ox.NamedNode:
    if ":" in raw:
        return ox.NamedNode(raw)
    return ox.NamedNode(KQA_NS + raw)


PRED_INSTANCE = "pred:instance_of"
PRED_NAME = "pred:name"
PRED_VALUE = "pred:value"
PRED_UNIT = "pred:unit"
PRED_YEAR = "pred:year"
PRED_DATE = "pred:date"
PRED_FACT_H = "pred:fact_h"
PRED_FACT_R = "pred:fact_r"
PRED_FACT_T = "pred:fact_t"

SPECIAL_PREDICATES = (
    PRED_INSTANCE,
    PRED_NAME,
    PRED_VALUE,
    PRED_UNIT,
    PRED_YEAR,
    PRED_DATE,
    PRED_FACT_H,
    PRED_FACT_R,
    PRED_FACT_T,
)

_STRUCTURAL = frozenset(_nn(_legal(pred)) for pred in SPECIAL_PREDICATES)
_PN = {pred: _nn(_legal(pred)) for pred in SPECIAL_PREDICATES}


def _add_value_quads(buf: list, value, pn) -> ox.BlankNode:
    bn = ox.BlankNode()
    if value.type == "string":
        buf.append(ox.Quad(bn, pn[PRED_VALUE], ox.Literal(value.value)))
    elif value.type == "quantity":
        buf.append(
            ox.Quad(bn, pn[PRED_VALUE], ox.Literal(str(value.value), datatype=_XSD_DOUBLE))
        )
        buf.append(ox.Quad(bn, pn[PRED_UNIT], ox.Literal(value.unit)))
    elif value.type == "year":
        buf.append(
            ox.Quad(bn, pn[PRED_YEAR], ox.Literal(str(value.value), datatype=_XSD_INTEGER))
        )
    elif value.type == "date":
        buf.append(
            ox.Quad(
                bn,
                pn[PRED_YEAR],
                ox.Literal(str(value.value.year), datatype=_XSD_INTEGER),
            )
        )
        buf.append(
            ox.Quad(
                bn,
                pn[PRED_DATE],
                ox.Literal(str(value.value), datatype=_XSD_DATE),
            )
        )
    return bn


def _add_fact_quads(buf: list, h, r, t, pn) -> ox.BlankNode:
    fn = ox.BlankNode()
    buf.append(ox.Quad(fn, pn[PRED_FACT_H], h))
    buf.append(ox.Quad(fn, pn[PRED_FACT_R], r))
    buf.append(ox.Quad(fn, pn[PRED_FACT_T], t))
    return fn


class LocalSparqlEngine:
    def __init__(self, data, store_path: str = ""):
        self.store = ox.Store(store_path) if store_path else ox.Store()
        pn = _PN

        nodes: dict[str, ox.NamedNode] = {}
        for item in chain(data.concepts, data.entities):
            nodes[item] = _nn(item)
        for pred in chain(data.predicates, data.attribute_keys, SPECIAL_PREDICATES):
            nodes[pred] = _nn(_legal(pred))

        batch = 200_000
        buf: list[ox.Quad] = []

        def flush():
            if buf:
                self.store.bulk_extend(buf)
                buf.clear()

        pred_name = pn[PRED_NAME]
        pred_inst = pn[PRED_INSTANCE]

        for item in chain(data.concepts, data.entities):
            buf.append(ox.Quad(nodes[item], pred_name, ox.Literal(data.get_name(item))))
        flush()

        for ent_id in tqdm(data.entities, desc="Building RDF (Oxigraph)"):
            ent = nodes[ent_id]

            for concept_id in data.get_all_concepts(ent_id):
                buf.append(ox.Quad(ent, pred_inst, nodes[concept_id]))

            for key, value, qualifiers in data.get_attribute_facts(ent_id):
                h, r = ent, nodes[key]
                t = _add_value_quads(buf, value, pn)
                buf.append(ox.Quad(h, r, t))
                fn = _add_fact_quads(buf, h, r, t, pn)
                for qk, qvs in qualifiers.items():
                    for qv in qvs:
                        qt = _add_value_quads(buf, qv, pn)
                        buf.append(ox.Quad(fn, nodes[qk], qt))

            for pred, obj_id, direction, qualifiers in data.get_relation_facts(ent_id):
                if direction == "backward":
                    if data.is_concept(obj_id):
                        h, r, t = nodes[obj_id], nodes[pred], ent
                    else:
                        continue
                else:
                    h, r, t = ent, nodes[pred], nodes[obj_id]
                buf.append(ox.Quad(h, r, t))
                fn = _add_fact_quads(buf, h, r, t, pn)
                for qk, qvs in qualifiers.items():
                    for qv in qvs:
                        qt = _add_value_quads(buf, qv, pn)
                        buf.append(ox.Quad(fn, nodes[qk], qt))

            if len(buf) >= batch:
                flush()

        flush()
        if store_path:
            self.store.flush()

    def query(self, sparql_str: str):
        return self.store.query(sparql_str)


def build_or_load_engine(data, store_path: str = "./data/kqa_oxigraph"):
    path = Path(store_path)
    if path.exists():
        try:
            store = ox.Store(str(path))
            quad_count = len(store)
            if quad_count > 1_000_000:
                engine = LocalSparqlEngine.__new__(LocalSparqlEngine)
                engine.store = store
                return engine
        except Exception:
            pass
        shutil.rmtree(path, ignore_errors=True)

    path.mkdir(parents=True, exist_ok=True)
    engine = LocalSparqlEngine(data, store_path=str(path))
    return engine


_RE_ANGLE_URI = re.compile(r"<([^>]+)>")
_SPARQL_PREFIX = (
    f"PREFIX : <{KQA_NS}>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)


def _rewrite_uri_token(match: re.Match) -> str:
    uri = match.group(1)
    if ":" in uri:
        return match.group(0)
    return f":{uri}"


def rewrite_sparql(sparql: str) -> str:
    rewritten = _RE_ANGLE_URI.sub(_rewrite_uri_token, sparql)
    return _SPARQL_PREFIX + rewritten


def _extract_where_body(sparql: str) -> str:
    text = sparql.strip()
    depth = 0
    start = None
    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i]
    return ""


def _flatten_to_triple_patterns(body: str) -> str:
    cleaned = re.sub(r"FILTER\s*\([^)]*\)", "", body)
    cleaned = re.sub(r"ORDER\s+BY\s+[^\}]+", "", cleaned)
    cleaned = re.sub(r"LIMIT\s+\d+", "", cleaned)
    cleaned = re.sub(r"OFFSET\s+\d+", "", cleaned)
    cleaned = cleaned.replace("UNION", " ").replace("OPTIONAL", " ")
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def sparql_to_construct(sparql: str) -> str:
    body = _extract_where_body(sparql)
    if not body:
        return ""
    patterns = _flatten_to_triple_patterns(body)
    raw = f"CONSTRUCT {{\n{patterns}\n}} WHERE {{\n{body}\n}}"
    return rewrite_sparql(raw)


def _lit_py(lit: ox.Literal):
    dt = lit.datatype
    if dt:
        raw_dt = dt.value
        if "double" in raw_dt or "float" in raw_dt or "decimal" in raw_dt:
            try:
                return float(lit.value)
            except ValueError:
                pass
        if "integer" in raw_dt or "int" in raw_dt or "long" in raw_dt or "short" in raw_dt:
            try:
                return int(lit.value)
            except ValueError:
                pass
    return lit.value


def resolve_node(node, store, uri_to_name, include_id=False, bn_idx=None):
    if isinstance(node, ox.NamedNode):
        raw = node.value[len(KQA_NS) :] if node.value.startswith(KQA_NS) else node.value
        if raw in uri_to_name:
            name = uri_to_name[raw]
            return f"{name} ({raw})" if include_id else name
        return raw.replace("_", " ")

    if isinstance(node, ox.BlankNode):
        if bn_idx and node in bn_idx:
            props = bn_idx[node]
            for key in (PRED_VALUE, PRED_YEAR):
                if key in props:
                    value = props[key][0]
                    return _lit_py(value) if isinstance(value, ox.Literal) else str(value)
        for pred_key in (PRED_VALUE, PRED_YEAR):
            nn_p = _nn(_legal(pred_key))
            for quad in store.quads_for_pattern(node, nn_p, None, None):
                value = quad.object
                return _lit_py(value) if isinstance(value, ox.Literal) else str(value)
        return str(node)

    if isinstance(node, ox.Literal):
        return _lit_py(node)

    return str(node)


def extract_subgraph(sparql_str: str, engine, uri_to_name: dict, include_id: bool = True) -> list:
    construct_query = sparql_to_construct(sparql_str)
    if not construct_query:
        return []
    try:
        result_triples = list(engine.query(construct_query))
    except Exception:
        return []

    bn_idx: dict = {}
    for triple in result_triples:
        if isinstance(triple.subject, ox.BlankNode):
            bn_idx.setdefault(triple.subject, {}).setdefault(
                triple.predicate.value, []
            ).append(triple.object)

    ft_pred_nn = _PN[PRED_FACT_T]
    triples = []
    for triple in result_triples:
        if triple.predicate in _STRUCTURAL:
            continue

        raw_rel = triple.predicate.value
        raw_rel = raw_rel[len(KQA_NS) :] if raw_rel.startswith(KQA_NS) else raw_rel
        relation = raw_rel.replace("_", " ")

        subj = triple.subject
        if isinstance(subj, ox.BlankNode):
            props = bn_idx.get(subj, {})
            ft_vals = props.get(PRED_FACT_T, [])
            if ft_vals:
                head = resolve_node(ft_vals[0], engine.store, uri_to_name, include_id, bn_idx)
            else:
                found = False
                for quad in engine.store.quads_for_pattern(subj, ft_pred_nn, None, None):
                    head = resolve_node(
                        quad.object, engine.store, uri_to_name, include_id, bn_idx
                    )
                    found = True
                    break
                if not found:
                    continue
        else:
            head = resolve_node(subj, engine.store, uri_to_name, include_id, bn_idx)

        tail = resolve_node(triple.object, engine.store, uri_to_name, include_id, bn_idx)
        triples.append([head, relation, tail])

    return triples
