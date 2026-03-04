import json
from itertools import chain
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .kb_loader import DataForSPARQL
from .literal_utils import serialize_literal
from .question_node_extractor import extract_question_nodes
from .sparql_engine import build_or_load_engine, extract_subgraph


def build_uri_to_name(kg: DataForSPARQL) -> dict[str, str]:
    return {eid: kg.get_name(eid) for eid in chain(kg.entities, kg.concepts)}


def derive_heads_tails_entities(triples: list) -> tuple[list, list, list]:
    nodes, indeg, outdeg = set(), {}, {}
    for s, _, o in triples:
        nodes.add(s)
        nodes.add(o)
        indeg[o] = indeg.get(o, 0) + 1
        indeg.setdefault(s, indeg.get(s, 0))
        outdeg[s] = outdeg.get(s, 0) + 1
        outdeg.setdefault(o, outdeg.get(o, 0))

    heads = [node for node in nodes if indeg.get(node, 0) == 0]
    tails = [node for node in nodes if outdeg.get(node, 0) == 0]

    for s, _, o in triples:
        if s == o:
            if s not in heads:
                heads.append(s)
            if s not in tails:
                tails.append(s)

    heads = [head for head in heads if not isinstance(head, (list, tuple))]

    entities = []
    seen = set()
    for s, _, o in triples:
        if s not in seen:
            entities.append(s)
            seen.add(s)
        if o not in seen:
            entities.append(o)
            seen.add(o)
    return heads, tails, entities


def _load_checkpoint(checkpoint_path: Path) -> dict | None:
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open() as f:
        return json.load(f)


def _save_checkpoint(
    checkpoint_path: Path,
    done: int,
    subgraphs: list[str],
    heads_list: list[str],
    tails_list: list[str],
    entities_list: list[str],
    question_nodes_list: list[str],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w") as f:
        json.dump(
            {
                "done": done,
                "subgraphs": subgraphs,
                "heads": heads_list,
                "tails": tails_list,
                "entities": entities_list,
                "question_nodes": question_nodes_list,
            },
            f,
        )


def build_gold_subgraphs_df(
    base_df: pd.DataFrame,
    kb_json_path: str | Path,
    oxigraph_store_path: str | Path,
    checkpoint_path: str | Path,
    save_every: int = 200,
) -> pd.DataFrame:
    kg = DataForSPARQL(str(kb_json_path))
    engine = build_or_load_engine(kg, store_path=str(oxigraph_store_path))
    uri_to_name = build_uri_to_name(kg)

    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    total = len(base_df)
    if checkpoint and "question_nodes" in checkpoint:
        start = checkpoint["done"]
        subgraphs = checkpoint["subgraphs"]
        heads_list = checkpoint["heads"]
        tails_list = checkpoint["tails"]
        entities_list = checkpoint["entities"]
        question_nodes_list = checkpoint["question_nodes"]
    else:
        start = 0
        subgraphs, heads_list, tails_list, entities_list, question_nodes_list = [], [], [], [], []

    for i in tqdm(range(start, total), desc="Extracting subgraphs", initial=start, total=total):
        sparql = base_df.iloc[i].get("sparql", "")
        question = base_df.iloc[i].get("question", "")
        if not isinstance(sparql, str) or not sparql.strip():
            triples = []
        else:
            triples = extract_subgraph(sparql, engine, uri_to_name, include_id=True)

        heads, tails, entities = derive_heads_tails_entities(triples)
        question_nodes = extract_question_nodes(sparql, question)
        subgraphs.append(serialize_literal(triples))
        heads_list.append(serialize_literal(heads))
        tails_list.append(serialize_literal(tails))
        entities_list.append(serialize_literal(entities))
        question_nodes_list.append(serialize_literal(question_nodes))

        if (i + 1) % save_every == 0:
            _save_checkpoint(
                checkpoint_path,
                i + 1,
                subgraphs,
                heads_list,
                tails_list,
                entities_list,
                question_nodes_list,
            )

    _save_checkpoint(
        checkpoint_path,
        total,
        subgraphs,
        heads_list,
        tails_list,
        entities_list,
        question_nodes_list,
    )

    gold_df = base_df.loc[:, ["idx", "question", "typ"]].copy()
    gold_df["gold_subgraph_edges"] = subgraphs
    gold_df["gold_heads"] = heads_list
    gold_df["gold_tails"] = tails_list
    gold_df["gold_entities"] = entities_list
    gold_df["gold_question_nodes"] = question_nodes_list

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return gold_df
