import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .literal_utils import extract_qid, safe_literal_eval, serialize_literal


def _load_kb_maps(kb_info_path: str | Path) -> tuple[dict, dict]:
    with open(kb_info_path, "rb") as f:
        kb_info = pickle.load(f)
    return kb_info.get("ent_to_id_map", {}), kb_info.get("id_to_ent_map", {})


def _extract_id(text):
    return extract_qid(text)


def _map_to_name(value, id_to_name):
    if isinstance(value, str) and value in id_to_name:
        return f"{id_to_name[value][0]} ({value})"
    return value


def _ids_for_name(name, name_to_id):
    if not isinstance(name, str):
        return []
    qid = _extract_id(name)
    if qid:
        return [qid]
    return name_to_id.get(name, [])


def _load_trip_and_fact_maps(
    trips_path: str | Path,
    facts_path: str | Path,
) -> tuple[dict, dict]:
    trips = np.load(trips_path, allow_pickle=True).tolist()
    facts = np.load(facts_path, allow_pickle=True).tolist()

    trips_by_head = defaultdict(list)
    for h, r, t in trips:
        trips_by_head[h].append((h, r, t))

    facts_by_head = defaultdict(list)
    for (_, _, t), qk, qv in facts:
        facts_by_head[t].append((qk, qv))

    return trips_by_head, facts_by_head


def build_onehop_by_seed_df(
    gold_df: pd.DataFrame,
    kb_info_path: str | Path,
    trips_path: str | Path,
    facts_path: str | Path,
    topk: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    name_to_id, id_to_name = _load_kb_maps(kb_info_path)
    trips_by_head, facts_by_head = _load_trip_and_fact_maps(trips_path, facts_path)
    rng = random.Random(random_seed)

    all_rows = []
    for i in range(len(gold_df)):
        gold_edges = safe_literal_eval(
            gold_df.iloc[i].get("gold_subgraph_edges", "[]"),
            [],
        )

        outdeg = defaultdict(int)
        for edge in gold_edges:
            if isinstance(edge, (list, tuple)) and len(edge) == 3:
                s, _, o = edge
                outdeg[s] += 1
                outdeg.setdefault(o, outdeg.get(o, 0))

        seen_seed = set()
        seed_list = []
        for edge in gold_edges:
            if isinstance(edge, (list, tuple)) and len(edge) == 3:
                s, _, o = edge
                if outdeg.get(s, 0) > 0 and s not in seen_seed:
                    seen_seed.add(s)
                    seed_list.append(s)
                if outdeg.get(o, 0) > 0 and o not in seen_seed:
                    seen_seed.add(o)
                    seed_list.append(o)

        gold_by_seed = defaultdict(list)
        for edge in gold_edges:
            if isinstance(edge, (list, tuple)) and len(edge) == 3:
                s, _, _ = edge
                gold_by_seed[s].append(edge)

        per_seed = []
        for seed in seed_list:
            gold_out = gold_by_seed.get(seed, [])

            cand = []
            kb_ids = _ids_for_name(seed, name_to_id) or [seed]
            for head_id in kb_ids:
                for h, r, t in trips_by_head.get(head_id, []):
                    cand.append([_map_to_name(h, id_to_name), r, _map_to_name(t, id_to_name)])
                for qk, qv in facts_by_head.get(head_id, []):
                    cand.append([_map_to_name(head_id, id_to_name), qk, _map_to_name(qv, id_to_name)])

            uniq = []
            seen = set()
            for edge in cand:
                key = str(edge)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(edge)

            gold_keys = {str(edge) for edge in gold_out}
            uniq = [edge for edge in uniq if str(edge) not in gold_keys]
            remaining = topk - len(gold_out)
            if remaining < 0:
                topk_edges = gold_out[:topk]
            else:
                if len(uniq) > remaining:
                    pick = rng.sample(uniq, remaining)
                else:
                    pick = uniq
                topk_edges = gold_out + pick

            per_seed.append(
                {
                    "seed": seed,
                    "gold_edges": gold_out,
                    "topk_edges": topk_edges,
                }
            )

        all_rows.append(serialize_literal(per_seed))

    onehop_df = gold_df.loc[:, ["idx", "question", "typ"]].copy()
    onehop_df["onehop_by_seed"] = all_rows
    return onehop_df
