import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from langchain_community.llms import VLLM

from .io import ensure_columns, read_csv_with_idx
from .literal_utils import safe_literal_eval, strip_qid_suffix


def normalize_name(text: Any) -> str:
    return str(text).strip().strip('"').strip("'").lower()


def parse_typ_list(raw: Any) -> list[str]:
    value = safe_literal_eval(raw, [])
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def filter_by_excluded_types(df: pd.DataFrame, exclude_types: str) -> pd.DataFrame:
    if not exclude_types or "typ" not in df.columns:
        return df.copy()
    excluded = {item.strip() for item in exclude_types.split(",") if item.strip()}
    if not excluded:
        return df.copy()

    def has_excluded(raw: Any) -> bool:
        return bool(set(parse_typ_list(raw)) & excluded)

    return df[~df["typ"].apply(has_excluded)].reset_index(drop=True)


def limit_df(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit and limit > 0:
        return df.head(limit).reset_index(drop=True)
    return df.copy()


def clean_entities(items: Iterable[Any]) -> list[str]:
    cleaned = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip().strip('"').strip("'")
        if not text:
            continue
        if text.startswith("?"):
            continue
        if text in {"{", "}", "[", "]"}:
            continue
        cleaned.append(text)
    return cleaned


def to_name_set(names: Iterable[Any]) -> set[str]:
    return {normalize_name(name) for name in names if str(name).strip()}


def to_name_only_set(names: Iterable[Any]) -> set[str]:
    return {
        normalize_name(strip_qid_suffix(name))
        for name in names
        if str(name).strip()
    }


def _normalize_entity_anchor_text(text: Any) -> str:
    return normalize_name(strip_qid_suffix(text))


def f1_from_sets(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    inter = len(pred & gold)
    precision = inter / len(pred) if pred else 0.0
    recall = inter / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def f1_from_entity_anchors(
    pred_entities: list[Any],
    gold_nodes: list[Any],
    gold_mentions: list[Any],
) -> tuple[float, float, float]:
    if len(gold_nodes) != len(gold_mentions):
        raise ValueError("gold_nodes and gold_mentions must have the same length")

    if not pred_entities and not gold_nodes:
        return 1.0, 1.0, 1.0
    if not pred_entities or not gold_nodes:
        return 0.0, 0.0, 0.0

    anchors = []
    for node, mention in zip(gold_nodes, gold_mentions):
        normalized = {
            _normalize_entity_anchor_text(node),
            _normalize_entity_anchor_text(mention),
        }
        normalized.discard("")
        if normalized:
            anchors.append(normalized)

    if not anchors:
        return (1.0, 1.0, 1.0) if not pred_entities else (0.0, 0.0, 0.0)

    matched = [False] * len(anchors)
    tp = 0
    fp = 0
    for pred in pred_entities:
        normalized_pred = _normalize_entity_anchor_text(pred)
        if not normalized_pred:
            continue
        matched_idx = None
        for index, allowed in enumerate(anchors):
            if matched[index]:
                continue
            if normalized_pred in allowed:
                matched_idx = index
                break
        if matched_idx is None:
            fp += 1
            continue
        matched[matched_idx] = True
        tp += 1

    fn = len(anchors) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(anchors) if anchors else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def extract_first_list(text: Any) -> list[Any]:
    text = str(text).strip()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    for match in re.finditer(r"\[[\s\S]*?\]", text):
        chunk = match.group(0)
        try:
            value = ast.literal_eval(chunk)
            if isinstance(value, list):
                return value
        except Exception:
            pass
        try:
            fixed = re.sub(
                r"(?<=[\[,])\s*'|'\s*(?=[,\]])",
                lambda found: found.group().replace("'", '"'),
                chunk,
            )
            value = ast.literal_eval(fixed)
            if isinstance(value, list):
                return value
        except Exception:
            continue
    return []


def normalize_element(value: Any) -> str:
    return str(value).strip().strip('"').strip("'").lower()


def normalize_triple(edge: Any) -> tuple[str, str, str] | None:
    if not isinstance(edge, (list, tuple)) or len(edge) != 3:
        return None
    return tuple(normalize_element(part) for part in edge)


def _find_outer_brackets(text: str) -> list[str]:
    results = []
    i = 0
    while i < len(text):
        if text[i] == "[":
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1
            if depth == 0:
                results.append(text[i:j])
            i = j
        else:
            i += 1
    return results


def _fix_single_quote_delimiters(chunk: str) -> str:
    return re.sub(
        r"(?<=[\[,])\s*'|'\s*(?=[,\]])",
        lambda found: found.group().replace("'", '"'),
        chunk,
    )


def _validate_triples(value: Any) -> list[list[Any]] | None:
    if not value:
        return None
    if all(isinstance(item, (list, tuple)) and len(item) == 3 for item in value):
        return [list(item) for item in value]
    if len(value) == 3 and all(not isinstance(item, (list, tuple)) for item in value):
        return [list(value)]
    return None


def extract_triples_from_response(text: Any) -> list[list[Any]]:
    text = str(text).strip()
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    for chunk in _find_outer_brackets(text):
        try:
            value = ast.literal_eval(chunk)
            if isinstance(value, list):
                triples = _validate_triples(value)
                if triples is not None:
                    return triples
        except Exception:
            pass
        try:
            value = ast.literal_eval(_fix_single_quote_delimiters(chunk))
            if isinstance(value, list):
                triples = _validate_triples(value)
                if triples is not None:
                    return triples
        except Exception:
            continue
    return []


def f1_pr(pred: Iterable[Any], gold: Iterable[Any]) -> tuple[float, float, float]:
    pred_set = set()
    for edge in pred:
        normalized = normalize_triple(edge)
        if normalized is not None:
            pred_set.add(normalized)
    gold_set = set()
    for edge in gold:
        normalized = normalize_triple(edge)
        if normalized is not None:
            gold_set.add(normalized)

    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0

    inter = len(pred_set & gold_set)
    precision = inter / len(pred_set)
    recall = inter / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def load_vllm_llm(
    model_path: str,
    max_new_tokens: int,
    gpu_mem_util: float,
    max_num_seqs: int,
):
    return VLLM(
        model=model_path,
        trust_remote_code=True,
        max_new_tokens=max_new_tokens,
        top_p=0.25,
        temperature=0,
        vllm_kwargs={
            "max_model_len": 4096,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_mem_util,
        },
    )


def load_processed_table(
    data_dir: str | Path,
    filename: str,
    required_columns: list[str],
) -> pd.DataFrame:
    path = Path(data_dir) / filename
    df = read_csv_with_idx(path)
    ensure_columns(df, required_columns, str(path))
    return df


def require_shot_data_dir(few_shot: int, shot_data_dir: str | None) -> None:
    if few_shot > 0 and not shot_data_dir:
        raise ValueError("--shot_data_dir is required when --few_shot > 0")


def take_first_n_valid(
    items: Iterable[Any],
    is_valid: Callable[[Any], bool],
    n: int,
) -> list[Any]:
    if n <= 0:
        return []
    selected = []
    for item in items:
        if is_valid(item):
            selected.append(item)
            if len(selected) >= n:
                break
    return selected


def make_output_path(
    output_dir: str | Path,
    output_path: str | None,
    prefix: str,
) -> Path:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{ts}.csv"


def make_sidecar_path(csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_suffix(".prompts.jsonl")


def write_prompt_sidecar(records: list[dict[str, Any]], csv_path: str | Path) -> None:
    if not records:
        return
    sidecar_path = make_sidecar_path(csv_path)
    with sidecar_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def warn_insufficient_shots(task_name: str, requested: int, actual: int) -> None:
    if requested > 0 and actual < requested:
        print(
            f"WARNING: requested {requested} few-shot examples for {task_name}, "
            f"but only found {actual} valid examples."
        )


def parse_list_field(raw: Any, default: list[Any] | None = None) -> list[Any]:
    if default is None:
        default = []
    value = safe_literal_eval(raw, default)
    return value if isinstance(value, list) else default


def parse_output_list(raw: Any) -> list[Any]:
    value = safe_literal_eval(raw, [])
    return value if isinstance(value, list) else []


def ensure_entity_result_columns(df: pd.DataFrame) -> None:
    ensure_columns(df, ["idx", "pred_entities"], "entity result")


def merge_on_idx(left: pd.DataFrame, right: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    return left.merge(right, on="idx", how=how)
