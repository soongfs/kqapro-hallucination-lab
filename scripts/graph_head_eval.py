from pathlib import Path
import argparse
import sys

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kqapro_hallucination.eval_common import (  # noqa: E402
    clean_entities,
    ensure_entity_result_columns,
    extract_first_list,
    f1_from_sets,
    filter_by_excluded_types,
    limit_df,
    load_processed_table,
    load_vllm_llm,
    make_output_path,
    merge_on_idx,
    parse_list_field,
    parse_output_list,
    require_shot_data_dir,
    take_first_n_valid,
    to_name_only_set,
    warn_insufficient_shots,
    write_prompt_sidecar,
)
from kqapro_hallucination.io import read_csv_with_idx, write_csv  # noqa: E402
from kqapro_hallucination.literal_utils import strip_qid_suffix  # noqa: E402
from kqapro_hallucination.prompt_builders import build_head_prompt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--shot_data_dir", default=None)
    parser.add_argument("--entity_result_path", required=True)
    parser.add_argument("--output_dir", default=str(ROOT / "output"))
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--debug_prompts", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--exclude_types",
        type=str,
        default="",
        help="Comma-separated typ tags to exclude, e.g. count,verify",
    )
    args = parser.parse_args()

    question_df = load_processed_table(
        args.data_dir,
        "question_base.csv",
        ["idx", "question", "typ", "q_ent"],
    )
    gold_df = load_processed_table(
        args.data_dir,
        "gold_subgraphs.csv",
        ["idx", "gold_heads"],
    )
    entity_df = read_csv_with_idx(args.entity_result_path)
    ensure_entity_result_columns(entity_df)

    merged = merge_on_idx(question_df, gold_df, how="inner")
    merged = merge_on_idx(
        merged,
        entity_df.loc[:, ["idx", "pred_entities"]],
        how="left",
    )
    if merged["pred_entities"].isna().any():
        missing = merged.loc[merged["pred_entities"].isna(), "idx"].tolist()[:10]
        raise ValueError(
            f"Entity result is missing pred_entities for some idx values, e.g. {missing}"
        )

    merged = filter_by_excluded_types(merged, args.exclude_types)
    merged = limit_df(merged, args.limit)

    require_shot_data_dir(args.few_shot, args.shot_data_dir)
    shots = []
    if args.few_shot > 0:
        shot_question_df = load_processed_table(
            args.shot_data_dir,
            "question_base.csv",
            ["idx", "question", "typ", "q_ent"],
        )
        shot_gold_df = load_processed_table(
            args.shot_data_dir,
            "gold_subgraphs.csv",
            ["idx", "gold_heads"],
        )
        shot_merged = merge_on_idx(shot_question_df, shot_gold_df, how="inner")
        shot_merged = filter_by_excluded_types(shot_merged, args.exclude_types)

        def is_valid(row):
            entities = clean_entities(parse_list_field(row.get("q_ent", "[]")))
            heads = clean_entities(parse_list_field(row.get("gold_heads", "[]")))
            return bool(entities and heads)

        shot_rows = take_first_n_valid(
            (row for _, row in shot_merged.iterrows()),
            is_valid,
            args.few_shot,
        )
        for row in shot_rows:
            heads_name_only = [
                strip_qid_suffix(head)
                for head in clean_entities(parse_list_field(row.get("gold_heads", "[]")))
            ]
            shots.append(
                (
                    row["question"],
                    clean_entities(parse_list_field(row.get("q_ent", "[]"))),
                    heads_name_only,
                )
            )
        warn_insufficient_shots("head", args.few_shot, len(shots))

    llm = load_vllm_llm(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        gpu_mem_util=args.gpu_mem_util,
        max_num_seqs=128,
    )

    out_rows = []
    prompt_records = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Head Eval"):
        question = row["question"]
        gold_heads = clean_entities(parse_list_field(row.get("gold_heads", "[]")))
        gold_head_set = to_name_only_set(gold_heads)
        gold_entities = clean_entities(parse_list_field(row.get("q_ent", "[]")))
        pred_entities = clean_entities(parse_output_list(row.get("pred_entities", "[]")))

        prompt_gold = build_head_prompt(question, gold_entities, shots)
        res_gold = llm.invoke(prompt_gold)
        text_gold = (
            res_gold if isinstance(res_gold, str) else getattr(res_gold, "content", str(res_gold))
        )
        pred_heads_gold = clean_entities(extract_first_list(text_gold))
        pred_heads_gold_set = to_name_only_set(pred_heads_gold)
        p_gold, r_gold, f1_gold = f1_from_sets(pred_heads_gold_set, gold_head_set)
        acc_gold = 1 if (pred_heads_gold_set & gold_head_set) else 0

        prompt_pred = build_head_prompt(question, pred_entities, shots)
        res_pred = llm.invoke(prompt_pred)
        text_pred = (
            res_pred if isinstance(res_pred, str) else getattr(res_pred, "content", str(res_pred))
        )
        pred_heads_pred = clean_entities(extract_first_list(text_pred))
        pred_heads_pred_set = to_name_only_set(pred_heads_pred)
        p_pred, r_pred, f1_pred = f1_from_sets(pred_heads_pred_set, gold_head_set)
        acc_pred = 1 if (pred_heads_pred_set & gold_head_set) else 0

        out_rows.append(
            {
                "idx": row["idx"],
                "question": question,
                "gold_heads": str(gold_heads),
                "pred_heads_goldlist": str(pred_heads_gold),
                "head_precision_goldlist": p_gold,
                "head_recall_goldlist": r_gold,
                "head_f1_goldlist": f1_gold,
                "head_acc_goldlist": acc_gold,
                "pred_heads_predlist": str(pred_heads_pred),
                "head_precision_predlist": p_pred,
                "head_recall_predlist": r_pred,
                "head_f1_predlist": f1_pred,
                "head_acc_predlist": acc_pred,
                "response_head_goldlist_raw": text_gold,
                "response_head_predlist_raw": text_pred,
            }
        )

        if args.debug_prompts:
            prompt_records.append(
                {
                    "idx": int(row["idx"]),
                    "question": question,
                    "mode": "head_goldlist",
                    "prompt": prompt_gold,
                }
            )
            prompt_records.append(
                {
                    "idx": int(row["idx"]),
                    "question": question,
                    "mode": "head_predlist",
                    "prompt": prompt_pred,
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_path = make_output_path(args.output_dir, args.output_path, "graph_head")
    write_csv(out_df, out_path)
    if args.debug_prompts:
        write_prompt_sidecar(prompt_records, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
