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
    extract_first_list,
    f1_from_entity_anchors,
    filter_by_excluded_types,
    limit_df,
    load_processed_table,
    load_vllm_llm,
    make_output_path,
    parse_list_field,
    require_shot_data_dir,
    take_first_n_valid,
    warn_insufficient_shots,
    write_prompt_sidecar,
)
from kqapro_hallucination.io import write_csv  # noqa: E402
from kqapro_hallucination.prompt_builders import build_entity_prompt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--shot_data_dir", default=None)
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
        ["idx", "question", "typ"],
    )
    gold_df = load_processed_table(
        args.data_dir,
        "gold_subgraphs.csv",
        ["idx", "gold_question_nodes", "gold_question_mentions"],
    )
    question_df = question_df.merge(gold_df, on="idx", how="inner", validate="one_to_one")
    question_df = filter_by_excluded_types(question_df, args.exclude_types)
    question_df = limit_df(question_df, args.limit)

    require_shot_data_dir(args.few_shot, args.shot_data_dir)
    shots = []
    if args.few_shot > 0:
        shot_df = load_processed_table(
            args.shot_data_dir,
            "question_base.csv",
            ["idx", "question", "typ"],
        )
        shot_gold_df = load_processed_table(
            args.shot_data_dir,
            "gold_subgraphs.csv",
            ["idx", "gold_question_nodes", "gold_question_mentions"],
        )
        shot_df = shot_df.merge(shot_gold_df, on="idx", how="inner", validate="one_to_one")
        shot_df = filter_by_excluded_types(shot_df, args.exclude_types)

        def is_valid(row):
            entities = clean_entities(parse_list_field(row.get("gold_question_mentions", "[]")))
            return bool(entities)

        shot_rows = take_first_n_valid(
            (row for _, row in shot_df.iterrows()),
            is_valid,
            args.few_shot,
        )
        for row in shot_rows:
            shots.append(
                (
                    row["question"],
                    clean_entities(parse_list_field(row.get("gold_question_mentions", "[]"))),
                )
            )
        warn_insufficient_shots("entity", args.few_shot, len(shots))

    llm = load_vllm_llm(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        gpu_mem_util=args.gpu_mem_util,
        max_num_seqs=128,
    )

    out_rows = []
    prompt_records = []
    for _, row in tqdm(question_df.iterrows(), total=len(question_df), desc="Entity Eval"):
        question = row["question"]
        gold_nodes = clean_entities(parse_list_field(row.get("gold_question_nodes", "[]")))
        gold_mentions = clean_entities(parse_list_field(row.get("gold_question_mentions", "[]")))
        prompt_text = build_entity_prompt(question, shots)
        res = llm.invoke(prompt_text)
        res_text = res if isinstance(res, str) else getattr(res, "content", str(res))
        pred_entities = clean_entities(extract_first_list(res_text))

        precision, recall, f1 = f1_from_entity_anchors(
            pred_entities,
            gold_nodes,
            gold_mentions,
        )

        out_rows.append(
            {
                "idx": row["idx"],
                "question": question,
                "gold_entities": str(gold_mentions),
                "pred_entities": str(pred_entities),
                "entity_precision": precision,
                "entity_recall": recall,
                "entity_f1": f1,
                "response_raw": res_text,
            }
        )

        if args.debug_prompts:
            prompt_records.append(
                {
                    "idx": int(row["idx"]),
                    "question": question,
                    "mode": "entity",
                    "prompt": prompt_text,
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_path = make_output_path(args.output_dir, args.output_path, "graph_entity")
    write_csv(out_df, out_path)
    if args.debug_prompts:
        write_prompt_sidecar(prompt_records, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
