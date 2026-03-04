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
    extract_triples_from_response,
    f1_pr,
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
from kqapro_hallucination.prompt_builders import (  # noqa: E402
    build_reason_prompt,
    format_candidate_edges,
)


def expand_reason_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        per_seed = parse_list_field(row.get("onehop_by_seed", "[]"))
        for item in per_seed:
            rows.append(
                {
                    "idx": row["idx"],
                    "question": row["question"],
                    "seed": item.get("seed"),
                    "gold_edges": item.get("gold_edges", []),
                    "topk_edges": item.get("topk_edges", []),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--shot_data_dir", default=None)
    parser.add_argument("--output_dir", default=str(ROOT / "output"))
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
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

    onehop_df = load_processed_table(
        args.data_dir,
        "onehop_by_seed.csv",
        ["idx", "question", "typ", "onehop_by_seed"],
    )
    onehop_df = filter_by_excluded_types(onehop_df, args.exclude_types)
    onehop_df = limit_df(onehop_df, args.limit)
    eval_rows = expand_reason_rows(onehop_df)

    require_shot_data_dir(args.few_shot, args.shot_data_dir)
    shots = []
    if args.few_shot > 0:
        shot_df = load_processed_table(
            args.shot_data_dir,
            "onehop_by_seed.csv",
            ["idx", "question", "typ", "onehop_by_seed"],
        )
        shot_df = filter_by_excluded_types(shot_df, args.exclude_types)
        shot_rows = take_first_n_valid(
            expand_reason_rows(shot_df),
            lambda item: bool(item["topk_edges"] and item["gold_edges"]),
            args.few_shot,
        )
        for item in shot_rows:
            shots.append(
                {
                    "question": item["question"],
                    "seed": str(item["seed"]),
                    "edges": format_candidate_edges(item["topk_edges"]),
                    "answer": str(item["gold_edges"]),
                }
            )
        warn_insufficient_shots("reason", args.few_shot, len(shots))

    llm = load_vllm_llm(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        gpu_mem_util=args.gpu_mem_util,
        max_num_seqs=64,
    )

    out_rows = []
    prompt_records = []
    for item in tqdm(eval_rows, total=len(eval_rows), desc="Reason Eval"):
        question = item["question"]
        seed = str(item["seed"])
        gold_edges = item["gold_edges"]
        topk_edges = item["topk_edges"]

        if not topk_edges:
            out_rows.append(
                {
                    "idx": item["idx"],
                    "question": question,
                    "seed": seed,
                    "gold_edges": str(gold_edges),
                    "llm_selected_edges": "[]",
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "response_raw": "",
                }
            )
            continue

        prompt_text = build_reason_prompt(
            question=question,
            seed=seed,
            edges_text=format_candidate_edges(topk_edges),
            shots=shots,
        )
        res = llm.invoke(prompt_text)
        res_text = res if isinstance(res, str) else getattr(res, "content", str(res))
        selected = extract_triples_from_response(res_text)
        precision, recall, f1 = f1_pr(selected, gold_edges)

        out_rows.append(
            {
                "idx": item["idx"],
                "question": question,
                "seed": seed,
                "gold_edges": str(gold_edges),
                "llm_selected_edges": str(selected),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "response_raw": res_text,
            }
        )

        if args.debug_prompts:
            prompt_records.append(
                {
                    "idx": int(item["idx"]),
                    "question": question,
                    "mode": "reason",
                    "seed": seed,
                    "prompt": prompt_text,
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_path = make_output_path(args.output_dir, args.output_path, "graph_reason")
    write_csv(out_df, out_path)
    if args.debug_prompts:
        write_prompt_sidecar(prompt_records, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
