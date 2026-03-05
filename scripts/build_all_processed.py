from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kqapro_hallucination.gold_subgraph_builder import build_gold_subgraphs_df  # noqa: E402
from kqapro_hallucination.io import ensure_columns, read_csv_with_idx, write_csv  # noqa: E402
from kqapro_hallucination.onehop_builder import build_onehop_by_seed_df  # noqa: E402
from kqapro_hallucination.paths import (  # noqa: E402
    PROCESSED_DIR,
    default_facts_path,
    default_kb_info_path,
    default_kb_json_path,
    default_kqa_v2_path,
    default_oxigraph_store_path,
    default_trips_path,
    ensure_processed_dirs,
)
from kqapro_hallucination.schemas import QUESTION_BASE_COLUMNS  # noqa: E402
from export_question_base import build_question_base  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build the full processed KQA-Pro pipeline.")
    parser.add_argument("--input_path", default=str(default_kqa_v2_path()))
    parser.add_argument("--kb_json_path", default=str(default_kb_json_path()))
    parser.add_argument("--oxigraph_store_path", default=str(default_oxigraph_store_path()))
    parser.add_argument(
        "--checkpoint_path",
        default=str(PROCESSED_DIR / "gold_subgraphs.checkpoint.json"),
    )
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--kb_info_path", default=str(default_kb_info_path()))
    parser.add_argument("--trips_path", default=str(default_trips_path()))
    parser.add_argument("--facts_path", default=str(default_facts_path()))
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--name_embed_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--name_embed_threshold", type=float, default=0.78)
    parser.add_argument("--name_embed_margin", type=float, default=0.03)
    parser.add_argument("--name_span_max_tokens", type=int, default=8)
    args = parser.parse_args()

    ensure_processed_dirs()

    question_base = build_question_base(args.input_path)
    ensure_columns(question_base, QUESTION_BASE_COLUMNS, "question_base input")
    write_csv(question_base, PROCESSED_DIR / "question_base.csv")

    gold_df = build_gold_subgraphs_df(
        base_df=question_base,
        kb_json_path=args.kb_json_path,
        oxigraph_store_path=args.oxigraph_store_path,
        checkpoint_path=args.checkpoint_path,
        save_every=args.save_every,
        name_embed_model=args.name_embed_model,
        name_embed_threshold=args.name_embed_threshold,
        name_embed_margin=args.name_embed_margin,
        name_span_max_tokens=args.name_span_max_tokens,
    )
    write_csv(gold_df, PROCESSED_DIR / "gold_subgraphs.csv")

    onehop_df = build_onehop_by_seed_df(
        gold_df=gold_df,
        kb_info_path=args.kb_info_path,
        trips_path=args.trips_path,
        facts_path=args.facts_path,
        topk=args.topk,
        random_seed=args.random_seed,
    )
    write_csv(onehop_df, PROCESSED_DIR / "onehop_by_seed.csv")

    print(f"saved: {PROCESSED_DIR}")
    print(f"rows: {len(question_base)}")


if __name__ == "__main__":
    main()
