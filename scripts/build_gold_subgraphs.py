from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kqapro_hallucination.gold_subgraph_builder import build_gold_subgraphs_df  # noqa: E402
from kqapro_hallucination.io import ensure_columns, read_csv_with_idx, write_csv  # noqa: E402
from kqapro_hallucination.paths import (  # noqa: E402
    default_kb_json_path,
    default_oxigraph_store_path,
)
from kqapro_hallucination.schemas import QUESTION_BASE_COLUMNS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build gold_subgraphs.csv from question_base.csv.")
    parser.add_argument(
        "--base_path",
        default=str(ROOT / "data" / "processed" / "question_base.csv"),
    )
    parser.add_argument(
        "--output_path",
        default=str(ROOT / "data" / "processed" / "gold_subgraphs.csv"),
    )
    parser.add_argument(
        "--checkpoint_path",
        default=str(ROOT / "data" / "processed" / "gold_subgraphs.checkpoint.json"),
    )
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--oxigraph_store_path", default=str(default_oxigraph_store_path()))
    parser.add_argument("--kb_json_path", default=str(default_kb_json_path()))
    parser.add_argument("--name_embed_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--name_embed_threshold", type=float, default=0.78)
    parser.add_argument("--name_embed_margin", type=float, default=0.03)
    parser.add_argument("--name_span_max_tokens", type=int, default=8)
    args = parser.parse_args()

    base_df = read_csv_with_idx(args.base_path)
    ensure_columns(base_df, QUESTION_BASE_COLUMNS, "question_base")
    gold_df = build_gold_subgraphs_df(
        base_df=base_df,
        kb_json_path=args.kb_json_path,
        oxigraph_store_path=args.oxigraph_store_path,
        checkpoint_path=args.checkpoint_path,
        save_every=args.save_every,
        name_embed_model=args.name_embed_model,
        name_embed_threshold=args.name_embed_threshold,
        name_embed_margin=args.name_embed_margin,
        name_span_max_tokens=args.name_span_max_tokens,
    )
    write_csv(gold_df, args.output_path)
    print(f"saved: {args.output_path}")
    print(f"rows: {len(gold_df)}")


if __name__ == "__main__":
    main()
