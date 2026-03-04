from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kqapro_hallucination.io import ensure_columns, read_csv_with_idx, write_csv  # noqa: E402
from kqapro_hallucination.onehop_builder import build_onehop_by_seed_df  # noqa: E402
from kqapro_hallucination.paths import (  # noqa: E402
    default_facts_path,
    default_kb_info_path,
    default_trips_path,
)


def main():
    parser = argparse.ArgumentParser(description="Build onehop_by_seed.csv from gold_subgraphs.csv.")
    parser.add_argument(
        "--gold_path",
        default=str(ROOT / "data" / "processed" / "gold_subgraphs.csv"),
    )
    parser.add_argument(
        "--output_path",
        default=str(ROOT / "data" / "processed" / "onehop_by_seed.csv"),
    )
    parser.add_argument("--kb_info_path", default=str(default_kb_info_path()))
    parser.add_argument("--trips_path", default=str(default_trips_path()))
    parser.add_argument("--facts_path", default=str(default_facts_path()))
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    gold_df = read_csv_with_idx(args.gold_path)
    ensure_columns(
        gold_df,
        ["idx", "question", "typ", "gold_subgraph_edges"],
        "gold_subgraphs",
    )
    onehop_df = build_onehop_by_seed_df(
        gold_df=gold_df,
        kb_info_path=args.kb_info_path,
        trips_path=args.trips_path,
        facts_path=args.facts_path,
        topk=args.topk,
        random_seed=args.random_seed,
    )
    write_csv(onehop_df, args.output_path)
    print(f"saved: {args.output_path}")
    print(f"rows: {len(onehop_df)}")


if __name__ == "__main__":
    main()
