from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kqapro_hallucination.io import (  # noqa: E402
    ensure_columns,
    filter_and_order_columns,
    read_csv_with_idx,
    write_csv,
)
from kqapro_hallucination.paths import default_kqa_v2_path  # noqa: E402
from kqapro_hallucination.schemas import QUESTION_BASE_COLUMNS  # noqa: E402


def build_question_base(input_path: str | Path):
    df = read_csv_with_idx(input_path)
    ensure_columns(df, QUESTION_BASE_COLUMNS, "question_base input")
    return filter_and_order_columns(df, QUESTION_BASE_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description="Export the semantic question_base.csv table.")
    parser.add_argument("--input_path", default=str(default_kqa_v2_path()))
    parser.add_argument(
        "--output_path",
        default=str(ROOT / "data" / "processed" / "question_base.csv"),
    )
    args = parser.parse_args()

    question_base = build_question_base(args.input_path)
    write_csv(question_base, args.output_path)
    print(f"saved: {args.output_path}")
    print(f"rows: {len(question_base)}")


if __name__ == "__main__":
    main()
