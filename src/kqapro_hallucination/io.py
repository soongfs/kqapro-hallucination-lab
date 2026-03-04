from pathlib import Path

import pandas as pd

from .schemas import OLD_TO_NEW_COLUMNS


def _normalize_idx_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns=OLD_TO_NEW_COLUMNS)
    if "idx" not in out.columns:
        first_col = str(out.columns[0]) if len(out.columns) else ""
        if first_col.startswith("Unnamed:"):
            out = out.rename(columns={out.columns[0]: "idx"})
    if "idx" not in out.columns:
        raise ValueError("Could not find an idx column in input dataframe.")
    return out


def read_csv_with_idx(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_idx_column(df)
    return df.sort_values("idx").reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", errors="replace")


def ensure_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def filter_and_order_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = [col for col in columns if col in df.columns]
    return df.loc[:, keep].copy()


def subset_by_idx(df: pd.DataFrame, kept_idx: list[int]) -> pd.DataFrame:
    order = {idx: pos for pos, idx in enumerate(kept_idx)}
    out = df[df["idx"].isin(kept_idx)].copy()
    out["_order"] = out["idx"].map(order)
    out = out.sort_values("_order").drop(columns="_order")
    return out.reset_index(drop=True)
