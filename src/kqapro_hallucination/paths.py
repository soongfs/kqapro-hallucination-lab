from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_kqa_v2_path() -> Path:
    return _first_existing(
        DATA_DIR / "kqa_v2.csv",
        REPO_ROOT.parent / "data" / "kqa_v2.csv",
    )


def default_kb_info_path() -> Path:
    return _first_existing(
        DATA_DIR / "kga_kb_info.pkl",
        REPO_ROOT.parent / "data" / "kga_kb_info.pkl",
    )


def default_trips_path() -> Path:
    return _first_existing(
        DATA_DIR / "KGApro_trips.npy",
        REPO_ROOT.parent / "data" / "KGApro_trips.npy",
    )


def default_facts_path() -> Path:
    return _first_existing(
        DATA_DIR / "KGApro_facts.npy",
        REPO_ROOT.parent / "data" / "KGApro_facts.npy",
    )


def default_kb_json_path() -> Path:
    return _first_existing(
        DATA_DIR / "kb.json",
        REPO_ROOT.parent.parent / "kqa-pro" / "dataset" / "kb.json",
        REPO_ROOT.parent / ".." / "kqa-pro" / "dataset" / "kb.json",
    )


def default_oxigraph_store_path() -> Path:
    return DATA_DIR / "kqa_oxigraph"


def ensure_processed_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
