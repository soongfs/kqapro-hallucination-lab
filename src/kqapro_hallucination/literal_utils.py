import ast
import re
from typing import Any


_QID_PATTERN = re.compile(r"\(([QP]\d+)\)\s*$")


def safe_literal_eval(value: Any, default: Any):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return default
    if value is None:
        return default
    return value


def serialize_literal(value: Any) -> str:
    return repr(value)


def extract_qid(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    match = _QID_PATTERN.search(text)
    return match.group(1) if match else None


def strip_qid_suffix(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return _QID_PATTERN.sub("", text).rstrip()


def contains_unbound_var(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("?")
    if isinstance(value, dict):
        return any(
            contains_unbound_var(k) or contains_unbound_var(v)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(contains_unbound_var(v) for v in value)
    return False
