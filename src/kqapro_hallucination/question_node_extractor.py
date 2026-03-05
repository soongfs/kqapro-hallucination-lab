import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


_TRIPLE_RE = re.compile(
    r"(?P<subj>\?[A-Za-z_]\w*|<[^>]+>)\s+"
    r"<(?P<pred>pred:(?:name|value|unit|year|date))>\s+"
    r"(?P<obj>\?[A-Za-z_]\w*|\"(?:[^\"\\]|\\.)*\"(?:\^\^[^\s.]+)?|-?\d+(?:\.\d+)?)\s*\.",
    re.DOTALL,
)

_FILTER_RE = re.compile(
    r"FILTER\s*\(\s*(?P<var>\?[A-Za-z_]\w*)\s*"
    r"(?:!=|<=|>=|=|<|>)\s*"
    r"(?P<const>\"(?:[^\"\\]|\\.)*\"(?:\^\^[^\s)]+)?|-?\d+(?:\.\d+)?)\s*\)",
    re.DOTALL,
)

_HYPHEN_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        **{char: "-" for char in _HYPHEN_CHARS},
    }
)
_WORD_RE = re.compile(r"([A-Za-z][A-Za-z'-]*)")
_OF_PATTERN = re.compile(r"^(?P<head>.+?)\s+of\s+(?P<tail>.+)$", re.IGNORECASE)
_TOKEN_RE = re.compile(
    r"https?://[^\s]+|"
    r"\d{4}-\d{2}-\d{2}|"
    r"\d+(?:\.\d+)?(?:st|nd|rd|th)\b|"
    r"[A-Za-z](?:\.[A-Za-z])+\.?|"
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
)
_NUMERIC_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?(?:st|nd|rd|th)?")
_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
}


class EmbedderProtocol(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class CandidateSpan:
    surface: str
    normalized: str
    token_count: int


def normalize_text_anchor(text: str) -> str:
    normalized = str(text).translate(_QUOTE_MAP)
    normalized = normalized.strip().strip('"').strip("'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower()


def _normalize_embeddings(embeddings: Any) -> np.ndarray:
    array = np.asarray(embeddings, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return array / norms


def _extract_where_body(sparql: str) -> str:
    text = sparql.strip()
    depth = 0
    start = None
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:idx]
    return ""


def _parse_literal_token(token: str) -> str | None:
    raw = token.strip()
    if raw.startswith("?"):
        return None
    if "^^" in raw:
        raw = raw.split("^^", 1)[0]
    if raw.startswith('"') and raw.endswith('"'):
        inner = raw[1:-1]
        inner = inner.replace(r"\"", '"').replace(r"\\", "\\")
        return inner
    return raw


def _format_value(value: str | None, unit: str | None) -> str | None:
    if not value:
        return None
    if unit and unit != "1":
        return f"{value} {unit}"
    return value


def _pluralize_word(word: str) -> str:
    lower = word.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if lower.endswith("y") and len(word) > 1 and word[-2].lower() not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _singularize_word(word: str) -> str:
    lower = word.lower()
    if lower.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if lower.endswith("es") and lower[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return word[:-2]
    if lower.endswith("s") and not lower.endswith("ss"):
        return word[:-1]
    return word


def _replace_last_word(text: str, transform) -> str | None:
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    start, end = match.span()
    return text[:start] + transform(match.group(0)) + text[end:]


def _candidate_variants(text: str) -> set[str]:
    variants = {normalize_text_anchor(text)}

    plural_variant = _replace_last_word(text, _pluralize_word)
    if plural_variant:
        variants.add(normalize_text_anchor(plural_variant))

    singular_variant = _replace_last_word(text, _singularize_word)
    if singular_variant:
        variants.add(normalize_text_anchor(singular_variant))

    of_match = _OF_PATTERN.match(text)
    if of_match:
        reordered = f"{of_match.group('tail')} {of_match.group('head')}"
        variants.add(normalize_text_anchor(reordered))
        reordered_plural = _replace_last_word(reordered, _pluralize_word)
        if reordered_plural:
            variants.add(normalize_text_anchor(reordered_plural))
        reordered_singular = _replace_last_word(reordered, _singularize_word)
        if reordered_singular:
            variants.add(normalize_text_anchor(reordered_singular))

    return {variant for variant in variants if variant}


def _enumerate_question_spans(question: str, max_span_tokens: int) -> list[CandidateSpan]:
    matches = list(_TOKEN_RE.finditer(question))
    spans: list[CandidateSpan] = []
    seen = set()
    for start_idx in range(len(matches)):
        for end_idx in range(start_idx, min(len(matches), start_idx + max_span_tokens)):
            start = matches[start_idx].start()
            end = matches[end_idx].end()
            surface = question[start:end].strip().rstrip("?!,;:")
            normalized = normalize_text_anchor(surface)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            spans.append(
                CandidateSpan(
                    surface=surface,
                    normalized=normalized,
                    token_count=end_idx - start_idx + 1,
                )
            )
    return spans


def _pick_best_span(spans: list[CandidateSpan]) -> str | None:
    if not spans:
        return None
    best = min(spans, key=lambda span: (span.token_count, len(span.normalized), len(span.surface)))
    return best.surface


def _find_surface_mention(candidate: str, spans: list[CandidateSpan]) -> str | None:
    variants = _candidate_variants(candidate)
    exact_matches = [span for span in spans if span.normalized in variants]
    if exact_matches:
        return _pick_best_span(exact_matches)

    partial_matches = []
    for span in spans:
        for variant in variants:
            if not variant:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(variant)}(?:'s)?(?![a-z0-9])"
            if re.search(pattern, span.normalized):
                partial_matches.append(span)
                break
    return _pick_best_span(partial_matches)


def _extract_numeric_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _NUMERIC_RE.finditer(normalize_text_anchor(text))}


def _content_tokens(text: str) -> set[str]:
    normalized = normalize_text_anchor(text)
    tokens = {token for token in _CONTENT_TOKEN_RE.findall(normalized) if token not in _STOPWORDS}
    return tokens


def _is_valid_name_span(canonical_name: str, span: CandidateSpan) -> bool:
    if not span.normalized or not re.search(r"[a-z0-9]", span.normalized):
        return False

    canonical_numbers = _extract_numeric_tokens(canonical_name)
    if canonical_numbers and not canonical_numbers.issubset(_extract_numeric_tokens(span.surface)):
        return False

    span_tokens = _content_tokens(span.surface)
    if not span_tokens:
        return False

    canonical_tokens = _content_tokens(canonical_name)
    if canonical_tokens and not (canonical_tokens & span_tokens):
        return False
    return True


class NameMentionAligner:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.78,
        similarity_margin: float = 0.03,
        max_span_tokens: int = 8,
        embedder: EmbedderProtocol | None = None,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.similarity_margin = similarity_margin
        self.max_span_tokens = max_span_tokens
        self._embedder = embedder

    def _get_embedder(self) -> EmbedderProtocol:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for pred:name mention alignment"
                ) from exc
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def align(self, canonical_name: str, question: str) -> str | None:
        return self.align_many([canonical_name], question).get(canonical_name)

    def align_many(self, canonical_names: list[str], question: str) -> dict[str, str]:
        spans = _enumerate_question_spans(question, self.max_span_tokens)
        if not canonical_names or not spans:
            return {}

        matches: dict[str, str] = {}
        unresolved: list[str] = []
        for canonical_name in canonical_names:
            surface = _find_surface_mention(canonical_name, spans)
            if surface:
                matches[canonical_name] = surface
            else:
                unresolved.append(canonical_name)

        if not unresolved:
            return matches

        embedder = self._get_embedder()
        span_embeddings = _normalize_embeddings(embedder.encode([span.surface for span in spans]))

        for canonical_name in unresolved:
            valid_indices = [
                index
                for index, span in enumerate(spans)
                if _is_valid_name_span(canonical_name, span)
            ]
            if not valid_indices:
                continue

            query_embedding = _normalize_embeddings(embedder.encode([canonical_name]))[0]
            scores = span_embeddings[valid_indices] @ query_embedding
            best_local = int(np.argmax(scores))
            best_score = float(scores[best_local])
            second_score = None
            if len(scores) > 1:
                sorted_scores = np.sort(scores)
                second_score = float(sorted_scores[-2])

            if best_score < self.similarity_threshold:
                continue
            if second_score is not None and (best_score - second_score) < self.similarity_margin:
                continue

            matches[canonical_name] = spans[valid_indices[best_local]].surface

        return matches


def extract_question_anchors(
    sparql: str,
    question: str,
    name_aligner: NameMentionAligner | None = None,
) -> tuple[list[str], list[str]]:
    if not isinstance(sparql, str) or not sparql.strip():
        return [], []

    body = _extract_where_body(sparql)
    if not body:
        return [], []

    triples = []
    for match in _TRIPLE_RE.finditer(body):
        triples.append(
            {
                "pos": match.start(),
                "subj": match.group("subj"),
                "pred": match.group("pred"),
                "obj": match.group("obj"),
            }
        )

    unit_by_subject: dict[str, str] = {}
    value_var_bindings: dict[str, tuple[str, str]] = {}

    for triple in triples:
        pred = triple["pred"]
        obj_value = _parse_literal_token(triple["obj"])
        if pred == "pred:unit" and obj_value is not None:
            unit_by_subject[triple["subj"]] = obj_value
        elif pred in {"pred:value", "pred:year", "pred:date"} and triple["obj"].startswith("?"):
            value_var_bindings[triple["obj"]] = (pred, triple["subj"])

    candidate_events: list[tuple[int, str, str]] = []

    for triple in triples:
        pred = triple["pred"]
        if pred == "pred:name":
            literal = _parse_literal_token(triple["obj"])
            if literal is not None:
                candidate_events.append((triple["pos"], "name", literal))
        elif pred in {"pred:value", "pred:year", "pred:date"}:
            literal = _parse_literal_token(triple["obj"])
            if literal is None:
                continue
            unit = unit_by_subject.get(triple["subj"]) if pred == "pred:value" else None
            formatted = _format_value(literal, unit)
            if formatted is not None:
                candidate_events.append((triple["pos"], "literal", formatted))

    for match in _FILTER_RE.finditer(body):
        var_name = match.group("var")
        binding = value_var_bindings.get(var_name)
        if not binding:
            continue
        pred, subj = binding
        literal = _parse_literal_token(match.group("const"))
        if literal is None:
            continue
        unit = unit_by_subject.get(subj) if pred == "pred:value" else None
        formatted = _format_value(literal, unit)
        if formatted is not None:
            candidate_events.append((match.start(), "literal", formatted))

    ordered = sorted(candidate_events, key=lambda item: item[0])
    spans = _enumerate_question_spans(question, max_span_tokens=8 if name_aligner is None else name_aligner.max_span_tokens)

    ordered_name_candidates = []
    seen_name_candidates = set()
    for _, candidate_type, canonical in ordered:
        if candidate_type != "name" or canonical in seen_name_candidates:
            continue
        seen_name_candidates.add(canonical)
        ordered_name_candidates.append(canonical)

    if name_aligner is not None:
        aligned_name_mentions = name_aligner.align_many(ordered_name_candidates, question)
    else:
        aligned_name_mentions = {
            canonical: surface
            for canonical in ordered_name_candidates
            if (surface := _find_surface_mention(canonical, spans))
        }

    nodes = []
    mentions = []
    seen_nodes = set()
    for _, candidate_type, canonical in ordered:
        if canonical in seen_nodes:
            continue
        if candidate_type == "name":
            surface = aligned_name_mentions.get(canonical)
        else:
            surface = _find_surface_mention(canonical, spans)
        if not surface:
            continue
        seen_nodes.add(canonical)
        nodes.append(canonical)
        mentions.append(surface)
    return nodes, mentions


def extract_question_nodes(sparql: str, question: str) -> list[str]:
    nodes, _ = extract_question_anchors(sparql, question)
    return nodes
