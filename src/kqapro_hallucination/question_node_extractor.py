import re


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


def normalize_text_anchor(text: str) -> str:
    normalized = str(text).translate(_QUOTE_MAP)
    normalized = normalized.strip().strip('"').strip("'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower()


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


def _anchor_in_question(candidate: str, question: str) -> bool:
    question_norm = normalize_text_anchor(question)
    for variant in _candidate_variants(candidate):
        if variant and variant in question_norm:
            return True
    return False


def extract_question_nodes(sparql: str, question: str) -> list[str]:
    if not isinstance(sparql, str) or not sparql.strip():
        return []

    body = _extract_where_body(sparql)
    if not body:
        return []

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

    candidate_events: list[tuple[int, str]] = []

    for triple in triples:
        pred = triple["pred"]
        if pred == "pred:name":
            literal = _parse_literal_token(triple["obj"])
            if literal is not None:
                candidate_events.append((triple["pos"], literal))
        elif pred in {"pred:value", "pred:year", "pred:date"}:
            literal = _parse_literal_token(triple["obj"])
            if literal is None:
                continue
            unit = unit_by_subject.get(triple["subj"]) if pred == "pred:value" else None
            formatted = _format_value(literal, unit)
            if formatted is not None:
                candidate_events.append((triple["pos"], formatted))

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
            candidate_events.append((match.start(), formatted))

    ordered = sorted(candidate_events, key=lambda item: item[0])
    nodes = []
    seen = set()
    for _, candidate in ordered:
        if candidate in seen:
            continue
        if not _anchor_in_question(candidate, question):
            continue
        seen.add(candidate)
        nodes.append(candidate)
    return nodes
