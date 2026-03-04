from typing import Iterable


LLAMA_PREFIX = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
)
LLAMA_SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def build_llama_prompt(user_text: str) -> str:
    return LLAMA_PREFIX + user_text + LLAMA_SUFFIX


def build_entity_prompt(question: str, shots: Iterable[tuple[str, list[str]]]) -> str:
    parts = [
        "Please extract the entities mentioned in the question.",
        "Return ONLY a Python list of entity names and nothing else.",
        "",
    ]
    for shot_question, shot_answer in shots:
        parts.append(f"Question: {shot_question}")
        parts.append(f"Answer: {shot_answer}")
        parts.append("")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return build_llama_prompt("\n".join(parts))


def build_head_prompt(
    question: str,
    entities: list[str],
    shots: Iterable[tuple[str, list[str], list[str]]],
) -> str:
    parts = [
        "Given the question and the candidate entity list, choose the head entities (one or more) that can start reasoning.",
        "Return ONLY a Python list of entity names and nothing else.",
        "",
    ]
    for shot_question, shot_entities, shot_heads in shots:
        parts.append(f"Question: {shot_question}")
        parts.append(f"Entities: {shot_entities}")
        parts.append(f"Answer: {shot_heads}")
        parts.append("")
    parts.append(f"Question: {question}")
    parts.append(f"Entities: {entities}")
    parts.append("Answer:")
    return build_llama_prompt("\n".join(parts))


def format_candidate_edges(edges: list[list]) -> str:
    return "\n".join(str(edge) for edge in edges)


def build_reason_prompt(
    question: str,
    seed: str,
    edges_text: str,
    shots: Iterable[dict[str, str]],
) -> str:
    parts = [
        "You will be given a question, a start node, and candidate 1-hop edges.",
        "Select the edges that support reasoning for the question.",
        "Return ONLY a Python list of the selected triples, e.g., [['entity1', 'relation', 'entity2']].",
        "Do NOT answer the example questions. Only answer the LAST question.",
        "",
    ]
    for shot in shots:
        parts.append(f"Question: {shot['question']}")
        parts.append(f"Start Node: {shot['seed']}")
        parts.append("Candidate Edges:")
        parts.append(shot["edges"])
        parts.append(f"Answer: {shot['answer']}")
        parts.append("")
    parts.append(f"Question: {question}")
    parts.append(f"Start Node: {seed}")
    parts.append("Candidate Edges:")
    parts.append(edges_text)
    parts.append("Answer:")
    return build_llama_prompt("\n".join(parts))
