QUESTION_BASE_COLUMNS = [
    "idx",
    "question",
    "typ",
    "choices",
    "answer",
    "q_ent",
    "que_ent_ids",
    "a_ent",
    "ans_ent_ids",
    "sparql",
]

GOLD_SUBGRAPH_COLUMNS = [
    "idx",
    "question",
    "typ",
    "gold_subgraph_edges",
    "gold_heads",
    "gold_tails",
    "gold_entities",
    "gold_question_nodes",
    "gold_question_mentions",
]

ONEHOP_COLUMNS = [
    "idx",
    "question",
    "typ",
    "onehop_by_seed",
]

OLD_TO_NEW_COLUMNS = {
    "Unnamed: 0": "idx",
    "gold_subgraph_edges_sparql": "gold_subgraph_edges",
    "gold_heads_sparql": "gold_heads",
    "gold_tails_sparql": "gold_tails",
    "gold_entities_sparql": "gold_entities",
    "subgraph_1hop_by_seed": "onehop_by_seed",
}
