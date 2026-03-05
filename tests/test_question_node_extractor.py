import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kqapro_hallucination.question_node_extractor import (
    NameMentionAligner,
    extract_question_anchors,
    extract_question_nodes,
    normalize_text_anchor,
)


class FakeEmbedder:
    def __init__(self):
        self.vectors = {
            normalize_text_anchor("108th United States Congress"): np.array([1.0, 0.0, 0.0]),
            normalize_text_anchor("108th US Congress"): np.array([1.0, 0.0, 0.0]),
            normalize_text_anchor("legislative term"): np.array([0.0, 1.0, 0.0]),
            normalize_text_anchor("legislative session"): np.array([0.0, 1.0, 0.0]),
            normalize_text_anchor("United States Congress"): np.array([1.0, 0.0, 0.0]),
            normalize_text_anchor("Congress calendar"): np.array([0.75, 0.66, 0.0]),
        }

    def encode(self, texts, **kwargs):
        return np.vstack(
            [self.vectors.get(normalize_text_anchor(text), np.zeros(3)) for text in texts]
        )


class QuestionNodeExtractorTest(unittest.TestCase):
    def setUp(self):
        self.aligner = NameMentionAligner(
            similarity_threshold=0.78,
            similarity_margin=0.03,
            max_span_tokens=8,
            embedder=FakeEmbedder(),
        )

    def test_extract_question_nodes_wrapper_returns_nodes(self):
        question = "How many musical compositions are named Yellow Submarine?"
        sparql = (
            'SELECT (COUNT(DISTINCT ?e) AS ?count) WHERE { '
            '?e <pred:instance_of> ?c . ?c <pred:name> "musical composition" . '
            '?e <title> ?pv . ?pv <pred:value> "Yellow Submarine" . }'
        )
        self.assertEqual(
            extract_question_nodes(sparql, question),
            ["musical composition", "Yellow Submarine"],
        )

    def test_count_filter_numbers_and_mentions(self):
        question = (
            "How many Pennsylvania counties have a population greater than 7800 "
            "or a population less than 40000000?"
        )
        sparql = (
            'SELECT (COUNT(DISTINCT ?e) AS ?count) WHERE { '
            '?e <pred:instance_of> ?c . ?c <pred:name> "county of Pennsylvania" . '
            '{ ?e <population> ?pv . ?pv <pred:unit> "1" . ?pv <pred:value> ?v . '
            'FILTER ( ?v > "7800"^^xsd:double ) . } '
            'UNION '
            '{ ?e <population> ?pv . ?pv <pred:unit> "1" . ?pv <pred:value> ?v . '
            'FILTER ( ?v < "40000000"^^xsd:double ) . } }'
        )
        nodes, mentions = extract_question_anchors(sparql, question)
        self.assertEqual(nodes, ["county of Pennsylvania", "7800", "40000000"])
        self.assertEqual(mentions, ["Pennsylvania counties", "7800", "40000000"])

    def test_title_value_question_mentions(self):
        question = "How many musical compositions are named Yellow Submarine?"
        sparql = (
            'SELECT (COUNT(DISTINCT ?e) AS ?count) WHERE { '
            '?e <pred:instance_of> ?c . ?c <pred:name> "musical composition" . '
            '?e <title> ?pv . ?pv <pred:value> "Yellow Submarine" . }'
        )
        nodes, mentions = extract_question_anchors(sparql, question)
        self.assertEqual(nodes, ["musical composition", "Yellow Submarine"])
        self.assertEqual(mentions, ["musical compositions", "Yellow Submarine"])

    def test_unit_aware_literals_keep_rule_matching(self):
        question = (
            "Which silent film's duration is equal to 67 minute and less than 770 minute ?"
        )
        sparql = (
            'SELECT DISTINCT ?e WHERE { '
            '?e <pred:instance_of> ?c . ?c <pred:name> "silent film" . '
            '?e <duration> ?pv . ?pv <pred:unit> "minute" . '
            '?pv <pred:value> "67"^^xsd:double . '
            '?e <duration> ?pv_1 . ?pv_1 <pred:unit> "minute" . '
            '?pv_1 <pred:value> ?v . FILTER ( ?v < "770"^^xsd:double ) . }'
        )
        nodes, mentions = extract_question_anchors(sparql, question)
        self.assertEqual(nodes, ["silent film", "67 minute", "770 minute"])
        self.assertEqual(mentions, ["silent film's", "67 minute", "770 minute"])

    def test_date_literal(self):
        question = "Scream 4 was released on 2011-05-05 in what location?"
        sparql = (
            'SELECT DISTINCT ?qpv WHERE { ?e <pred:name> "Scream 4" . '
            '?e <publication_date> ?pv . ?pv <pred:date> "2011-05-05"^^xsd:date . '
            '[ <pred:fact_h> ?e ; <pred:fact_r> <publication_date> ; <pred:fact_t> ?pv ] '
            '<place_of_publication> ?qpv . }'
        )
        nodes, mentions = extract_question_anchors(sparql, question)
        self.assertEqual(nodes, ["Scream 4", "2011-05-05"])
        self.assertEqual(mentions, ["Scream 4", "2011-05-05"])

    def test_url_literal(self):
        question = "Which video game has the official website http://moh.ea.com/?"
        sparql = (
            'SELECT DISTINCT ?e WHERE { '
            '?e <pred:instance_of> ?c . ?c <pred:name> "video game" . '
            '?e <official_website> ?pv . ?pv <pred:value> "http://moh.ea.com/" . }'
        )
        nodes, mentions = extract_question_anchors(sparql, question)
        self.assertEqual(nodes, ["video game", "http://moh.ea.com/"])
        self.assertEqual(mentions, ["video game", "http://moh.ea.com/"])

    def test_embedding_alignment_for_name_constants(self):
        question = "When was the opening of the legislative session preceding the 108th US Congress?"
        sparql = (
            'SELECT DISTINCT ?pv WHERE { ?e <pred:instance_of> ?c . '
            '?c <pred:name> "legislative term" . ?e_1 <follows> ?e . '
            '?e_1 <pred:name> "108th United States Congress" . ?e <start_time> ?pv . }'
        )
        nodes, mentions = extract_question_anchors(sparql, question, name_aligner=self.aligner)
        self.assertEqual(nodes, ["legislative term", "108th United States Congress"])
        self.assertEqual(mentions, ["legislative session", "108th US Congress"])

    def test_name_aligner_rejects_weak_match(self):
        question = "What is the Congress calendar?"
        mention = self.aligner.align("United States Congress", question)
        self.assertIsNone(mention)


if __name__ == "__main__":
    unittest.main()
