import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kqapro_hallucination.question_node_extractor import extract_question_nodes


class QuestionNodeExtractorTest(unittest.TestCase):
    def test_entity_and_qualifier_question(self):
        question = (
            "Who was the prize winner when Mrs. Miniver got the Academy Award for "
            "Best Writing, Adapted Screenplay?"
        )
        sparql = (
            'SELECT DISTINCT ?qpv WHERE { ?e_1 <pred:name> "Mrs. Miniver" . '
            '?e_2 <pred:name> "Academy Award for Best Writing, Adapted Screenplay" . '
            '?e_1 <award_received> ?e_2 . '
            '[ <pred:fact_h> ?e_1 ; <pred:fact_r> <award_received> ; <pred:fact_t> ?e_2 ] '
            '<statement_is_subject_of> ?qpv . }'
        )
        self.assertEqual(
            extract_question_nodes(sparql, question),
            ["Mrs. Miniver", "Academy Award for Best Writing, Adapted Screenplay"],
        )

    def test_count_filter_numbers(self):
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
        self.assertEqual(
            extract_question_nodes(sparql, question),
            ["county of Pennsylvania", "7800", "40000000"],
        )

    def test_title_value_question(self):
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

    def test_unit_aware_literals(self):
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
        self.assertEqual(
            extract_question_nodes(sparql, question),
            ["silent film", "67 minute", "770 minute"],
        )

    def test_date_literal(self):
        question = "Scream 4 was released on 2011-05-05 in what location?"
        sparql = (
            'SELECT DISTINCT ?qpv WHERE { ?e <pred:name> "Scream 4" . '
            '?e <publication_date> ?pv . ?pv <pred:date> "2011-05-05"^^xsd:date . '
            '[ <pred:fact_h> ?e ; <pred:fact_r> <publication_date> ; <pred:fact_t> ?pv ] '
            '<place_of_publication> ?qpv . }'
        )
        self.assertEqual(
            extract_question_nodes(sparql, question),
            ["Scream 4", "2011-05-05"],
        )

    def test_url_literal(self):
        question = "Which video game has the official website http://moh.ea.com/?"
        sparql = (
            'SELECT DISTINCT ?e WHERE { '
            '?e <pred:instance_of> ?c . ?c <pred:name> "video game" . '
            '?e <official_website> ?pv . ?pv <pred:value> "http://moh.ea.com/" . }'
        )
        self.assertEqual(
            extract_question_nodes(sparql, question),
            ["video game", "http://moh.ea.com/"],
        )

    def test_pluralized_unit_in_question(self):
        question = (
            "How many news programs either received a 60th Primetime Emmy Award "
            "or have a duration longer than 3.9 years?"
        )
        sparql = (
            'SELECT (COUNT(DISTINCT ?e) AS ?count) WHERE { '
            '?e <pred:instance_of> ?c . ?c <pred:name> "news" . '
            '{ ?e <award_received> ?e_1 . ?e_1 <pred:name> "60th Primetime Emmy Awards" . } '
            'UNION '
            '{ ?e <duration> ?pv . ?pv <pred:unit> "year" . ?pv <pred:value> ?v . '
            'FILTER ( ?v > "3.9"^^xsd:double ) . } }'
        )
        self.assertEqual(
            extract_question_nodes(sparql, question),
            ["news", "60th Primetime Emmy Awards", "3.9 year"],
        )


if __name__ == "__main__":
    unittest.main()
