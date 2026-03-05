import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kqapro_hallucination.eval_common import f1_from_entity_anchors


class EntityAnchorF1Test(unittest.TestCase):
    def test_surface_form_counts_as_hit(self):
        precision, recall, f1 = f1_from_entity_anchors(
            pred_entities=["108th US Congress"],
            gold_nodes=["108th United States Congress"],
            gold_mentions=["108th US Congress"],
        )
        self.assertEqual((precision, recall, f1), (1.0, 1.0, 1.0))

    def test_canonical_form_counts_as_hit(self):
        precision, recall, f1 = f1_from_entity_anchors(
            pred_entities=["108th United States Congress"],
            gold_nodes=["108th United States Congress"],
            gold_mentions=["108th US Congress"],
        )
        self.assertEqual((precision, recall, f1), (1.0, 1.0, 1.0))

    def test_duplicate_predictions_do_not_double_count(self):
        precision, recall, f1 = f1_from_entity_anchors(
            pred_entities=[
                "108th US Congress",
                "108th United States Congress",
                "legislative session",
            ],
            gold_nodes=["108th United States Congress", "legislative term"],
            gold_mentions=["108th US Congress", "legislative session"],
        )
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 0.8)


if __name__ == "__main__":
    unittest.main()
