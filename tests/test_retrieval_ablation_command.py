import argparse
import unittest

from rag_system.retrieval import FusionWeights
from scripts.ablate_retrieval import DEFAULT_PROFILES, _fusion_weight, build_parser


class RetrievalAblationCommandTests(unittest.TestCase):
    def test_parser_has_safe_reproducible_defaults(self) -> None:
        arguments = build_parser().parse_args(["suite.json"])
        self.assertEqual(tuple(arguments.profiles), DEFAULT_PROFILES)
        self.assertEqual(arguments.baseline, "fusion-diverse")
        self.assertEqual(arguments.repetitions, 3)
        self.assertIsNone(arguments.dotenv)

    def test_weight_spec_requires_a_named_normalized_quadruple(self) -> None:
        name, weights = _fusion_weight("sparse-10:0.45:0.10:0.25:0.20")
        self.assertEqual(name, "sparse-10")
        self.assertEqual(weights, FusionWeights(0.45, 0.10, 0.25, 0.20))
        for value in ("Bad:0.4:0.1:0.3:0.2", "missing", "bad:1:1:1:1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                _fusion_weight(value)


if __name__ == "__main__":
    unittest.main()
