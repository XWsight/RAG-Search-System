import unittest

from rag_system.calibration import ConfidenceSample, calibrate_threshold


class ThresholdCalibrationTests(unittest.TestCase):
    def test_selects_separator_and_reports_confusion_matrix(self) -> None:
        samples = [
            ConfidenceSample("positive-high", 0.9, True),
            ConfidenceSample("positive-mid", 0.7, True),
            ConfidenceSample("negative-mid", 0.4, False),
            ConfidenceSample("negative-low", 0.1, False),
        ]
        report = calibrate_threshold(samples)
        self.assertGreater(report.threshold, 0.4)
        self.assertLessEqual(report.threshold, 0.7)
        self.assertEqual(report.false_positives, 0)
        self.assertEqual(report.false_negatives, 0)
        self.assertEqual(report.f1, 1.0)
        self.assertIn("RAG_LOCAL_CONFIDENCE", report.to_markdown())

    def test_false_positive_cost_prefers_conservative_threshold(self) -> None:
        samples = [
            ConfidenceSample("answerable", 0.5, True),
            ConfidenceSample("unknown", 0.5, False),
        ]
        conservative = calibrate_threshold(
            samples,
            false_positive_cost=5.0,
            false_negative_cost=1.0,
        )
        self.assertGreater(conservative.threshold, 0.5)
        self.assertEqual(conservative.false_positives, 0)

    def test_boundaries_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConfidenceSample("", 0.5, True)
        with self.assertRaises(ValueError):
            ConfidenceSample("x", 1.1, True)
        with self.assertRaises(ValueError):
            calibrate_threshold([])
        with self.assertRaises(ValueError):
            calibrate_threshold([ConfidenceSample("x", 0.5, True)], false_positive_cost=-1)
        with self.assertRaises(ValueError):
            calibrate_threshold(
                [ConfidenceSample("x", 0.5, True), ConfidenceSample("x", 0.4, False)]
            )


if __name__ == "__main__":
    unittest.main()
