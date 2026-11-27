import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evidence import CONDITIONS, KEY, check_keys, independent_metrics, save_split


class EvidenceTests(unittest.TestCase):
    def test_binary_fpr_uses_only_actual_negatives(self):
        truth = np.array([0, 0, 0, 1])
        predicted = np.array([0, 1, 1, 1])
        binary = independent_metrics(truth, predicted, [0, 1], True)
        multiclass = independent_metrics(truth, predicted, [0, 1], False)
        self.assertAlmostEqual(binary["FPR"], 2 / 3)
        self.assertAlmostEqual(multiclass["FPR"], 1 / 3)

    def test_macro_scores_include_absent_classes(self):
        truth = np.array(["benign", "unseen"])
        predicted = np.array(["benign", "benign"])
        scores = independent_metrics(truth, predicted, ["benign", "unseen", "absent"], False)
        self.assertAlmostEqual(scores["Accuracy"], 1 / 2)
        self.assertAlmostEqual(scores["Macro F1"], 2 / 9)
        self.assertAlmostEqual(scores["Weighted F1"], 1 / 3)

    def test_invalid_decoded_class_is_rejected(self):
        with self.assertRaises(ValueError):
            independent_metrics(np.array([0]), np.array([2]), [0, 1], True)

    def test_split_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            save_split(Path(directory), 42, "Reference", np.array([0, 1]), np.array([1, 2]),
                       np.array([0, 1, 0]))

    def test_condition_grid_requires_every_key(self):
        grid = pd.MultiIndex.from_product([
            [42], ["binary", "multiclass"], ["Logistic Regression", "Random Forest", "XGBoost"],
            CONDITIONS,
        ], names=KEY).to_frame(index=False)
        check_keys(grid, [42])
        with self.assertRaises(ValueError):
            check_keys(pd.concat([grid.iloc[1:], grid.iloc[[1]]]), [42])


if __name__ == "__main__":
    unittest.main()
