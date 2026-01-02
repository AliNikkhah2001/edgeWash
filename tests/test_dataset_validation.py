import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import dataset_validation as dv


class TestDatasetValidation(unittest.TestCase):
    def test_raw_datasets_ready(self):
        report = dv.validate_all_datasets(
            validate_raw=True,
            validate_processed=False,
            max_samples=2,
            max_rows=5,
        )
        if report.errors:
            self.fail("Raw dataset validation failed:\n" + "\n".join(report.errors))

    def test_processed_datasets_ready(self):
        report = dv.validate_all_datasets(
            validate_raw=False,
            validate_processed=True,
            max_samples=2,
            max_rows=10,
        )
        if report.errors:
            self.fail("Processed dataset validation failed:\n" + "\n".join(report.errors))


if __name__ == "__main__":
    unittest.main()
