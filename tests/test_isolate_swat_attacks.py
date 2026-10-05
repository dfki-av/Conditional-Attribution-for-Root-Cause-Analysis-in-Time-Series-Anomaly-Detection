"""Check the SWaT workbook to CSV preprocessing boundary."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from preprocessing.isolate_swat_attacks import ATTACK_INTERVALS, AttackInterval, isolate_swat_attacks


class IsolateSwatAttacksTests(unittest.TestCase):
    def test_intervals_have_ordered_dates_and_expected_attack_ids(self):
        expected_ids = {
            1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41,
        }
        self.assertEqual({int(item.name.split("_", 1)[0]) for item in ATTACK_INTERVALS}, expected_ids)
        for item in ATTACK_INTERVALS:
            self.assertLessEqual(pd.Timestamp(item.start), pd.Timestamp(item.end), item.name)

    def test_normal_export_excludes_attack_intervals_and_incomplete_sensor_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "SWaT_Dataset_Attack_v0.xlsx"
            normal_output = root / "processed/normal.csv"
            attacks_dir = root / "processed/swat_attacks"
            frame = pd.DataFrame({
                " Timestamp": [
                    "28/12/2015 10:00:00 AM", "28/12/2015 10:01:00 AM",
                    "28/12/2015 10:02:00 AM", "28/12/2015 10:03:00 AM",
                    "28/12/2015 10:04:00 AM", "28/12/2015 10:05:00 AM",
                    "28/12/2015 10:06:00 AM", "28/12/2015 10:07:00 AM",
                ],
                "MV101": [1, 2, 3, 4, 5, 3, 7, 8],
                "P102": [10, 20, 30, 40, 50, 30, 70, None],
                "Index": list(range(8)),
                "Normal/Attack": [
                    "Normal", "Attack", " Normal ", "Attack",
                    "Normal", "Attack", None, "Normal",
                ],
            })
            frame.to_excel(workbook, sheet_name="Combined Data", startrow=1, index=False)
            intervals = (
                AttackInterval("1_MV-101", "2015-12-28 10:01:00", "2015-12-28 10:02:00"),
                AttackInterval("2_P-102", "2015-12-28 10:03:00", "2015-12-28 10:03:00"),
            )
            isolate_swat_attacks(workbook, normal_output, attacks_dir, intervals)

            normal = pd.read_csv(normal_output)
            first_attack = pd.read_csv(attacks_dir / "1_MV-101.csv")
            second_attack = pd.read_csv(attacks_dir / "2_P-102.csv")
            self.assertEqual(normal.columns.tolist(), ["MV101", "P102"])
            self.assertEqual(normal["MV101"].tolist(), [1, 5, 3, 7])
            self.assertEqual(normal["P102"].tolist(), [10, 50, 30, 70])
            self.assertEqual(first_attack["MV101"].tolist(), [2, 3])
            self.assertEqual(second_attack["MV101"].tolist(), [4])
            self.assertEqual(sorted(path.suffix for path in attacks_dir.iterdir()), [".csv", ".csv"])


if __name__ == "__main__":
    unittest.main()
