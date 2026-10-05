import contextlib
import io
import unittest

import evaluation.feature_metrics as feature_metrics
from evaluation.feature_metrics import (
    ATTACKED_SENSORS_BY_DATASET,
    aggregate_early_metrics,
    compute_topk_score_for_all_keys,
    get_attack_sensors,
    print_metric_results,
)


class FeatureMetricsTests(unittest.TestCase):
    def test_dataset_selector_uses_swat_and_wadi_labels(self):
        swat = get_attack_sensors("swat")
        wadi = get_attack_sensors("wadi")
        self.assertEqual(set(ATTACKED_SENSORS_BY_DATASET), {"swat", "wadi"})
        self.assertEqual(swat["1"], ["MV101"])
        self.assertEqual(wadi["1"], ["1_MV_001_STATUS"])
        self.assertEqual(wadi["5"], [
            "2_MCV_101_CO", "2_MCV_201_CO", "2_MCV_301_CO",
            "2_MCV_401_CO", "2_MCV_501_CO", "2_MCV_601_CO",
        ])
        self.assertEqual(wadi["15"], ["2_LIT_002"])
        self.assertIs(feature_metrics.ATTACKED_SENSORS_DICT, wadi)
        get_attack_sensors("swat")
        self.assertIs(feature_metrics.ATTACKED_SENSORS_DICT, swat)
        with self.assertRaisesRegex(ValueError, "Unknown attack-sensor dataset"):
            get_attack_sensors("smd")

    def test_selected_labels_feed_recall_and_early_metrics(self):
        for dataset, sensor in (("swat", "MV101"), ("wadi", "1_MV_001_STATUS")):
            ranked = {"1": [(sensor, 100.0)]}
            windows = {"1": [[(sensor, 100.0)]]}
            results = compute_topk_score_for_all_keys(
                ranked, k=5, attack_sensors_dataset=dataset
            )
            results.update(aggregate_early_metrics(
                windows, k=5, attack_sensors_dataset=dataset
            ))
            self.assertEqual(results["Top_K5_Recall"], 1.0)
            self.assertEqual(results["early_per_key"]["1"]["first_hit_index"], 0)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_metric_results(results, k=5)
        self.assertIn("K5_CWRCS", output.getvalue())
        self.assertIn("K5_TempHM", output.getvalue())
        self.assertNotIn("K5_Recall_Conf", output.getvalue())
        self.assertNotIn("Harmonic_Mean", output.getvalue())

    def test_metric_table_keeps_full_wadi_sensor_lists(self):
        sensor = "2_MCV_101_CO"
        results = compute_topk_score_for_all_keys(
            {"5": [(sensor, 100.0)]}, k=5, attack_sensors_dataset="wadi"
        )
        results.update(aggregate_early_metrics(
            {"5": [[(sensor, 100.0)]]}, k=5, attack_sensors_dataset="wadi"
        ))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_metric_results(results, k=5)
        self.assertIn("2_MCV_601_CO", output.getvalue())
        self.assertIn("Averages:", output.getvalue())
        self.assertNotIn("…", output.getvalue())


if __name__ == "__main__":
    unittest.main()
