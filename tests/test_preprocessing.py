import ast
import inspect
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import torch

from data.readers import load_csv_directory, read_csv
from experiment.config import PROJECT_ROOT, load_config, resolve_project_path
from experiment.pipeline import PROGRESS_BAR_FORMAT, _create_directories, _get_feature_names, _validate_inputs, compute_shats_attributions_for_attacks, prepare_datasets, resolve_experiment_paths, run_experiment, train_configured_model
from preprocessing.scaling import split_windows
from preprocessing.windows import make_windows


class PreprocessingRegressionTests(unittest.TestCase):
    def test_csv_readers_return_filename_stems_without_passing_flag_to_pandas(self):
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "26_P-101_LIT-301.csv"
            pd.DataFrame({"sensor": [1, 2]}).to_csv(csv_path, index=False)
            frame, stem = read_csv(csv_path, return_filenames=True)
            frames, filenames = load_csv_directory(directory, return_filenames=True)
        self.assertEqual(frame["sensor"].tolist(), [1, 2])
        self.assertEqual(stem, "26_P-101_LIT-301")
        self.assertEqual(frames["26"]["sensor"].tolist(), [1, 2])
        self.assertEqual(filenames["26"], stem)

    def test_windows_follow_the_configured_stride(self):
        frame = pd.DataFrame({"sensor": np.arange(7)})
        actual = make_windows(frame, window_size=3, stride=2)
        expected = np.array([[[0], [1], [2]], [[2], [3], [4]], [[4], [5], [6]]])
        np.testing.assert_array_equal(actual, expected)

    def test_short_series_repeats_its_last_row(self):
        frame = pd.DataFrame({"sensor": [1, 2]})
        actual = make_windows(frame, window_size=3, stride=3)
        np.testing.assert_array_equal(actual, np.array([[[1], [2], [2]]]))

    def test_split_is_repeatable_with_the_experiment_split_seed(self):
        values = np.arange(30).reshape(10, 3)
        first = split_windows(values, valid_perc=0.1, shuffle=True, seed=123)
        second = split_windows(values, valid_perc=0.1, shuffle=True, seed=123)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_attack_windows_use_training_scaler_and_keep_window_shape(self):
        config = load_config()
        config["dataset"].update(window_size=2, stride=2, validation_fraction=0.25)
        normal = pd.DataFrame({"a": range(8), "b": range(10, 18)})
        attack = pd.DataFrame({"a": [8, 9, 10, 11], "b": [18, 19, 20, 21]})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            normal_csv = root / "normal.csv"
            anomaly_dir = root / "attacks"
            anomaly_dir.mkdir()
            normal.to_csv(normal_csv, index=False)
            attack.to_csv(anomaly_dir / "1_attack.csv", index=False)
            datasets = prepare_datasets(
                config, {"normal_csv": normal_csv, "anomaly_csv_dir": anomaly_dir}
            )
            with patch(
                "experiment.pipeline.load_csv_directory",
                side_effect=lambda folder: {"1": pd.read_csv(Path(folder) / "1_attack.csv")},
            ):
                legacy_datasets = prepare_datasets(
                    config, {"normal_csv": normal_csv, "anomaly_csv_dir": anomaly_dir}
                )
            config["dataset"]["normalization_method"] = "standard"
            standard_datasets = prepare_datasets(
                config, {"normal_csv": normal_csv, "anomaly_csv_dir": anomaly_dir}
            )

        normal_windows = make_windows(normal, window_size=2, stride=2)
        train_windows, _ = split_windows(
            normal_windows, valid_perc=0.25, seed=config["random_seeds"]["split"]
        )
        train_rows = train_windows.reshape(-1, 2)
        mini = np.min(train_windows, axis=0)
        data_range = np.max(train_windows, axis=0) - mini
        attack_windows = attack.to_numpy().reshape(2, 2, 2)
        expected = (attack_windows - mini) / (data_range + 1e-7)
        self.assertEqual(datasets["test"].shape, (2, 2, 2))
        self.assertEqual(datasets["attacks"]["1"].shape, (2, 2, 2))
        self.assertEqual(datasets["attack_filenames"]["1"], "1_attack")
        self.assertEqual(legacy_datasets["attack_filenames"]["1"], "1_attack")
        self.assertEqual(datasets["feature_names"], ["a", "b"])
        np.testing.assert_allclose(datasets["test"], expected, rtol=1e-6)
        np.testing.assert_allclose(datasets["attacks"]["1"], expected, rtol=1e-6)
        from sklearn.preprocessing import StandardScaler
        standard = StandardScaler().fit(train_rows)
        np.testing.assert_allclose(
            standard_datasets["test"], standard.transform(attack.to_numpy()).reshape(2, 2, 2), rtol=1e-6
        )

    def test_config_preserves_experiment_values_and_resolves_data_paths(self):
        config = load_config()
        self.assertEqual(config["dataset"]["window_size"], 50)
        self.assertEqual(config["dataset"]["stride"], 50)
        self.assertEqual(config["dataset"]["validation_fraction"], 0.1)
        self.assertEqual(config["dataset"]["normalization_method"], "min_max")
        self.assertTrue(config["dataset"]["mask_variable_names"])
        self.assertEqual(config["model"]["name"], "TemporalVAE")
        self.assertEqual(config["model"]["max_epochs"], 300)
        self.assertEqual(config["model"]["parameters"], {
            "latent_dim": 8,
            "hidden_layer_sizes": [50, 100, 200],
            "reconstruction_wt": 3.0,
            "batch_size": 512,
            "use_residual_conn": True,
            "trend_poly": 0,
            "custom_seas": None,
        })
        self.assertEqual(config["experiment"]["knn_mode"], "vae")
        self.assertEqual(config["evaluation"]["attack_sensors_dataset"], "swat")
        self.assertEqual(config["outputs"]["checkpoint_subdirectory"], "checkpoints")
        self.assertTrue(config["training"]["enabled"])
        self.assertFalse(config["runtime"]["print_inference_time"])
        self.assertIsNone(config["runtime"]["faiss_gpu_id"])
        self.assertEqual(config["experiment"]["support_size"], 3)
        self.assertEqual(config["experiment"]["umap"], {
            "n_neighbors": 30, "n_components": 10, "min_dist": 0.1,
        })
        self.assertEqual(config["random_seeds"]["global"], 42)
        self.assertEqual(config["random_seeds"]["split"], 123)
        self.assertEqual(resolve_project_path("data/processed/input.csv"), PROJECT_ROOT / "data/processed/input.csv")
        self.assertEqual(resolve_project_path("/tmp/input.csv"), Path("/tmp/input.csv"))
        resolved = resolve_experiment_paths(config)
        self.assertEqual(resolved["normal_csv"], PROJECT_ROOT / config["paths"]["normal_csv"])
        self.assertEqual(resolved["normal_csv"].parent, PROJECT_ROOT / "data/processed")
        self.assertEqual(resolved["normal_csv"], PROJECT_ROOT / "data/processed/SWAT_Non_Attack_Instances.csv")
        self.assertEqual(resolved["anomaly_csv_dir"], PROJECT_ROOT / "data/processed/SWAT_Attacks")
        self.assertNotIn("run_subdirectory", config["outputs"])
        override = resolve_experiment_paths(config, {"normal_csv": "/tmp/normal.csv"})
        self.assertEqual(override["normal_csv"], Path("/tmp/normal.csv"))

    def test_feature_names_follow_mask_flag(self):
        config = load_config()
        datasets = {"feature_names": ["MV-101", "P-102"], "training": np.zeros((1, 2, 2))}
        self.assertEqual(_get_feature_names(config, datasets), ["MV-101", "P-102"])
        config["dataset"]["mask_variable_names"] = False
        self.assertEqual(_get_feature_names(config, datasets), ["Feature1", "Feature2"])

    def test_explainability_progress_advances_for_each_test_window(self):
        config = load_config()
        config["experiment"]["knn_mode"] = "none"
        config["evaluation"]["attack_sensors_dataset"] = "wadi"
        datasets = {
            "training": np.zeros((1, 2, 2), dtype=np.float32),
            "validation": np.zeros((1, 2, 2), dtype=np.float32),
            "test": np.zeros((3, 2, 2), dtype=np.float32),
            "attacks": {
                "1": np.zeros((2, 2, 2), dtype=np.float32),
                "26": np.zeros((1, 2, 2), dtype=np.float32),
            },
            "attack_filenames": {"1": "1_MV-101", "26": "26_P-101_LIT-301"},
            "feature_names": ["MV-101", "P-101"],
        }
        with TemporaryDirectory() as directory, patch(
            "experiment.pipeline.VAEWrapperTorch"
        ), patch("experiment.pipeline._support_indices", return_value=[0]), patch(
            "experiment.pipeline.FastShaTS"
        ) as estimator, patch("experiment.pipeline.tqdm") as progress_bar, patch(
            "experiment.pipeline._print_top_k_metrics"
        ) as print_top_k:
            estimator.return_value.compute.return_value = torch.ones((1, 2, 2))
            bars = [MagicMock(), MagicMock(), MagicMock()]
            progress_bar.side_effect = bars
            compute_shats_attributions_for_attacks(
                config, {"checkpoint_dir": Path(directory)}, datasets,
                model=object(), threshold=0.5, device=torch.device("cpu"),
                output_dir=Path(directory),
            )
        progress_stream = sys.__stderr__ or sys.stderr
        progress_options = {
            "unit": "window", "dynamic_ncols": True, "ascii": False,
            "bar_format": PROGRESS_BAR_FORMAT, "file": progress_stream,
            "mininterval": 0.2,
        }
        self.assertEqual(progress_bar.call_args_list, [
            call(total=3, desc="Overall explainability", position=0,
                 colour="cyan" if progress_stream.isatty() else None, **progress_options),
            call(total=2, desc="Sample 1/2 (1_MV-101)", position=1, leave=False,
                 colour="green" if progress_stream.isatty() else None, **progress_options),
            call(total=1, desc="Sample 2/2 (26_P-101_LIT-301)", position=1, leave=False,
                 colour="green" if progress_stream.isatty() else None, **progress_options),
        ])
        self.assertEqual([bar.update.call_count for bar in bars], [3, 2, 1])
        for bar in bars:
            bar.close.assert_called_once()
        self.assertEqual(print_top_k.call_args.args[2], "wadi")

    def test_orchestration_keeps_preprocess_evaluate_explain_order(self):
        tree = ast.parse(inspect.getsource(run_experiment))

        class CallCollector(ast.NodeVisitor):
            def __init__(self):
                self.calls = []

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    self.calls.append(node.func.id)
                self.generic_visit(node)

        collector = CallCollector()
        collector.visit(tree)
        self.assertLess(collector.calls.index("prepare_datasets"), collector.calls.index("evaluate_model"))
        self.assertLess(collector.calls.index("evaluate_model"), collector.calls.index("compute_shats_attributions_for_attacks"))

    def test_training_does_not_require_an_existing_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            normal_csv = root / "normal.csv"
            normal_csv.touch()
            anomaly_dir = root / "attacks"
            anomaly_dir.mkdir()
            paths = {
                "normal_csv": normal_csv,
                "anomaly_csv_dir": anomaly_dir,
                "checkpoint_dir": root / "missing_checkpoint",
            }
            _validate_inputs(paths, training_enabled=True)
            with self.assertRaises(FileNotFoundError):
                _validate_inputs(paths, training_enabled=False)
            paths["checkpoint_dir"] = None
            with self.assertRaisesRegex(ValueError, "checkpoint_dir"):
                _validate_inputs(paths, training_enabled=False)

    def test_configured_directories_are_created_without_inventing_input_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "normal_csv": root / "data/processed/swat/normal.csv",
                "anomaly_csv_dir": root / "data/processed/swat/attacks",
                "output_dir": root / "outputs",
                "checkpoint_dir": root / "outputs/run/checkpoints",
            }
            _create_directories(paths)
            for key in ("anomaly_csv_dir", "output_dir", "checkpoint_dir"):
                self.assertTrue(paths[key].is_dir())
            self.assertTrue(paths["normal_csv"].parent.is_dir())
            with self.assertRaises(FileNotFoundError):
                _validate_inputs(paths, training_enabled=True)

    def test_training_uses_configured_model_and_saves_checkpoint(self):
        config = load_config()
        training_data = np.zeros((2, 4, 3), dtype=np.float32)
        validation_data = np.zeros((1, 4, 3), dtype=np.float32)
        checkpoint_dir = Path("/tmp/example_checkpoint")
        with patch("experiment.pipeline.TemporalVAE") as model_class, patch(
            "experiment.pipeline.train_variational_autoencoder"
        ) as train:
            model = train_configured_model(
                config,
                {"checkpoint_dir": checkpoint_dir},
                {"training": training_data, "validation": validation_data},
                torch.device("cpu"),
            )
        model_class.assert_called_once_with(
            seq_len=4, feat_dim=3, **config["model"]["parameters"]
        )
        train.assert_called_once_with(
            model, training_data, validation_data,
            epochs=config["model"]["max_epochs"], verbose=1,
            device=torch.device("cpu"),
        )
        model.save.assert_called_once_with(str(checkpoint_dir))

    def test_run_saves_dated_results_file(self):
        config = load_config()
        with TemporaryDirectory() as directory, patch(
            "experiment.pipeline._validate_inputs"
        ), patch("experiment.pipeline.prepare_datasets", return_value={}), patch(
            "experiment.pipeline.train_configured_model", return_value="trained_model"
        ) as train, patch(
            "experiment.pipeline.evaluate_model", return_value=("model", 0.5)
        ), patch("experiment.pipeline.compute_shats_attributions_for_attacks"):
            run_experiment(config, {"output_dir": directory})
            results_files = list(Path(directory).glob("*/results_*.txt"))
            self.assertEqual(len(results_files), 1)
            self.assertRegex(results_files[0].name, r"results_\d{8}_\d{6}_\d{6}\.txt")
            results_text = results_files[0].read_text(encoding="utf-8")
            self.assertIn("Experiment configuration:\n", results_text)
            self.assertIn("mask_variable_names: true", results_text)
            self.assertIn("Run overrides:\noutput_dir:", results_text)
            self.assertIn("Seed set to 42", results_text)
            self.assertEqual(
                train.call_args.args[1]["checkpoint_dir"], results_files[0].parent / "checkpoints"
            )


if __name__ == "__main__":
    unittest.main()
