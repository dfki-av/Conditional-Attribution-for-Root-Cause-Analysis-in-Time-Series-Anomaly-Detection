"""Orchestrate the SWaT reconstruction and feature-attribution experiment."""
import time
import sys
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import umap
import yaml
from tqdm import tqdm

from data.readers import load_csv_directory, read_csv
from evaluation.anomaly_detection import calculate_reconstruction_roc_auc, save_roc_curve
from evaluation.feature_metrics import (
    aggregate_early_metrics,
    compute_topk_score_for_all_keys,
    get_attack_sensors,
    print_metric_results,
)
from explainability.estimators import FastShaTS
from explainability.grouping import FeaturesGroupingStrategy
from explainability.percentages import shap_percentages_clip_whole, shap_percentages_clip_window
from experiment.config import resolve_project_path
from models.time_vae import TemporalVAE
from preprocessing.scaling import scale_windows, split_windows, transform_windows
from preprocessing.windows import make_windows
from retrieval.neighbors import search_neighbors
from training.vae_training import train_variational_autoencoder
from utils.randomness import seed_everything
from models.base_vae import VAEWrapperTorch


PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed} / ETA {remaining}, {rate_fmt}]"
)


class _Tee:
    """Write experiment messages to the terminal and the run's results file."""

    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def resolve_experiment_paths(config, overrides=None):
    """Resolve dataset, checkpoint and output locations from YAML and CLI overrides."""
    overrides = overrides or {}
    configured = config["paths"]
    values = {
        "normal_csv": overrides.get("normal_csv") or configured["normal_csv"],
        "anomaly_csv_dir": overrides.get("anomaly_dir") or configured["anomaly_csv_dir"],
        "checkpoint_dir": overrides.get("checkpoint_dir") or configured.get("checkpoint_dir"),
        "output_dir": overrides.get("output_dir") or config["outputs"]["output_dir"],
    }
    return {name: resolve_project_path(value) if value is not None else None for name, value in values.items()}


def _create_directories(paths):
    """Create configured data and output directories before validating input files."""
    directories = (
        paths["normal_csv"].parent,
        paths["anomaly_csv_dir"],
        paths["output_dir"],
        paths["checkpoint_dir"],
    )
    for directory in directories:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)


def _validate_inputs(paths, training_enabled=False):
    required = {
        "normal_csv": "normal-data CSV",
        "anomaly_csv_dir": "attack CSV directory",
    }
    for key, description in required.items():
        path = paths[key]
        if not path.exists():
            raise FileNotFoundError(f"Required {description} not found: {path}")
    if not paths["normal_csv"].is_file():
        raise FileNotFoundError(f"Normal-data path is not a file: {paths['normal_csv']}")
    if not paths["anomaly_csv_dir"].is_dir():
        raise NotADirectoryError(f"Attack-data path is not a directory: {paths['anomaly_csv_dir']}")
    if not training_enabled:
        if paths["checkpoint_dir"] is None:
            raise ValueError("Set paths.checkpoint_dir or --checkpoint-dir when training.enabled is false")
        expected_checkpoint_files = ("TimeVAE_parameters.pkl", "TimeVAE_weights.pth")
        missing_checkpoint_files = [
            name for name in expected_checkpoint_files if not (paths["checkpoint_dir"] / name).is_file()
        ]
        if missing_checkpoint_files:
            raise FileNotFoundError(
                f"Checkpoint is missing required files: {', '.join(missing_checkpoint_files)} "
                f"under {paths['checkpoint_dir']}"
            )


def prepare_datasets(config, paths):
    """Read, window, split and normalize normal and attack series."""
    dataset_config = config["dataset"]
    normal_frame = read_csv(paths["normal_csv"])
    attack_frames = load_csv_directory(paths["anomaly_csv_dir"])
    if not attack_frames:
        raise FileNotFoundError(f"No readable attack CSV files found in {paths['anomaly_csv_dir']}")
    attack_filenames = {
        filename.split("_", 1)[0]: filename[:-4]
        for filename in os.listdir(paths["anomaly_csv_dir"])
        if filename.endswith(".csv") and filename.split("_", 1)[0] in attack_frames
    }

    normal_windows = make_windows(
        normal_frame,
        window_size=dataset_config["window_size"],
        stride=dataset_config["stride"],
    )
    attack_windows = {
        key: make_windows(
            frame,
            window_size=dataset_config["window_size"],
            stride=dataset_config["stride"],
        )
        for key, frame in attack_frames.items()
    }
    all_attack_windows = []
    for windows in attack_windows.values():
        all_attack_windows.extend(windows)
    test_windows = np.array(all_attack_windows)

    training_data, validation_data = split_windows(
        normal_windows,
        valid_perc=dataset_config["validation_fraction"],
        shuffle=True,
        seed=config["random_seeds"]["split"],
    )
    scaled_training, scaled_validation, scaler = scale_windows(
        training_data, validation_data, method=dataset_config["normalization_method"]
    )
    scaled_test = transform_windows(test_windows, scaler)

    scaled_attacks = {}
    for key, windows in attack_windows.items():
        scaled_attacks[key] = transform_windows(windows, scaler).astype(np.float32)

    return {
        "training": scaled_training.astype(np.float32),
        "validation": scaled_validation.astype(np.float32),
        "test": scaled_test.astype(np.float32),
        "attacks": scaled_attacks,
        "attack_filenames": attack_filenames,
        "feature_names": [str(name).strip() for name in normal_frame.columns],
    }


def _select_device(config, override=None):
    selected = override or config["runtime"]["device"]
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(selected)


def train_configured_model(config, paths, datasets, device):
    """Fit a new model on normal windows and save the existing checkpoint format."""
    training_data = datasets["training"]
    model = TemporalVAE(
        seq_len=training_data.shape[1],
        feat_dim=training_data.shape[2],
        **config["model"]["parameters"],
    )
    train_variational_autoencoder(
        model,
        training_data,
        datasets["validation"],
        epochs=config["model"]["max_epochs"],
        verbose=1,
        device=device,
    )
    model.save(str(paths["checkpoint_dir"]))
    print(f"Saved trained TemporalVAE checkpoint to: {paths['checkpoint_dir']}")
    return model


def evaluate_model(config, paths, datasets, device, output_dir, model=None):
    """Evaluate a trained or loaded model and find the ROC threshold."""
    if config["model"]["name"] != "TemporalVAE":
        raise ValueError("This experiment expects model.name: TemporalVAE")
    if model is None:
        model = TemporalVAE.load(str(paths["checkpoint_dir"]))
        print(f"Loaded pretrained TemporalVAE checkpoint from: {paths['checkpoint_dir']}")
    model = model.to(device)
    print(f"Saving results under: {output_dir}")

    print("Normal Data Results:")
    print(model.evaluate_dataset(data=datasets["validation"], batch_size=model.batch_size))
    print("Anomaly Data Results:")
    print(model.evaluate_dataset(data=datasets["test"], batch_size=model.batch_size))
    print("Evaluation on data completed :) ")

    roc_auc, anomaly_scores, labels = calculate_reconstruction_roc_auc(
        model,
        normal_data=datasets["validation"],
        anomaly_data=datasets["test"],
        batch_size=model.batch_size,
    )
    print(f"ROC-AUC score: {roc_auc:.4f}")
    roc_auc, threshold = save_roc_curve(
        anomaly_scores=anomaly_scores,
        labels=labels,
        save_dir=str(output_dir),
        scenario_name="vae_mixed_data",
    )
    print(f"ROC-AUC: {roc_auc:.4f}, Best Threshold: {threshold:.4f}")
    return model, threshold


def _fit_support_representation(config, mode, model, combined_data):
    if mode == "vae":
        return model.encode_dataset_means(data=combined_data, batch_size=model.batch_size)
    if mode == "umap":
        flat_data = combined_data.reshape(combined_data.shape[0], -1)
        umap_config = config["experiment"]["umap"]
        reducer = umap.UMAP(
            n_neighbors=umap_config["n_neighbors"],
            n_components=umap_config["n_components"],
            min_dist=umap_config["min_dist"],
            random_state=config["random_seeds"]["umap"],
        ).fit(flat_data)
        return reducer, reducer.embedding_
    if mode == "none":
        return None
    raise ValueError(f"Unsupported neighbor mode: {mode}")


def _support_indices(config, mode, representation, reducer, query, combined_data, device, query_model):
    if mode == "vae":
        query_representation = query_model.encode_dataset_means(
            data=query, batch_size=query_model.batch_size
        )
        training_representation = torch.as_tensor(
            representation, dtype=torch.float32, device=device
        )
        query_representation = torch.as_tensor(
            query_representation, dtype=torch.float32, device=device
        )
        _, indices = search_neighbors(
            mode=mode,
            train_repr=training_representation,
            query_repr=query_representation,
            k=config["experiment"]["support_size"],
            return_distance=True,
            batch_size=2048,
        )
        return indices[0].tolist()
    if mode == "umap":
        query_embedding = reducer.transform(query.detach().cpu().numpy().reshape(1, -1))
        _, indices = search_neighbors(
            mode=mode,
            train_repr=representation,
            query_repr=query_embedding,
            k=config["experiment"]["support_size"],
            return_distance=True,
            batch_size=None,
            gpu_id=(config["runtime"]["faiss_gpu_id"] if device.type == "cuda" else None),
        )
        return indices[0].tolist()
    return np.random.choice(
        len(combined_data), config["experiment"]["support_size"], replace=False
    )


def _write_feature_outputs(key, feature_names, whole_percentages, overall_csv, sample_csv):
    with sample_csv.open("w", encoding="utf-8") as stream:
        stream.write("Feature,Absolute%,Clipped%\n")
        for name, value in zip(feature_names, whole_percentages):
            stream.write(f"{name},{value:.2f}\n")

    top_features = sorted(
        zip(feature_names, whole_percentages), key=lambda item: item[1], reverse=True
    )[:5]
    with overall_csv.open("a", encoding="utf-8") as stream:
        for name, value in top_features:
            stream.write(f"{key},{name},{value:.2f}\n")
        stream.write("\n")


def _get_feature_names(config, datasets):
    if config["dataset"]["mask_variable_names"]:
        return datasets["feature_names"]
    return [f"Feature{index}" for index in range(1, datasets["training"].shape[-1] + 1)]


def compute_shats_attributions_for_attacks(config, paths, datasets, model, threshold, device, output_dir):
    """Compute neighborhood selected ShaTS contributions per window and attack."""
    feature_names = _get_feature_names(config, datasets)
    wrapped_model = VAEWrapperTorch(model, threshold=threshold)
    combined_data = np.concatenate([datasets["training"], datasets["validation"]], axis=0)

    mode = config["experiment"]["knn_mode"]
    support_size = config["experiment"]["support_size"]
    print_inference_time = config["runtime"].get("print_inference_time", False)
    print(f"Filtering Mode is set as: {mode}")
    print(f"Number of Samples for Support Dataset set as: {support_size}")

    support_model = model
    if mode == "vae":
        support_model = TemporalVAE.load(str(paths["checkpoint_dir"])).to(device)
    fitted = _fit_support_representation(config, mode, support_model, combined_data)
    if mode == "umap":
        reducer, support_representation = fitted
    else:
        reducer = None
        support_representation = fitted

    overall_dir = output_dir / f"knn_filtering_mode{mode}_K{support_size}"
    overall_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving percentage files in OVERALL_DIR: {overall_dir}")
    overall_csv = overall_dir / "top5_percentages_all_samples.csv"
    overall_csv.write_text("Sample,Feature,Clipped%\n", encoding="utf-8")

    all_ranked_features = {}
    all_ranked_features_per_window = {}
    progress_stream = sys.__stderr__ or sys.stderr
    progress_options = {
        "unit": "window",
        "dynamic_ncols": True,
        "ascii": False,
        "bar_format": PROGRESS_BAR_FORMAT,
        "file": progress_stream,
        "mininterval": 0.2,
    }
    attack_count = len(datasets["attacks"])
    progress = tqdm(
        total=sum(len(windows) for windows in datasets["attacks"].values()),
        desc="Overall explainability", position=0,
        colour="cyan" if progress_stream.isatty() else None,
        **progress_options,
    )
    for attack_number, key in enumerate(sorted(datasets["attacks"]), start=1):
        attack_tensor = torch.tensor(datasets["attacks"][key], dtype=torch.float32).to(device)
        sample_name = datasets["attack_filenames"][key]
        sample_progress = tqdm(
            total=len(attack_tensor),
            desc=f"Sample {attack_number}/{attack_count} ({sample_name})",
            position=1, leave=False,
            colour="green" if progress_stream.isatty() else None,
            **progress_options,
        )
        sample_dir = output_dir / key
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_csv = sample_dir / f"{key}_percentages.csv"
        shapley_values = []

        for sample_index in range(attack_tensor.shape[0]):
            attack_window = attack_tensor[sample_index : sample_index + 1]
            if print_inference_time:
                start_inference = time.perf_counter()

            query_model = model
            if mode == "vae":
                # Match the source experiment's per-query checkpoint reload.
                query_model = TemporalVAE.load(str(paths["checkpoint_dir"])).to(device)
            neighbor_indices = _support_indices(
                config,
                mode,
                support_representation,
                reducer,
                attack_window,
                combined_data,
                device,
                query_model,
            )
            support_data = [
                torch.tensor(combined_data[index], dtype=torch.float32).to(device)
                for index in neighbor_indices
            ]
            support_tensor = torch.stack(support_data)
            estimator = FastShaTS(
                wrapped_model,
                support_dataset=support_tensor,
                grouping_strategy=FeaturesGroupingStrategy(names=feature_names),
            )
            if print_inference_time:
                start_explanation = time.perf_counter()
            shapley_values.append(estimator.compute(attack_window))
            progress.update(1)
            sample_progress.update(1)
            if print_inference_time:
                tqdm.write(f"Sample Index: {sample_index}")
                end_inference = time.perf_counter()
                end_explanation = time.perf_counter()
                print(f"Inference time VAE: {end_inference - start_inference:.6f} seconds")
                print(f"Inference time overall: {end_explanation - start_explanation:.6f} seconds")

        sample_progress.close()
        values = torch.cat(shapley_values, dim=0)
        anomaly_values = values[..., 1]
        whole_values = values[..., 1].sum(axis=0)
        window_percentages = shap_percentages_clip_window(anomaly_values)
        whole_percentages = shap_percentages_clip_whole(whole_values)
        window_percentages = window_percentages.cpu().numpy()
        whole_percentages = whole_percentages.cpu().numpy()

        all_ranked_features_per_window[key] = [
            list(zip(feature_names, row)) for row in window_percentages
        ]
        all_ranked_features[key] = list(zip(feature_names, whole_percentages))
        _write_feature_outputs(key, feature_names, whole_percentages, overall_csv, sample_csv)
        tqdm.write(
            f"Completed sample {attack_number}/{attack_count}: {sample_name} "
            f"({len(attack_tensor)} windows)", file=progress_stream,
        )

    progress.close()
    _print_top_k_metrics(
        all_ranked_features, all_ranked_features_per_window,
        config["evaluation"]["attack_sensors_dataset"],
    )
    print("All samples processed. Individual CSVs + Overall Top-5 CSV generated.")


def _print_top_k_metrics(all_ranked_features, all_ranked_features_per_window, attack_sensors_dataset):
    for k in (3, 5, 10):
        print(f"\n###METRICS FOR TOP K: {k}###")
        results = compute_topk_score_for_all_keys(
            all_ranked_features=all_ranked_features, k=k, normalize=True,
            attack_sensors_dataset=attack_sensors_dataset,
        )
        results.update(
            aggregate_early_metrics(
                all_ranked_features_per_window=all_ranked_features_per_window,
                k=k,
                mode="any",
                attack_sensors_dataset=attack_sensors_dataset,
            )
        )
        print_metric_results(results, k=k)


def run_experiment(config, overrides=None):
    """Run configured training, evaluation, and attribution."""
    paths = resolve_experiment_paths(config, overrides)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = paths["output_dir"] / f"{config['dataset']['name']}_E{config['model']['max_epochs']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"results_{timestamp}.txt"
    with results_file.open("x", encoding="utf-8") as stream:
        with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
            print(f"Results file: {results_file}")
            if (overrides or {}).get("config"):
                print(f"Config file: {overrides['config']}")
            print("Experiment configuration:")
            print(yaml.safe_dump(config, sort_keys=False).rstrip())
            active_overrides = {
                key: value for key, value in (overrides or {}).items()
                if key != "config" and value is not None
            }
            if active_overrides:
                print("Run overrides:")
                print(yaml.safe_dump(active_overrides, sort_keys=False).rstrip())
            training_enabled = config["training"]["enabled"]
            if training_enabled and paths["checkpoint_dir"] is None:
                paths["checkpoint_dir"] = output_dir / config["outputs"]["checkpoint_subdirectory"]
            print(f"Checkpoint directory: {paths['checkpoint_dir']}")
            _create_directories(paths)
            get_attack_sensors(config["evaluation"]["attack_sensors_dataset"])
            _validate_inputs(paths, training_enabled=training_enabled)
            device = _select_device(config, (overrides or {}).get("device"))
            seed = config["random_seeds"]["global"]
            seed_everything(seed, config["random_seeds"]["deterministic"])
            print(f"Seed set to {seed}")
            print("Using device:", device)

            datasets = prepare_datasets(config, paths)
            trained_model = train_configured_model(config, paths, datasets, device) if training_enabled else None
            model, threshold = evaluate_model(
                config, paths, datasets, device, output_dir, model=trained_model
            )
            compute_shats_attributions_for_attacks(config, paths, datasets, model, threshold, device, output_dir)
