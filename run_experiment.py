"""Single command-line entry point for the research experiment."""
import argparse
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from experiment.config import DEFAULT_CONFIG, load_config
from experiment.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the SWaT TemporalVAE feature-attribution experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment YAML file")
    parser.add_argument("--normal-csv", help="Override configured normal-data CSV")
    parser.add_argument("--anomaly-dir", help="Override configured attack CSV directory")
    parser.add_argument("--checkpoint-dir", help="Override configured pretrained checkpoint directory")
    parser.add_argument("--output-dir", help="Override configured output root")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override configured PyTorch device")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config, vars(args))


if __name__ == "__main__":
    main()
