import argparse
import yaml
from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.evaluate import start_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained NeuralHydrology model.")
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to the training run directory (must contain config.yml)."
    )
    parser.add_argument(
        "--gage-file", type=Path, default=Path("data/n5/gages/gage_list_clean.txt"),
                        help="Whole gage list file (all mode)."
    )
    args = parser.parse_args()

    # Load config
    config_path = args.run_dir / "config.yml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Clean problematic keys
    config_dict.pop("run_dir_is_set", None)
    config_dict.pop("resume_training", None)

    # Override test basins if provided
    if args.gage_file is not None:
        config_dict["test_basin_file"] = str(args.gage_file)

    # Create config object
    cfg = Config(config_dict)

    # Run evaluation
    start_evaluation(cfg=cfg, run_dir=args.run_dir, period='test')


if __name__ == "__main__":
    main()
