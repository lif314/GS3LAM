import argparse
import os
import shutil
from importlib.machinery import SourceFileLoader
from src.GS3LAM import run_gs3lam


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )

    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    run_gs3lam(experiment.config)