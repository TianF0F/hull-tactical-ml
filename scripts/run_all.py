# scripts/run_all.py
import subprocess
import sys

CONFIGS = [
    "configs/baseline_ridge.yaml",
    "configs/xgboost.yaml",
    "configs/lstm.yaml",
    "configs/transformer.yaml",
]


def main():
    for cfg in CONFIGS:
        print("=" * 80)
        print(f"Running experiment: {cfg}")
        print("=" * 80)

        cmd = [
            sys.executable,
            "-m",
            "scripts.train",
            "--config",
            cfg,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] Experiment failed: {cfg}")
            break


if __name__ == "__main__":
    main()
