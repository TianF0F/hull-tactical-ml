from pathlib import Path

FILES = {
    "configs/baseline_ridge.yaml": """run:
  name: baseline_ridge
  seed: 42
  target_col: TARGET
  time_col: Date
  out_dir: runs
""",

    "configs/gbdt_hist.yaml": """run:
  name: gbdt_hist
  seed: 42
  target_col: TARGET
  time_col: Date
  out_dir: runs
""",

    "scripts/train.py": "# training entrypoint\n",
    "scripts/eval.py": "# evaluation entrypoint\n",

    "src/__init__.py": "",
    "src/utils/seed.py": "def set_seed(seed: int): pass\n",
    "src/utils/logging.py": "def get_logger(): pass\n",
    "src/utils/io.py": "",

    "src/data/load.py": "",
    "src/data/split.py": "",
    "src/data/preprocess.py": "",

    "src/features/build.py": "",

    "src/models/factory.py": "",
    "src/models/baseline.py": "",
    "src/models/gbdt.py": "",

    "src/evaluation/metrics.py": "",
    "src/evaluation/backtest.py": "",
}

for path, content in FILES.items():
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

print("✅ Project structure generated.")
