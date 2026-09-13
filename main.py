"""
Runs the full pipeline end to end:
  synthetic data (if needed) -> load -> feature engineer -> train ->
  evaluate -> explain

Usage:
  python main.py
"""
import os
import subprocess
import sys

STEPS = [
    ("Generating synthetic data (skip this once you add real data)",
     ["python", "scripts/generate_synthetic_data.py"]),
    ("Training models", ["python", "src/train.py"]),
    ("Evaluating models", ["python", "src/evaluate.py"]),
    ("Generating SHAP explainability plots", ["python", "src/explain.py"]),
]


def run():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    has_real_data = os.path.exists("data/raw/cs-training.csv")
    for i, (label, cmd) in enumerate(STEPS):
        if i == 0 and has_real_data:
            print(f"[skip] {label} -- data/raw/cs-training.csv already exists")
            continue
        print(f"\n{'#' * 60}\n# {label}\n{'#' * 60}")
        # src/ scripts import each other by relative module name, so run
        # them with src/ on the path rather than as plain file paths.
        env = dict(os.environ, PYTHONPATH="src")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"Step failed: {label}")
            sys.exit(result.returncode)

    print("\nPipeline complete. See reports/ for outputs, models/ for saved models.")


if __name__ == "__main__":
    run()
