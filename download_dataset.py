"""Download Qiskit HumanEval (151 tasks) from Hugging Face into data/qiskit_humaneval.jsonl.

  pip install datasets
  python download_dataset.py
"""
import json
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Run `pip install datasets` first.")

ds = load_dataset("Qiskit/qiskit_humaneval", split="test")
out = Path("data/qiskit_humaneval.jsonl")
out.parent.mkdir(exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(dict(row)) + "\n")
print(f"Saved {len(ds)} tasks to {out}")
