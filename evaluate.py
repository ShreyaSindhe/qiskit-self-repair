"""Run the experiment: every task, with the self-repair loop, results saved as JSONL.

Examples
  # 1) check your environment with the reference solutions (no LLM needed)
  python evaluate.py --provider canonical --max-attempts 1 --run canonical

  # 2) local model with Ollama, full error feedback, up to 4 attempts
  python evaluate.py --provider ollama --model qwen2.5-coder:7b --feedback full

  # 3) same model, no error text (ablation)
  python evaluate.py --provider ollama --model qwen2.5-coder:7b --feedback none

The run can be stopped and restarted: tasks already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset import filter_tasks, load_tasks
from executor import CATEGORY_LABELS
from llm import PRESETS, LLMError, make_client
from prompts import FEEDBACK_MODES, qiskit_version
from repair_loop import solve_task

DEFAULT_DATASET = "data/qiskit_humaneval.jsonl" if Path("data/qiskit_humaneval.jsonl").exists() \
    else "data/sample_tasks.jsonl"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--provider", required=True, choices=list(PRESETS) + ["mock", "canonical"])
    p.add_argument("--model", default="")
    p.add_argument("--base-url", default=None, help="override the provider's API URL")
    p.add_argument("--api-key", default=None, help="or set the provider's env var, e.g. GROQ_API_KEY")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--feedback", choices=FEEDBACK_MODES, default="full")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=60, help="seconds allowed per test run")
    p.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    p.add_argument("--difficulty", default=None, help="basic / intermediate / difficult")
    p.add_argument("--run", default=None, help="run name (default: model-feedback)")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args(argv)

    tasks = filter_tasks(load_tasks(a.dataset), limit=a.limit, difficulty=a.difficulty)
    model = a.model or (PRESETS[a.provider][2] if a.provider in PRESETS else a.provider)
    extra = {} if a.provider in ("mock", "canonical") else {"temperature": a.temperature}
    client = make_client(a.provider, model, base_url=a.base_url, api_key=a.api_key, tasks=tasks, **extra)

    run = a.run or f"{model.replace('/', '_').replace(':', '-')}__{a.feedback}"
    out = Path(a.out_dir) / f"{run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["task_id"] for l in out.read_text().splitlines() if l.strip()}

    print(f"Run '{run}': {len(tasks)} tasks ({len(done)} already done), model={model}, "
          f"feedback={a.feedback}, max_attempts={a.max_attempts}, qiskit={qiskit_version()}")
    print(f"Saving to {out}\n")

    n_pass1 = n_pass = n_total = 0
    with out.open("a", encoding="utf-8") as f:
        for i, task in enumerate(tasks, 1):
            if task["task_id"] in done:
                continue
            attempts = []
            try:
                for att in solve_task(task, client, a.max_attempts, a.feedback, a.timeout):
                    attempts.append(att)
            except LLMError as e:
                print(f"\nLLM error on {task['task_id']}: {e}\nStopping; rerun the same command to resume.")
                sys.exit(1)

            passed_at = attempts[-1].number if attempts[-1].passed else None
            record = {
                "run": run, "provider": a.provider, "model": model, "feedback": a.feedback,
                "max_attempts": a.max_attempts, "qiskit_version": qiskit_version(),
                "task_id": task["task_id"], "difficulty": task["difficulty_scale"],
                "passed": passed_at is not None, "passed_at": passed_at,
                "attempts": [x.to_dict() for x in attempts],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            n_total += 1
            n_pass += passed_at is not None
            n_pass1 += passed_at == 1
            trail = " -> ".join("PASS" if x.passed else CATEGORY_LABELS[x.category] for x in attempts)
            print(f"[{i:>3}/{len(tasks)}] {task['task_id']:<24} {trail}")

    if n_total:
        print(f"\nThis session: {n_total} tasks | first try {n_pass1}/{n_total} "
              f"({100 * n_pass1 / n_total:.1f}%) | with repair {n_pass}/{n_total} "
              f"({100 * n_pass / n_total:.1f}%)")
    print(f"Full report: python analyze.py {out}")


if __name__ == "__main__":
    main()
