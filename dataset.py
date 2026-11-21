"""Load Qiskit HumanEval (or any HumanEval-style) tasks from .json or .jsonl."""
from __future__ import annotations

import json
from pathlib import Path

REQUIRED = ("task_id", "prompt", "test", "entry_point")


def load_tasks(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        tasks = json.loads(text)
    else:
        tasks = [json.loads(line) for line in text.splitlines() if line.strip()]
    for t in tasks:
        missing = [k for k in REQUIRED if k not in t]
        if missing:
            raise ValueError(f"Task {t.get('task_id', '?')} is missing fields {missing}")
        t.setdefault("canonical_solution", "")
        t.setdefault("difficulty_scale", "unknown")
    return tasks


def filter_tasks(tasks, limit=None, difficulty=None, task_ids=None):
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t["task_id"] in wanted]
    if difficulty:
        tasks = [t for t in tasks if t["difficulty_scale"] == difficulty]
    if limit:
        tasks = tasks[:limit]
    return tasks
