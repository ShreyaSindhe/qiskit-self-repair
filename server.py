"""Backend for the Qiskit self-repair web tool.

  python server.py            ->  open http://localhost:8000

Uses only Python's standard library (no Flask/FastAPI to install).

API
  GET  /api/health             server status, Qiskit version, providers
  GET  /api/datasets           available task files in data/
  GET  /api/tasks?dataset=...  tasks in one file
  POST /api/solve              runs the self-repair loop; streams every backend step
                               to the browser as Server-Sent Events (text/event-stream)
  GET  /api/results            summary of results/*.jsonl (from evaluate.py)
"""
from __future__ import annotations

import argparse
import glob
import json
import mimetypes
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import llm
from ask import make_ask_task, parse_result
from dataset import load_tasks
from executor import CATEGORY_LABELS
from prompts import FEEDBACK_MODES, qiskit_version
from repair_loop import solve_task

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
_task_cache: dict[str, list[dict]] = {}


def get_tasks(name: str) -> list[dict]:
    path = (DATA / Path(name).name).resolve()  # only files directly inside data/
    if path.parent != DATA.resolve() or not path.exists():
        raise ValueError(f"Unknown dataset: {name}")
    key = f"{path}:{path.stat().st_mtime}"
    if key not in _task_cache:
        _task_cache[key] = load_tasks(path)
    return _task_cache[key]


def describe_event(stage: str, info: dict) -> tuple[str, str, dict]:
    """Turn a repair-loop event into (pipeline step, log line, details) for the frontend."""
    n = info.get("attempt")
    if stage == "llm_request":
        return "llm", (f"→ LLM request · attempt {n} · model {info['model']} · "
                       f"{info['messages']} messages, {info['prompt_chars']} chars"), \
            {"last message sent": info["last_message"][:3000]}
    if stage == "llm_reply":
        return "llm", f"← LLM reply in {info['seconds']}s · {info['chars']} chars", {}
    if stage == "extract":
        return "exec", (f"Extracted code from the reply: {info['lines']} lines" if info["found"]
                        else "No code block found in the reply"), {}
    if stage == "exec_start":
        return "exec", (f"▶ Running code + tests in a separate Python process "
                        f"(timeout {info['timeout']}s)"), {}
    if stage == "exec_result":
        if info["passed"]:
            return "exec", f"✓ Passed ({info['seconds']}s)", {}
        msg = f"{info['error_type']}: {info['error_message']}".strip(": ")
        return "exec", f"✗ {CATEGORY_LABELS[info['category']]} · {msg[:300]}", {}
    if stage == "feedback":
        return "llm", f"↩ Feeding the error back to the model (feedback mode: {info['mode']})", \
            {"feedback sent": info["text"]}
    return "api", stage, info


class Handler(BaseHTTPRequestHandler):
    server_version = "QiskitSelfRepair/1.0"

    # ---------------------------------------------------------------- helpers
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        if not path.is_file():
            return self.send_json({"error": "not found"}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                return self.send_file(FRONTEND / "index.html")
            if url.path.startswith("/static/"):
                return self.send_file(FRONTEND / Path(url.path).name)
            if url.path == "/api/health":
                return self.send_json({
                    "status": "ok", "qiskit_version": qiskit_version(),
                    "providers": {name: {"model": m, "key_env": env, "key_set": bool(env and os.environ.get(env))}
                                  for name, (_, env, m) in llm.PRESETS.items()},
                    "feedback_modes": FEEDBACK_MODES,
                })
            if url.path == "/api/datasets":
                files = sorted(p.name for p in DATA.glob("*.json*"))
                return self.send_json({"datasets": files})
            if url.path == "/api/tasks":
                name = parse_qs(url.query).get("dataset", ["sample_tasks.jsonl"])[0]
                tasks = get_tasks(name)
                return self.send_json({"tasks": [
                    {k: t[k] for k in ("task_id", "entry_point", "difficulty_scale", "prompt", "test")}
                    for t in tasks]})
            if url.path == "/api/results":
                return self.send_json(results_summary())
            self.send_json({"error": "not found"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    # ---------------------------------------------------------------- POST /api/solve
    def do_POST(self):
        if urlparse(self.path).path != "/api/solve":
            return self.send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(min(length, 100_000)) or b"{}")
            provider = req.get("provider", "mock")
            if provider not in llm.PRESETS and provider != "mock":
                raise ValueError(f"Unknown provider: {provider}")
            feedback = req.get("feedback", "full")
            if feedback not in FEEDBACK_MODES:
                raise ValueError(f"Unknown feedback mode: {feedback}")
            max_attempts = max(1, min(6, int(req.get("max_attempts", 4))))
            if req.get("mode") == "ask":
                question = str(req.get("question", "")).strip()
                if not question:
                    raise ValueError("Type a question first.")
                task, tasks = make_ask_task(question[:4000]), []
            else:
                tasks = get_tasks(req.get("dataset", "sample_tasks.jsonl"))
                task = next((t for t in tasks if t["task_id"] == req.get("task_id")), None)
                if task is None:
                    raise ValueError(f"Unknown task: {req.get('task_id')}")
            model = req.get("model") or (llm.PRESETS[provider][2] if provider in llm.PRESETS else "mock")
            client = llm.make_client(provider, model, api_key=req.get("api_key") or None, tasks=tasks)
        except (ValueError, json.JSONDecodeError) as e:
            return self.send_json({"error": str(e)}, 400)

        # stream the run as Server-Sent Events
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        start = time.time()

        def send(event: dict):
            event["t"] = round(time.time() - start, 2)
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

        def on_event(stage, **info):
            step, msg, details = describe_event(stage, info)
            send({"type": "log", "step": step, "stage": stage, "msg": msg, "details": details})

        try:
            shown = {k: v for k, v in req.items() if k != "api_key"}
            send({"type": "log", "step": "api", "stage": "request",
                  "msg": f"POST /api/solve received · mode={req.get('mode', 'benchmark')} · "
                         f"provider={provider} · max_attempts={max_attempts} · feedback={feedback}",
                  "details": {"request body (API key hidden)": shown}})
            solved = False
            for att in solve_task(task, client, max_attempts, feedback, on_event=on_event):
                d = att.to_dict()
                d.pop("reply")
                d["category_label"] = CATEGORY_LABELS[att.category]
                d["result"] = parse_result(att.stdout) if task.get("free_form") else None
                send({"type": "attempt", "attempt": d})
                solved = att.passed
            send({"type": "done", "solved": solved, "attempts": att.number,
                  "msg": f"Finished in {round(time.time() - start, 1)}s · "
                         + ("solved" if solved else "not solved") + f" after {att.number} attempt(s)"})
        except llm.LLMError as e:
            send({"type": "error", "msg": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            print("Browser disconnected during a run.")
        except Exception as e:
            traceback.print_exc()
            try:
                send({"type": "error", "msg": f"Server error: {type(e).__name__}: {e}"})
            except Exception:
                pass


def results_summary():
    files = sorted(glob.glob(str(RESULTS / "*.jsonl")))
    if not files:
        return {"runs": [], "pass_at_k": {}, "fix_rate": []}
    import analyze  # needs pandas; only loaded when results exist
    valid = RESULTS / "canonical.jsonl"
    runs = [f for f in files if Path(f).name != "canonical.jsonl"] or files
    df = analyze.load_runs(runs, valid_from=str(valid) if valid.exists() and len(runs) < len(files) else None)
    if df.empty:
        return {"runs": [], "pass_at_k": {}, "fix_rate": []}
    summary = analyze.summary_table(df).reset_index()
    pak = analyze.pass_at_k(df)
    return {
        "runs": json.loads(summary.to_json(orient="records")),
        "pass_at_k": {run: {str(k): (None if v != v else round(v, 1)) for k, v in row.items()}
                      for run, row in pak.iterrows()},
        "fix_rate": json.loads(analyze.fix_rate_by_category(df).to_json(orient="records")),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Qiskit self-repair backend · Qiskit {qiskit_version()}")
    print(f"Open http://localhost:{a.port} in your browser (Ctrl+C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
