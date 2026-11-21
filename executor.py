"""Run a candidate solution against the task's unit tests in a separate Python process.

Each run happens in a fresh temporary folder with a time limit, so a crash or an
infinite loop in model-written code cannot take the evaluator down with it.
NOTE: this is process isolation, not a security sandbox. Run the project inside a
virtual environment (or a container) since the code is written by an LLM.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

CATEGORY_LABELS = {
    "passed": "Passed",
    "syntax": "Syntax error",
    "api": "Import / API error",
    "wrong_result": "Wrong result (test failed)",
    "runtime": "Other runtime error",
    "timeout": "Timeout",
    "no_code": "No code in reply",
}

_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))\b:?\s?(.*)$")


@dataclass
class ExecResult:
    passed: bool
    category: str
    error_type: str = ""
    error_message: str = ""
    traceback: str = ""
    duration: float = 0.0
    stdout: str = ""


def build_program(code: str, test: str, entry_point: str) -> str:
    program = code.rstrip() + "\n\n\n" + test.rstrip() + "\n"
    if re.search(r"^def check\(", test, flags=re.M):
        program += f"\n\ncheck({entry_point})\n"
    return program


def classify(error_type: str, message: str) -> str:
    short = error_type.rsplit(".", 1)[-1]
    if short in ("SyntaxError", "IndentationError", "TabError"):
        return "syntax"
    if short in ("ImportError", "ModuleNotFoundError", "AttributeError"):
        return "api"
    if short == "TypeError" and re.search(r"argument|keyword|positional|not callable", message):
        return "api"  # wrong signature for a Qiskit function = API misuse
    if short == "AssertionError":
        return "wrong_result"
    return "runtime"


def run_candidate(code: str, test: str, entry_point: str, timeout: int = 60) -> ExecResult:
    program = build_program(code, test, entry_point)
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="qhe_") as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=tmp, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout", "Timeout",
                              f"Code did not finish within {timeout} seconds.",
                              duration=time.time() - start)
        duration = time.time() - start
        if proc.returncode == 0:
            return ExecResult(True, "passed", duration=duration, stdout=proc.stdout[-20000:])
        stderr = proc.stderr.replace(str(path), "solution.py").replace(tmp, ".")

    lines = [l for l in stderr.splitlines() if l.strip()]
    error_type, message = "UnknownError", (lines[-1] if lines else f"exit code {proc.returncode}")
    for line in reversed(lines):
        m = _EXC_LINE.match(line.strip())
        if m:
            error_type, message = m.group(1), m.group(2)
            break
    tb = "\n".join(lines[-30:])  # keep the useful end of long tracebacks
    return ExecResult(False, classify(error_type, message), error_type, message, tb, duration,
                      proc.stdout[-20000:])
