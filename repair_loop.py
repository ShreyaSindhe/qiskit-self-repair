"""The core idea of the project: generate -> run tests -> feed the error back -> retry."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from executor import ExecResult, run_candidate
from prompts import extract_code, feedback_text, initial_messages, repair_message


@dataclass
class Attempt:
    number: int
    code: str
    passed: bool
    category: str
    error_type: str
    error_message: str
    feedback: str        # what was (or would be) sent back to the model
    llm_seconds: float
    run_seconds: float
    reply: str
    stdout: str = ""

    def to_dict(self):
        return asdict(self)


def _noop(stage, **info):
    pass


def solve_task(task: dict, client, max_attempts: int = 4, feedback_mode: str = "full",
               timeout: int = 60, on_event=None):
    """Yields one Attempt per try and stops at the first pass or after max_attempts.
    Attempt 1 is the plain baseline; attempts 2+ are the self-repair loop.
    `on_event(stage, **info)` is called at every step (the web backend uses it to show its work)."""
    emit = on_event or _noop
    messages = initial_messages(task)
    for n in range(1, max_attempts + 1):
        emit("llm_request", attempt=n, messages=len(messages),
             prompt_chars=sum(len(m["content"]) for m in messages),
             model=getattr(client, "model", ""), last_message=messages[-1]["content"])
        t0 = time.time()
        reply = client.chat(messages)
        llm_s = time.time() - t0
        emit("llm_reply", attempt=n, seconds=round(llm_s, 2), chars=len(reply))

        code = extract_code(reply, task)
        emit("extract", attempt=n, lines=len(code.splitlines()), found=bool(code.strip()))
        if code.strip():
            emit("exec_start", attempt=n, timeout=timeout)
            result = run_candidate(code, task["test"], task["entry_point"], timeout=timeout)
        else:
            result = ExecResult(False, "no_code", "NoCode", "No code block found in reply")

        fb = "" if result.passed else feedback_text(result, feedback_mode)
        emit("exec_result", attempt=n, passed=result.passed, category=result.category,
             error_type=result.error_type, error_message=result.error_message,
             seconds=round(result.duration, 2))
        yield Attempt(n, code, result.passed, result.category, result.error_type,
                      result.error_message, fb, round(llm_s, 2), round(result.duration, 2), reply,
                      result.stdout)
        if result.passed:
            return
        if n < max_attempts:
            emit("feedback", attempt=n, mode=feedback_mode, text=fb)
        messages = messages + [{"role": "assistant", "content": reply}, repair_message(fb)]
