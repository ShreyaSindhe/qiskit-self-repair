"""Prompts sent to the model, the error feedback, and extraction of code from replies."""
from __future__ import annotations

import re

from executor import ExecResult


def qiskit_version() -> str:
    try:
        import qiskit
        return qiskit.__version__
    except Exception:
        return "unknown"


def system_prompt() -> str:
    return (
        "You are an expert Python programmer who writes code with Qiskit "
        f"(installed version: {qiskit_version()}). Complete the function you are given. "
        "Reply with ONE Python code block that contains the full function, including any imports. "
        "Do not include example usage or explanations."
    )


def initial_messages(task: dict) -> list[dict]:
    if task.get("free_form"):  # "Ask the assistant" mode in the web app
        user = ("Write Python code with Qiskit for this request:\n\n"
                f"{task['prompt'].strip()}\n\n"
                "Put all the work inside a function `solve()` that takes no arguments and RETURNS "
                "the main result: a QuantumCircuit, a counts dict, a Statevector, or a number. "
                "Use simulators only (e.g. qiskit.primitives.StatevectorSampler). "
                "Do not call solve() yourself.")
    else:
        user = ("Complete the following Python function:\n\n"
                f"```python\n{task['prompt'].rstrip()}\n```")
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


# --- feedback --------------------------------------------------------------------------
# "full":    the traceback from running the tests (most informative; note it can reveal
#            lines of the hidden test, which is worth discussing in the report)
# "minimal": only the final error line, e.g. "ImportError: cannot import name 'execute'"
# "none":    only "it failed" -> tells you whether the error text helps or the model
#            just gets a second chance
FEEDBACK_MODES = ("full", "minimal", "none")


def feedback_text(result: ExecResult, mode: str) -> str:
    if mode == "none":
        return "The code did not pass the tests."
    if result.category == "no_code":
        return "Your reply did not contain a Python code block."
    if result.category == "timeout":
        return result.error_message
    if mode == "minimal":
        msg = f": {result.error_message}" if result.error_message else ""
        hint = " (the function returned a wrong result)" if result.category == "wrong_result" else ""
        return f"{result.error_type}{msg}{hint}"
    return "Traceback (last lines):\n" + result.traceback


def repair_message(feedback: str) -> dict:
    return {"role": "user", "content": (
        "Your code failed when it was run against the tests.\n\n"
        f"```text\n{feedback}\n```\n\n"
        "Fix the code. Reply with the complete corrected function in one Python code block."
    )}


# --- code extraction ---------------------------------------------------------------------
_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def _prompt_imports(prompt: str) -> str:
    lines = []
    for line in prompt.splitlines():
        if line.startswith("def ") or line.startswith("class "):
            break
        if line.startswith(("import ", "from ")):
            lines.append(line)
    return "\n".join(lines)


def extract_code(reply: str, task: dict) -> str:
    blocks = _FENCE.findall(reply)
    if not blocks:
        # unclosed fence or no fence: take everything after an opening fence, else the raw text
        m = re.search(r"```(?:python|py)?[ \t]*\n(.*)", reply, re.S)
        blocks = [m.group(1)] if m else ([reply] if "def " in reply or "return" in reply else [])
    if not blocks:
        return ""
    entry = task["entry_point"]
    with_def = [b for b in blocks if re.search(rf"def\s+{re.escape(entry)}\s*\(", b)]
    code = max(with_def or blocks, key=len)
    if task.get("free_form"):
        return code.strip() + "\n"
    if not with_def:
        # the model returned only the body: put it under the original signature
        code = task["prompt"].rstrip() + "\n" + code
    # make sure the prompt's imports are present (duplicates are harmless)
    imports = _prompt_imports(task["prompt"])
    return (imports + "\n\n" + code.strip() + "\n") if imports else code.strip() + "\n"
