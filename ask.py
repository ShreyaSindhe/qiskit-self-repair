"""'Ask the assistant' mode: the user types any Qiskit request in plain English.

There are no hidden unit tests for a free-form request, so the check is:
does the code run, and does solve() return a real result? The harness below calls
solve(), describes the result (circuit drawing, counts, probabilities, value) and prints
it as JSON for the frontend. Any exception becomes the feedback for the repair loop.
"""
from __future__ import annotations

import json

MARKER = "<<<RESULT>>>"

HARNESS = r'''
import json as _json


def _describe(r):
    out = {"type": type(r).__name__}
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        QuantumCircuit = None
    if QuantumCircuit is not None and isinstance(r, QuantumCircuit):
        out.update(kind="circuit", qubits=r.num_qubits, depth=r.depth(),
                   ops={k: int(v) for k, v in r.count_ops().items()},
                   drawing=str(r.draw(output="text", fold=-1)))
        try:
            from qiskit.quantum_info import Statevector
            bare = r.remove_final_measurements(inplace=False)
            if bare.num_qubits <= 10:
                probs = Statevector.from_instruction(bare).probabilities_dict()
                out["probabilities"] = {k: round(float(v), 4) for k, v in probs.items() if v > 1e-6}
        except Exception:
            pass  # e.g. mid-circuit measurements: just show the drawing
        return out
    if hasattr(r, "probabilities_dict"):  # Statevector / DensityMatrix
        probs = r.probabilities_dict()
        out.update(kind="state", probabilities={k: round(float(v), 4) for k, v in probs.items() if v > 1e-6})
        return out
    if isinstance(r, dict) and r and all(isinstance(k, str) for k in r) \
            and all(isinstance(v, (int, float)) for v in r.values()):
        out.update(kind="counts", counts={k: r[k] for k in sorted(r)})
        return out
    out.update(kind="value", value=repr(r)[:3000])
    return out


_result = solve()
if _result is None:
    raise ValueError("solve() returned None. It must RETURN the result "
                     "(a QuantumCircuit, a counts dict, a Statevector or a number).")
print("''' + MARKER + r'''" + _json.dumps(_describe(_result)))
'''


def make_ask_task(question: str) -> dict:
    return {
        "task_id": "ask",
        "prompt": question.strip(),
        "test": HARNESS,
        "entry_point": "solve",
        "difficulty_scale": "user",
        "canonical_solution": "",
        "free_form": True,
    }


def parse_result(stdout: str):
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER):])
            except json.JSONDecodeError:
                return None
    return None
