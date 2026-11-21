"""Minimal chat client for any OpenAI-compatible API (Ollama, Groq, OpenAI, Gemini, OpenRouter).

Uses only the Python standard library, so there is no SDK to install.
Also includes two offline clients used for testing:
  - "canonical": returns the dataset's reference solution (checks your environment)
  - "mock":      returns a broken answer first, then the reference solution
                 (checks that the repair loop and the demo app work, without an API key)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

# provider -> (base_url, env var holding the API key, example model)
PRESETS = {
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5-coder:7b"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen-2.5-coder-32b-instruct"),
}
OFFLINE_PROVIDERS = ("mock", "canonical")


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, provider: str, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 2048, timeout: int = 180):
        if provider not in PRESETS and not base_url:
            raise ValueError(f"Unknown provider '{provider}'. Use one of {list(PRESETS)} or pass base_url.")
        preset_url, key_env, _ = PRESETS.get(provider, (None, None, None))
        self.provider = provider
        self.model = model
        self.base_url = (base_url or preset_url).rstrip("/")
        self.api_key = api_key or (os.environ.get(key_env) if key_env else None) or "not-needed"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        delay = 5
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(delay)  # rate limit / server hiccup: back off and retry
                    delay *= 2
                    continue
                raise LLMError(f"HTTP {e.code} from {self.base_url}: {detail}") from e
            except urllib.error.URLError as e:
                raise LLMError(f"Could not reach {self.base_url}: {e.reason}. "
                               "Is the server running / the URL right?") from e
        raise LLMError("Gave up after repeated rate-limit errors.")


class _OfflineClient:
    """Base for the offline clients: finds the task from the prompt text."""

    def __init__(self, tasks: list[dict], model: str = ""):
        self.tasks = tasks
        self.provider = self.__class__.__name__
        self.model = model or self.provider

    def _task_for(self, messages):
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        for t in self.tasks:
            if t["prompt"].strip() in first_user:
                return t
        raise LLMError("Offline client could not match the prompt to a task.")

    @staticmethod
    def _fence(code: str) -> str:
        return f"```python\n{code}\n```"


class CanonicalClient(_OfflineClient):
    def chat(self, messages):
        t = self._task_for(messages)
        return self._fence(t["prompt"] + t["canonical_solution"])


class MockClient(_OfflineClient):
    """First attempt uses the removed `qiskit.execute` API (a very common real LLM mistake),
    the repair attempt returns the reference solution."""

    ASK_ANSWER = (
        "from qiskit import QuantumCircuit\n"
        "from qiskit.primitives import StatevectorSampler\n\n\n"
        "def solve():\n"
        "    qc = QuantumCircuit(2)\n"
        "    qc.h(0)\n"
        "    qc.cx(0, 1)\n"
        "    qc.measure_all()\n"
        "    result = StatevectorSampler().run([qc], shots=1024).result()\n"
        "    return result[0].data.meas.get_counts()\n"
    )

    def chat(self, messages):
        first_try = sum(m["role"] == "assistant" for m in messages) == 0
        try:
            t = self._task_for(messages)
            good = t["prompt"] + t["canonical_solution"]
        except LLMError:  # free-form question: always answers with a Bell-state example
            good = self.ASK_ANSWER
        if first_try:
            return self._fence("from qiskit import execute  # removed in Qiskit 1.0\n" + good)
        return self._fence(good)


def make_client(provider: str, model: str = "", base_url: str | None = None,
                api_key: str | None = None, tasks: list[dict] | None = None, **kw):
    if provider == "mock":
        return MockClient(tasks or [], model)
    if provider == "canonical":
        return CanonicalClient(tasks or [], model)
    return ChatClient(provider, model or PRESETS[provider][2], base_url=base_url, api_key=api_key, **kw)
