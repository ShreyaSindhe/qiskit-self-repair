# LLM assistant that writes and self-repairs Qiskit code

Quantum Computing (25AM3504) course project.
Base paper: *Qiskit HumanEval: An Evaluation Benchmark for Quantum Code Generative Models* (Vishwakarma et al., 2024).

**Question:** if an LLM is shown the error its Qiskit code produced and asked to fix it,
how much does its pass rate on Qiskit HumanEval improve, and which kinds of mistakes can it fix?

```
 task prompt ──► LLM ──► code ──► run hidden unit tests ──► pass? ──yes──► done
                  ▲                                          │
                  └──── error / traceback fed back ◄──no─────┘   (up to N attempts)
```

Attempt 1 is the normal baseline (pass@1). Attempts 2…N are the self-repair loop.

## The web tool (frontend + backend)

```bash
python server.py          # then open http://localhost:8000
```

```
 Frontend (browser)            Backend (server.py)                 LLM / Executor
 frontend/index.html  ──POST /api/solve──►  repair_loop.solve_task  ──►  LLM API (Groq, Ollama…)
 frontend/app.js      ◄── live event stream ──   (Server-Sent Events) ──►  Python + Qiskit + tests
```

- **Ask the assistant:** type any request in plain English ("3-qubit GHZ state, return counts").
  The LLM writes a `solve()` function, the backend runs it, and errors go back to the LLM. The result
  is shown as a circuit drawing, a counts bar chart or probabilities.
- **Benchmark task:** run one Qiskit HumanEval task against its hidden unit tests.
- **Experiment results:** tables and the pass-rate chart from `results/*.jsonl`.
- The **Backend · Live activity** panel shows every step the server takes: request received, LLM
  call, code extracted, test run, error, feedback sent. Click ▸ on a line to see the exact data.
- The strip at the top lights up the part that's working: Frontend → Backend API → LLM → Executor.

The backend uses only Python's standard library, so there's nothing extra to install. API keys typed in the page
are sent to your local backend only and never stored. Setting the env var (e.g. `GROQ_API_KEY`) also works.
Select **mock** to try the tool without a key. It fails once on purpose and then shows the fix.

| Backend endpoint | Purpose |
|---|---|
| `GET /api/health` | status, Qiskit version, providers |
| `GET /api/datasets`, `GET /api/tasks?dataset=…` | benchmark files and tasks |
| `POST /api/solve` | runs the repair loop, streams each step as an event |
| `GET /api/results` | summary of finished experiment runs |

## Files

| File | What it does |
|---|---|
| `evaluate.py` | Runs the experiment over all tasks and saves one JSON line per task to `results/` |
| `repair_loop.py` | The loop: generate → test → feed back error → retry |
| `executor.py` | Runs code + tests in a separate process with a timeout; classifies the error |
| `prompts.py` | Prompts, the 3 feedback modes, and extracting code from the model's reply |
| `llm.py` | Client for any OpenAI-compatible API (Ollama, Groq, Gemini, OpenAI, OpenRouter) |
| `analyze.py` | Tables and charts for the report (`report/`) |
| `server.py` | **Backend**: web API that runs the repair loop and streams its steps |
| `frontend/` | **Frontend**: `index.html`, `app.js`, `style.css` |
| `ask.py` | "Ask the assistant" mode: runs `solve()` and describes the result |
| `data/sample_tasks.jsonl` | 9 small Qiskit tasks written for this project (works offline) |
| `download_dataset.py` | Downloads the real benchmark (151 tasks) |
| `run_experiments.sh` | All the runs for the report in one script |
| `tests/` | Tests for the harness itself (`python -m unittest discover tests`) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_dataset.py         # -> data/qiskit_humaneval.jsonl
python -m unittest discover tests  # should say OK
```

**Pick a model** (any one is enough; two makes a better report):

- **Ollama (free, local):** install from ollama.com, then `ollama pull qwen2.5-coder:7b`. Needs about 8 GB of RAM.
- **Groq (free tier, fast):** get a key at console.groq.com and run `export GROQ_API_KEY=...`
- **Gemini (free tier):** get a key from Google AI Studio and run `export GEMINI_API_KEY=...`

Model names in `llm.py` are only examples. Check your provider's current list.

## Running

```bash
# 0. check the environment: run the reference solutions (no LLM).
#    Some tasks may fail on your Qiskit version; analyze.py can exclude those.
python evaluate.py --provider canonical --max-attempts 1 --run canonical

# 1. a quick try on 10 tasks
python evaluate.py --provider ollama --model qwen2.5-coder:7b --limit 10

# 2. full runs (they resume if interrupted)
python evaluate.py --provider ollama --model qwen2.5-coder:7b --feedback full
python evaluate.py --provider ollama --model qwen2.5-coder:7b --feedback none

# 3. tables + charts
python analyze.py results/*.jsonl --valid results/canonical.jsonl --out report

# 4. the web tool (frontend + backend)
python server.py            # open http://localhost:8000
```

To test everything without any model or API key, use `--provider mock`. It fails once on purpose with
the removed `qiskit.execute` import, then returns the reference solution. Use it only to check the pipeline. **Don't report its numbers.**

## Experiments (the "variant" for your team)

| Run | Purpose |
|---|---|
| model A, `--feedback full` | main result: pass@1 vs pass with repair |
| model A, `--feedback minimal` | does the full traceback matter, or only the error line? |
| model A, `--feedback none` | is it the error message that helps, or only getting another try? |
| model B, `--feedback full` | does a bigger or smaller model benefit more from repair? |

**Fairness note for the report:** in `full` mode the traceback of a failed `assert` shows that line
of the hidden test, so the model learns something about the test. Discuss this. The `minimal` and
`none` runs show how much of the gain depends on it.

## Output

`report/summary.md` contains:
- **Summary:** pass@1, pass with repair, gain, % of first-try failures fixed, average attempts
- **Pass rate within k attempts:** the main chart (`pass_at_k.png`)
- **Fix rate by error type:** import/API errors (e.g. removed `execute`, old `Aer` imports), syntax,
  wrong result, other runtime errors (`first_try_errors.png`)
- **By difficulty:** basic / intermediate / difficult (`by_difficulty.png`)

## Safety

The code that runs is written by an LLM. It runs in a separate process with a timeout,
but that is **not** a security sandbox. Run the project in a virtual environment or a container.
