#!/usr/bin/env bash
# All runs for the report. Edit MODEL_A / MODEL_B to the models you are using.
set -e
MODEL_A="qwen2.5-coder:7b"            # small local model (Ollama)
MODEL_B="llama-3.3-70b-versatile"     # larger model (Groq, needs GROQ_API_KEY)

# 0. sanity check: which tasks pass with the reference solutions on this machine?
python evaluate.py --provider canonical --max-attempts 1 --run canonical

# 1. main result: small model, with and without useful error feedback
python evaluate.py --provider ollama --model "$MODEL_A" --feedback full
python evaluate.py --provider ollama --model "$MODEL_A" --feedback minimal
python evaluate.py --provider ollama --model "$MODEL_A" --feedback none

# 2. second model, full feedback
python evaluate.py --provider groq --model "$MODEL_B" --feedback full

# 3. tables + charts (only tasks whose reference solution works here)
python analyze.py results/*.jsonl --valid results/canonical.jsonl --out report
