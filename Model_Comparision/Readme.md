
# Ollama Compare UI

Compare any two Ollama models (e.g., `llama3.2:1b` vs your finetuned GGUF) with a tiny Streamlit UI.

## Prereqs
- Docker + Docker Compose
- Your finetuned `.gguf` file in `./models/gguf_model/`

## Quickstart
```bash
# 1) Clone files and place your GGUF
mkdir -p models/gguf_model eval_data app && cp your.gguf models/gguf_model/ggml-model-q4_k_m.gguf

# 2) Build & start
docker compose up -d --build

# 3) Create Ollama model from the GGUF
docker exec -it ollama ollama create finetuned-gguf -f /models/Modelfile

# 4) Open UI
open http://localhost:8501
# Pick: Model A = llama3.2:1b, Model B = finetuned-gguf
```

## Notes
- Metrics: BLEU‑4, ROUGE‑L (if you provide a reference), plus latency and tokens/sec.
- Perplexity is not exposed in Ollama HTTP as a single op; if you need it, run a separate HF/llama.cpp eval harness.
- To use Open WebUI instead of Streamlit, spin it up alongside Ollama and set its connection URL to `http://ollama:11434` in Docker.
*/
