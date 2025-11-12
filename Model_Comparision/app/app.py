# app.py (modified)
import os
import time
import json
import re
import requests
import streamlit as st
from eval import bleu_4, rouge_l

OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

st.set_page_config(page_title="Ollama Model Compare", layout="wide")
st.title("🔍 Ollama Model Compare")
st.caption("Side-by-side prompts, speed, and simple quality metrics (BLEU-4, ROUGE-L)")

@st.cache_data(ttl=10)
def list_models():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        names = [m["name"] for m in data.get("models", [])]
        return sorted(names)
    except Exception as e:
        st.error(f"Could not list models from Ollama at {OLLAMA}: {e}")
        return []

def normalize_text_for_metrics(s: str) -> str:
    """Lowercase, remove excessive whitespace and leading/trailing spaces."""
    if s is None:
        return ""
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def strip_prompt_from_response(prompt: str, response: str) -> str:
    """
    Remove the prompt if the model echoed it at the start of the response.
    This is a heuristic: if response starts with the prompt (or starts with a large prefix),
    remove that prefix so BLEU/ROUGE compare only the generated completion.
    """
    if not prompt or not response:
        return response or ""
    p = prompt.strip()
    r = response.strip()
    # If response begins exactly with the prompt (or prompt plus punctuation/newlines), strip it.
    if r.startswith(p):
        stripped = r[len(p):].lstrip(" \n:.-–—")  # remove leading separators
        if stripped:
            return stripped
        # fallback to removing prompt even if empty
        return ""
    # If the prompt is long and the first N chars of response match, remove that prefix
    # (handles small formatting differences)
    min_match_chars = min(len(p), 200)
    if min_match_chars >= 20:
        # compare first 80% of short prefix
        prefix_len = int(min_match_chars * 0.8)
        if prefix_len > 0 and r[:prefix_len] == p[:prefix_len]:
            stripped = r[prefix_len:].lstrip(" \n:.-–—")
            return stripped
    return r

models = list_models()
if not models:
    st.warning("No models found. Make sure Ollama is running and OLLAMA_BASE_URL is correct.")
colA, colB = st.columns(2)
with colA:
    model_a = st.selectbox("Model A", models, index=0 if models else None, key="model_a")
with colB:
    model_b = st.selectbox("Model B", models, index=1 if len(models) > 1 else 0, key="model_b")

prompt = st.text_area("Prompt", value="You are a helpful assistant. Explain what fine-tuning is in 3 bullet points.", height=120)
max_tokens = st.slider("Max new tokens", 16, 512, 128, step=16)
temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
reference = st.text_area("Reference (optional, used for BLEU/ROUGE)", value="")
run = st.button("Run Comparison", type="primary")

headers = {"Content-Type": "application/json"}

def call_ollama_generate(model_name: str, prompt_text: str, temperature: float, top_p: float, max_tokens: int, timeout=600):
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/generate", headers=headers, data=json.dumps(payload), timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    out = r.json()
    # Ollama response shape differs by version; handle safely
    text = out.get("response") or out.get("text") or ""
    text = text.strip()
    # rough token estimate (eval_count may be present)
    gen_tokens = out.get("eval_count") or len(text.split())
    return text, dt*1000.0, gen_tokens / max(1e-6, dt), gen_tokens, out

def safe_generate(model_name, prompt_text):
    try:
        return call_ollama_generate(model_name, prompt_text, temperature, top_p, max_tokens)
    except requests.exceptions.RequestException as e:
        st.error(f"Network error calling Ollama for model {model_name}: {e}")
        return "", float("nan"), float("nan"), 0, {}
    except Exception as e:
        st.error(f"Error generating from model {model_name}: {e}")
        return "", float("nan"), float("nan"), 0, {}

if run:
    if not models:
        st.error("No models available to query.")
    elif model_a == model_b:
        st.warning("You selected the same model for both A and B — choose two different models to compare.")
    else:
        with st.spinner("Querying models via Ollama..."):
            text_a_raw, ttft_a, tps_a, tokens_a, raw_a = safe_generate(model_a, prompt)
            text_b_raw, ttft_b, tps_b, tokens_b, raw_b = safe_generate(model_b, prompt)

        # Strip echoed prompt if present and normalize for display
        text_a = strip_prompt_from_response(prompt, text_a_raw)
        text_b = strip_prompt_from_response(prompt, text_b_raw)

        # Display outputs side-by-side
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"A · {model_a}")
            st.write(text_a or "_(no response)_")
            st.caption(f"TTFT ~ {ttft_a:.0f} ms · tokens/s ~ {tps_a:.1f} · tokens ~ {tokens_a}")
            st.text("Raw response (truncated):")
            st.code(json.dumps(raw_a)[:1000] + ("..." if len(json.dumps(raw_a)) > 1000 else ""))
        with c2:
            st.subheader(f"B · {model_b}")
            st.write(text_b or "_(no response)_")
            st.caption(f"TTFT ~ {ttft_b:.0f} ms · tokens/s ~ {tps_b:.1f} · tokens ~ {tokens_b}")
            st.text("Raw response (truncated):")
            st.code(json.dumps(raw_b)[:1000] + ("..." if len(json.dumps(raw_b)) > 1000 else ""))

        # If reference provided, compute BLEU & ROUGE (after normalization)
        bleu_a = bleu_b = rouge_a = rouge_b = None
        if reference.strip():
            # Normalize both reference and candidates to improve overlap robustness
            ref_norm = normalize_text_for_metrics(reference)
            cand_a_norm = normalize_text_for_metrics(text_a)
            cand_b_norm = normalize_text_for_metrics(text_b)

            try:
                # If the candidate ends up empty after stripping, warn
                if not cand_a_norm:
                    st.warning("Model A's generated text is empty after stripping prompt — BLEU/ROUGE will be 0.")
                if not cand_b_norm:
                    st.warning("Model B's generated text is empty after stripping prompt — BLEU/ROUGE will be 0.")

                bleu_a = bleu_4(cand_a_norm, ref_norm)
                bleu_b = bleu_4(cand_b_norm, ref_norm)
                rouge_a = rouge_l(cand_a_norm, ref_norm)
                rouge_b = rouge_l(cand_b_norm, ref_norm)
            except Exception as e:
                st.error(f"Error computing BLEU/ROUGE: {e}")

        # Summary metrics table
        summary = {
            "model": [model_a, model_b],
            "TTFT_ms": [round(ttft_a, 1), round(ttft_b, 1)],
            "tokens_per_s": [round(tps_a, 1), round(tps_b, 1)],
            "gen_tokens": [tokens_a, tokens_b],
            "BLEU-4": [round(bleu_a, 4) if bleu_a is not None else "-", round(bleu_b, 4) if bleu_b is not None else "-"],
            "ROUGE-L": [round(rouge_a, 4) if rouge_a is not None else "-", round(rouge_b, 4) if rouge_b is not None else "-"],
        }
        st.markdown("### 📊 Metrics")
        st.dataframe(summary, use_container_width=True)

        # Derived win/tie
        if bleu_a is not None and rouge_a is not None:
            score_a = (bleu_a or 0) + (rouge_a or 0)
            score_b = (bleu_b or 0) + (rouge_b or 0)
            if score_a > score_b:
                winner_text = f"🏆 Model A ({model_a}) wins ({score_a:.4f} vs {score_b:.4f})"
            elif score_b > score_a:
                winner_text = f"🏆 Model B ({model_b}) wins ({score_b:.4f} vs {score_a:.4f})"
            else:
                winner_text = f"⚖️ Tie ({score_a:.4f} vs {score_b:.4f})"
            st.success(winner_text)

        # Charts for latency and throughput
        st.markdown("### ⚡ Performance charts")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.write("TTFT (ms)")
            st.bar_chart({"model": [model_a, model_b], "TTFT_ms": [ttft_a, ttft_b]})
        with chart_col2:
            st.write("Tokens / sec")
            st.bar_chart({"model": [model_a, model_b], "tokens_per_s": [tps_a, tps_b]})

        # (download button removed as requested)
