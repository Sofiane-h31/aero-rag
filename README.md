# ✈️ Aero-RAG — Question Answering over Aviation Safety Reports

A clean, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that answers
natural-language questions about real aviation incidents — **grounded in the source
reports, with citations** — running fully locally.

Built on NASA's [ASRS](https://asrs.arc.nasa.gov/) (Aviation Safety Reporting System)
free-text incident reports.

## Why this project

Large language models are fluent but ungrounded: they predict the *most plausible*
answer, not the *most truthful* one. RAG fixes this by retrieving real documents and
forcing the model to answer from them — and to admit when the answer isn't there.
This repo is a minimal, readable implementation of that idea on a real, messy,
domain-specific corpus.

## Example

```
Q: What role does fatigue play in reported incidents?
A: Fatigue contributes to impaired decision-making and a higher risk of errors.
   In one report a co-pilot felt tired but chose to fly a third leg, and insidious
   fatigue degraded their judgement; another links fatigue to a busy schedule and
   poor sleep affecting cognitive performance.
Sources: RPT-00913, RPT-01434, RPT-03817, RPT-03846   | groundedness = 0.85
```

The guardrail in action (out-of-scope retrieval):

```
Q: How do crews handle an engine failure on departure?
A: The retrieved reports describe engine failures in cruise, not on departure,
   so: I don't know based on the reports.
```

That second answer is the point: when the sources don't contain the answer, the
system says so instead of inventing one.

## How it works

```
                    ┌─────────────── offline (build once) ───────────────┐
   ASRS reports ──▶ chunk ──▶ embed (bge-small) ──▶ ChromaDB vector store
                    └────────────────────────────────────────────────────┘

                    ┌─────────────────── online (per query) ─────────────┐
   question ──▶ embed ──▶ retrieve top-k ──▶ build prompt ──▶ LLM ──▶ grounded answer + sources
                    └────────────────────────────────────────────────────┘
```

- **Retrieval** — the question is embedded and matched against report chunks by
  cosine similarity (ChromaDB).
- **Generation** — the retrieved chunks are injected into the prompt of a local
  instruction-tuned LLM, instructed to answer *only* from that context and to say
  "I don't know" otherwise.
- **Reliability** — a groundedness score (answer ↔ retrieved-context similarity)
  flags answers that may not be supported by the sources. On the sample question
  set, mean groundedness ≈ **0.83**.

## Stack

| Component | Choice |
|-----------|--------|
| Embeddings | `sentence-transformers` · `BAAI/bge-small-en-v1.5` |
| Vector store | `ChromaDB` (persistent) |
| Generator | `transformers` · `Qwen/Qwen2.5-7B-Instruct` (local) |
| Reliability | cosine-similarity groundedness check |

Everything is pip-installable and runs locally — no external API keys. A GPU is
recommended for the 7B generator (≈16 GB VRAM); see *Tuning* for lighter options.

## Project layout

```
aero-rag/
├── config.py            # all tunable settings in one place
├── src/
│   ├── data.py          # load + clean the ASRS reports
│   ├── ingest.py        # chunk, embed, build the Chroma index
│   ├── rag.py           # retrieval + generation pipeline
│   └── evaluate.py      # groundedness check
├── scripts/
│   ├── 01_build_index.py
│   ├── 02_ask.py
│   └── 03_evaluate.py
└── notebooks/
    └── demo.ipynb       # narrated walkthrough
```

## Quickstart

```bash
pip install -r requirements.txt

# Optional: keep model/data caches off your home quota
export HF_HOME=/path/with/space/hf_cache

# 1. Build the vector index (run once)
python scripts/01_build_index.py

# 2. Ask a question
python scripts/02_ask.py "What happens during a loss of cabin pressurization?"

# 3. Or run the sample evaluation
python scripts/03_evaluate.py
```

`02_ask.py` with no argument starts an interactive prompt.

## Tuning

All knobs live in `config.py`:

- **Lower VRAM** → `LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"`, or
  `"google/flan-t5-base"` to run on CPU.
- **Better retrieval** → raise `TOP_K`, or use `BAAI/bge-base-en-v1.5`.
- **Bigger corpus** → set `SAMPLE_SIZE = None` to index all reports.

## Limitations & next steps

- The generator is small; answers are only as good as what retrieval surfaces.
- Groundedness is a heuristic, not a guarantee — a human stays in the loop for
  anything safety-critical.
- Natural extensions: a reranker on top of retrieval, an automated faithfulness
  evaluation (LLM-as-judge), and a small web UI.

## Data & license

NASA ASRS reports are public and de-identified. This project is for learning and
demonstration purposes.
