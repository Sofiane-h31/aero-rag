"""Central configuration for the aero-RAG project.

Everything tunable lives here so the scripts stay clean and you can adapt the
pipeline (model, chunk size, retrieval depth) without touching the logic.
"""
from pathlib import Path

# --- Data -------------------------------------------------------------------
DATASET = "elihoole/asrs-aviation-reports"   # NASA ASRS reports on the HF hub
TEXT_COLUMN = "Report 1_Narrative"           # the free-text narrative column
SAMPLE_SIZE = 4000                           # subsample for fast iteration (None = all)

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 600                             # characters per chunk
CHUNK_OVERLAP = 100                          # overlap so info isn't cut at a boundary

# --- Models -----------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"       # sentence-embedding model (fast, strong)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"       # local generator (ungated, ~16GB on GPU)
# Lighter fallbacks if VRAM is tight: "Qwen/Qwen2.5-3B-Instruct" or "google/flan-t5-base"

# --- Vector store -----------------------------------------------------------
CHROMA_DIR = str(Path(__file__).parent / "chroma_index")
COLLECTION = "asrs_reports"

# --- Retrieval / generation -------------------------------------------------
TOP_K = 4                                    # chunks retrieved per question
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0                            # 0 = deterministic, reduces hallucination
