"""Build the vector index: split reports into chunks, embed them, store in Chroma.

This is the "offline" half of RAG. You run it once; querying reuses the
persisted index.
"""
from __future__ import annotations

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

import config


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split a string into overlapping character windows.

    Overlap matters: it keeps a piece of information from being cut in half at
    a chunk boundary, which would make it unretrievable.
    """
    text = str(text)
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_index(df: pd.DataFrame, device: str = "cpu") -> int:
    """Chunk every report, embed the chunks, and store them in a Chroma collection.

    Returns the number of chunks indexed.
    """
    # 1. Split every report into chunks, keeping the source doc_id as metadata
    chunks, metadatas, ids = [], [], []
    for _, row in df.iterrows():
        for j, ch in enumerate(chunk_text(row["text"])):
            chunks.append(ch)
            metadatas.append({"doc_id": row["doc_id"]})
            ids.append(f"{row['doc_id']}-{j}")
    print(f"[ingest] {len(chunks)} chunks from {len(df)} reports")

    # 2. Embed the chunks (normalised -> cosine similarity == dot product)
    embedder = SentenceTransformer(config.EMBED_MODEL, device=device)
    embeddings = embedder.encode(
        chunks, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )

    # 3. Persist into a Chroma collection (reset if it already exists)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(config.COLLECTION)

    # Add in batches to stay within Chroma's per-call limits
    B = 2000
    for i in range(0, len(chunks), B):
        collection.add(
            documents=chunks[i:i + B],
            embeddings=[e.tolist() for e in embeddings[i:i + B]],
            metadatas=metadatas[i:i + B],
            ids=ids[i:i + B],
        )
    print(f"[ingest] index written to {config.CHROMA_DIR}")
    return len(chunks)
