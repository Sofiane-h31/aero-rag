"""The RAG pipeline: retrieve relevant chunks, then generate a grounded answer.

This is the "online" half. `RAGPipeline` loads the embedder, the Chroma index
and the local LLM once, then answers questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

SYSTEM_PROMPT = (
    "You are an aviation-safety analyst. Answer the question using ONLY the "
    "context provided. If the context does not contain the answer, reply exactly: "
    "'I don't know based on the reports.' and nothing else. "
    "Do NOT list report IDs in your answer — they are shown separately. "
    "Write a single, concise paragraph."
)


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[str]
    contexts: list[str] = field(default_factory=list)


class RAGPipeline:
    """End-to-end RAG over the ASRS vector index."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Embedder (same model used at ingest time -> consistent vector space)
        self.embedder = SentenceTransformer(config.EMBED_MODEL, device=self.device)

        # Vector store
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.collection = client.get_collection(config.COLLECTION)

        # Local generator
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL, torch_dtype="auto", device_map="auto"
        )

    # --- retrieval ---------------------------------------------------------
    def retrieve(self, question: str, k: int = config.TOP_K):
        """Return the k most relevant chunks and their source doc_ids."""
        q_vec = self.embedder.encode(
            [question], normalize_embeddings=True
        )[0].tolist()
        res = self.collection.query(query_embeddings=[q_vec], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return docs, metas

    # --- generation --------------------------------------------------------
    def generate(self, question: str, contexts: list[str], metas: list[dict]) -> str:
        context_block = "\n\n".join(
            f"[{m['doc_id']}] {c}" for c, m in zip(contexts, metas)
        )
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,            # <-- renvoie un dict {input_ids, attention_mask}
        ).to(self.model.device)

        gen_kwargs = dict(max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
        if config.TEMPERATURE > 0:       # ne passe 'temperature' QUE si on échantillonne
            gen_kwargs.update(do_sample=True, temperature=config.TEMPERATURE)

        output = self.model.generate(**inputs, **gen_kwargs)   # <-- **inputs (nommé)

        # Decode only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True
        ).strip()

    # --- public API --------------------------------------------------------
    def answer(self, question: str, k: int = config.TOP_K) -> RAGResult:
        contexts, metas = self.retrieve(question, k)
        text = self.generate(question, contexts, metas)
        sources = sorted({m["doc_id"] for m in metas})
        return RAGResult(question, text, sources, contexts)
