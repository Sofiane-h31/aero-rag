"""Ask a question against the indexed aviation reports.

    python scripts/02_ask.py "What are common causes of altitude deviations?"

With no argument, it runs a short interactive loop.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag import RAGPipeline
from src.evaluate import groundedness


def show(result, rag):
    print("\n" + "=" * 70)
    print("Q:", result.question)
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    print("Sources:", ", ".join(result.sources))
    score = groundedness(result.answer, result.contexts, embedder=rag.embedder)
    flag = "  <-- low, review" if score < 0.4 else ""
    print(f"Groundedness: {score:.3f}{flag}")
    print("=" * 70 + "\n")


def main():
    print("[ask] loading pipeline (this downloads the model on first run)...")
    rag = RAGPipeline()

    if len(sys.argv) > 1:
        show(rag.answer(" ".join(sys.argv[1:])), rag)
        return

    print("[ask] interactive mode — type a question, or 'quit' to exit.")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"quit", "exit", "q", ""}:
            break
        show(rag.answer(q), rag)


if __name__ == "__main__":
    main()
