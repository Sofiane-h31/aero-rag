"""Run a set of sample questions and report answers + groundedness scores.

    python scripts/03_evaluate.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag import RAGPipeline
from src.evaluate import groundedness

SAMPLE_QUESTIONS = [
    "What are common causes of altitude deviations?",
    "How do crews handle an engine failure on departure?",
    "What role does fatigue play in reported incidents?",
    "What happens during a loss of cabin pressurization?",
    "Are there reports about miscommunication with ATC?",
]


def main():
    print("[eval] loading pipeline...")
    rag = RAGPipeline()

    scores = []
    for q in SAMPLE_QUESTIONS:
        r = rag.answer(q)
        g = groundedness(r.answer, r.contexts, embedder=rag.embedder)
        scores.append(g)
        print("\nQ:", q)
        print("A:", r.answer)
        print("Sources:", ", ".join(r.sources), f"| groundedness={g:.3f}")

    print(f"\n[eval] mean groundedness over {len(scores)} questions: "
          f"{sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
