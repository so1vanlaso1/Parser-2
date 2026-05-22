import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class StructuralRAG:
    """
    Structural memory retriever.

    Stores CNL -> AST example pairs and retrieves the most similar ones
    by cosine similarity over sentence embeddings.
    """

    def __init__(self, examples_path: str = "data/structural_examples.jsonl"):
        self.examples_path = Path(examples_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.items: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.load()

    def load(self):
        rows = []
        with self.examples_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        self.items = rows
        if rows:
            texts = [r["cnl"] for r in rows]
            self.embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        else:
            self.embeddings = None

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.embeddings is None or len(self.items) == 0:
            return []

        query_emb = self.embedder.encode([query], convert_to_numpy=True)[0]

        # Vectorized cosine similarity: single matrix multiply + norm division.
        query_norm = np.linalg.norm(query_emb)
        emb_norms = np.linalg.norm(self.embeddings, axis=1)
        denom = query_norm * emb_norms
        # Avoid division by zero.
        denom = np.where(denom == 0, 1.0, denom)
        scores = (self.embeddings @ query_emb) / denom

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.items[i] | {"score": float(scores[i])} for i in top_indices]

    def format_examples(self, query: str, top_k: int = 3) -> str:
        """Format top-k retrieved examples for injection into Stage 3 prompt."""
        matches = self.retrieve(query, top_k=top_k)

        blocks = []
        for m in matches:
            blocks.append(
                "CNL:\n"
                f"{m['cnl']}\n"
                "AST:\n"
                f"{json.dumps(m['ast'], ensure_ascii=False)}"
            )

        return "\n\n".join(blocks)
