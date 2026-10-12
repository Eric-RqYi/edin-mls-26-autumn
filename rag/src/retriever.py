from __future__ import annotations

import time
from pathlib import Path
import json
import numpy as np


class ExactDenseRetriever:
    """Transparent exact cosine-similarity baseline over precomputed corpus vectors."""

    def __init__(self, embeddings_path: str | Path, ids_path: str | Path):
        self.embeddings = np.load(embeddings_path)
        self.record_ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))

        if len(self.record_ids) != self.embeddings.shape[0]:
            raise ValueError("embedding row count does not match ID list")

        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("corpus embeddings are expected to be L2-normalised")

    def retrieve(self, query_vector: np.ndarray, k: int = 5):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.size != self.embeddings.shape[1] or not np.isfinite(q).all():
            raise ValueError("query vector must be finite and match embedding dimension")
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            raise ValueError("zero-norm query vector")
        q = q / q_norm

        k = min(k, len(self.record_ids))

        start = time.perf_counter()
        scores = self.embeddings @ q
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {
            "record_ids": [self.record_ids[int(i)] for i in idx],
            "scores": [float(scores[int(i)]) for i in idx],
            "latency_ms": elapsed_ms,
        }
