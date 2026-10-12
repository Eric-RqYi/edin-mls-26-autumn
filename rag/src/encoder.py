from __future__ import annotations

from pathlib import Path
import numpy as np


class LocalEncoder:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if device is not None:
            kwargs["device"] = device

        self.model = SentenceTransformer(
            str(model_dir),
            local_files_only=True,
            **kwargs,
        )

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vec, dtype=np.float32)

    def encode_many(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        arr = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(arr, dtype=np.float32)
