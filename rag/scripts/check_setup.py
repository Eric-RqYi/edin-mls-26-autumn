from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import numpy as np
from src.data import CourseData
from src.asr_adapter import ASRAdapter

DATA_DIR = ROOT / "data" / "rag_corpus"
MODEL_DIR = ROOT / "models" / "enc_a"


def check(label, fn):
    try:
        value = fn()
        print(f"[PASS] {label}" + (f": {value}" if value is not None else ""))
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        return False


def main():
    ok = True

    ok &= check(
        "corpus.jsonl",
        lambda: f"{len(CourseData(DATA_DIR).records)} records",
    )

    ok &= check(
        "student audio manifest",
        lambda: f"{len(CourseData(DATA_DIR).load_audio_manifest())} audio items",
    )

    emb_path = DATA_DIR / "embeddings" / "corpus_enc_a.npy"
    ids_path = DATA_DIR / "embeddings" / "corpus_enc_a_ids.json"

    def embedding_check():
        x = np.load(emb_path, mmap_mode="r")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if x.shape[0] != len(ids):
            raise RuntimeError("embedding rows != ID count")
        return f"shape={tuple(x.shape)}"

    ok &= check("corpus embeddings", embedding_check)

    def model_check():
        if not MODEL_DIR.exists():
            raise RuntimeError(f"missing {MODEL_DIR}")
        required = ["config.json", "modules.json", "tokenizer.json", "model.safetensors"]
        missing = [name for name in required if not (MODEL_DIR / name).is_file()]
        if missing:
            raise RuntimeError(f"model payload missing: {', '.join(missing)}; install the course assets")
        return "model payload present"

    ok &= check("enc_a model directory", model_check)

    def elm_check():
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not (os.getenv("ELM_API_KEY") or os.getenv("OPENAI_API_KEY")):
            raise RuntimeError("ELM_API_KEY is not configured")
        return "API key configured"

    ok &= check("ELM configuration", elm_check)

    def asr_check():
        asr = ASRAdapter()
        if not (asr.asr_root / "glm_asr_cuda/model.py").is_file():
            raise RuntimeError("Set RAG_ASR_ROOT to the hw1-asr directory")
        return f"source found at {asr.asr_root}; inference checked by running an audio query"
    ok &= check("ASR source path", asr_check)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
