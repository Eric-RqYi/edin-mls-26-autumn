from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
import numpy as np

from src.asr_adapter import ASRAdapter
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.retriever import ExactDenseRetriever


def load_cache(path: Path):
    cache = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["query_id"]] = row["transcript"]
    return cache


def append_cache(path: Path, qid: str, transcript: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"query_id": qid, "transcript": transcript},
            ensure_ascii=False,
        ) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--transcript-cache", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--k-max", type=int, default=100)
    args = ap.parse_args()
    if args.k_max < 100:
        ap.error("--k-max must be at least 100 to report Recall@100")

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    qrels = data.load_qrels(args.split)

    asr = ASRAdapter()
    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings" / "corpus_enc_a.npy",
        data_dir / "embeddings" / "corpus_enc_a_ids.json",
    )

    cache = load_cache(args.transcript_cache) if args.transcript_cache else {}

    rows = []
    for i, item in enumerate(items, start=1):
        qid = item["query_id"]
        audio = data_dir / item["audio_file"]

        if qid in cache:
            transcript = cache[qid]
        else:
            transcript = asr.transcribe(audio)
            if args.transcript_cache:
                append_cache(args.transcript_cache, qid, transcript)

        qvec = encoder.encode(transcript)
        ret = retriever.retrieve(qvec, k=args.k_max)
        ranked = ret["record_ids"]
        rel = qrels[qid]

        row = {
            "query_id": qid,
            "Recall@5": recall_at_k(ranked, rel, 5),
            "Recall@10": recall_at_k(ranked, rel, 10),
            "Recall@20": recall_at_k(ranked, rel, 20),
            "Recall@100": recall_at_k(ranked, rel, 100),
            "MRR@10": mrr_at_k(ranked, rel, 10),
            "NDCG@10": ndcg_at_k(ranked, rel, 10),
        }
        rows.append(row)
        print(f"[{i:>3}/{len(items)}] {qid} R@10={row['Recall@10']:.3f}")

    keys = ["Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10"]
    print("\nAggregate:")
    for key in keys:
        print(f"  {key:<10} {np.mean([r[key] for r in rows]):.6f}")


if __name__ == "__main__":
    main()
