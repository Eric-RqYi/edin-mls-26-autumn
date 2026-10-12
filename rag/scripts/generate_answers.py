"""Generate answer candidates using the audio-to-answer baseline."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Diagnostic subset only")
    args = ap.parse_args()
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--k and --limit must be positive")
    if args.output.exists():
        ap.error("Output exists; choose a new filename to preserve prior results")
    asr = ASRAdapter()
    asr.prepare()
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    pipeline = RAGPipeline(asr, LocalEncoder(ROOT / "models/enc_a", device=args.device),
        ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json"),
        data, SimpleContextBuilder(), ELMGenerator())
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for item in items:
            result = pipeline.answer_audio(data_dir / item["audio_file"], k=args.k)
            row = {"query_id": item["query_id"], "split": args.split, "audio_file": item["audio_file"],
                   "transcript": result["transcript"], "answer": result["answer"],
                   "retrieved_record_ids": result["retrieval"]["record_ids"], "timing": result["timing"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(item["query_id"])

if __name__ == "__main__":
    main()
