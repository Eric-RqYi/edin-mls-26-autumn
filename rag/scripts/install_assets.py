"""Install the course data and enc_a model from an extracted asset bundle."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL_FILES=('config_sentence_transformers.json','config.json','course_encoder_metadata.json',
             'model.safetensors','modules.json','sentence_bert_config.json',
             'tokenizer_config.json','tokenizer.json','1_Pooling/config.json')

def plan(bundle):
    data=bundle/'data/rag_corpus'
    corpus=[json.loads(line) for line in (data/'corpus.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    ids={r['record_id'] for r in corpus}
    if len(corpus)!=24000 or len(ids)!=24000 or any(set(r)!={'record_id','text'} for r in corpus):
        raise ValueError('Expected the course corpus with 24,000 unique records')
    manifest=[json.loads(line) for line in (data/'student_audio_manifest.jsonl').read_text().splitlines() if line.strip()]
    if len(manifest)!=475 or len({r['query_id'] for r in manifest})!=475:
        raise ValueError('Expected 475 public audio queries')
    files=['data/rag_corpus/corpus.jsonl','data/rag_corpus/student_audio_manifest.jsonl']
    for split,count in (('practice',350),('dev',125)):
        items=[r for r in manifest if r['split']==split]
        if len(items)!=count:
            raise ValueError('Incorrect query split counts')
        qrels=data/f'qrels_{split}.tsv'
        with qrels.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f,delimiter='\t')
            if set(reader.fieldnames)!={'query_id','record_id','score'}:
                raise ValueError('Unexpected qrels fields')
            judgements=list(reader)
        if {r['query_id'] for r in judgements}!={r['query_id'] for r in items} or any(r['record_id'] not in ids for r in judgements):
            raise ValueError('Qrels IDs do not match the course data')
        files.append(f'data/rag_corpus/qrels_{split}.tsv')
        for row in items:
            rel=f"audio/{split}/{row['query_id']}.wav"
            if row['audio_file']!=rel or '/' in row['query_id'] or '\\' in row['query_id']:
                raise ValueError('Invalid manifest audio path')
            p=data/rel
            with wave.open(str(p),'rb') as w:
                if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(16000,1,2,'NONE'):
                    raise ValueError('Incorrect audio format')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Audio checksum mismatch')
            files.append('data/rag_corpus/'+rel)
    for alias in ('enc_a','enc_b'):
        p=data/f'embeddings/corpus_{alias}.npy'
        mapping=data/f'embeddings/corpus_{alias}_ids.json'
        vectors=np.load(p,mmap_mode='r',allow_pickle=False)
        mapped=json.loads(mapping.read_text())
        if vectors.shape!=(24000,384) or len(mapped)!=24000 or set(mapped)!=ids:
            raise ValueError(f'{alias} vectors and record IDs do not match')
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
            raise ValueError(f'{alias} vectors must be finite and normalised')
        files.extend([f'data/rag_corpus/embeddings/corpus_{alias}.npy',f'data/rag_corpus/embeddings/corpus_{alias}_ids.json'])
    files.extend('models/enc_a/'+name for name in MODEL_FILES)
    for rel in files:
        if not (bundle/rel).is_file():raise FileNotFoundError(bundle/rel)
    return files

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-dir',type=Path,required=True)
    args=ap.parse_args()
    bundle=args.from_dir.resolve()
    files=plan(bundle)
    # Check existing destinations before copying, so local changes remain visible.
    for rel in files:
        dest=ROOT/rel
        if dest.is_file() and dest.read_bytes()!=(bundle/rel).read_bytes():
            raise RuntimeError(f'{dest} differs from the bundle; move the old asset before reinstalling')
    copied=0
    for rel in files:
        dest=ROOT/rel
        if dest.exists():continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(bundle/rel,dest)
        copied+=1
    print(f'Course assets verified: {len(files)} files, {copied} installed')

if __name__=='__main__':
    main()
