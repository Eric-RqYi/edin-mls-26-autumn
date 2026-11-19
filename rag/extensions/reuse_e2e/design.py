"""Predeclared sampling, paired request order, and complete kernel configurations."""
import hashlib,json,random,wave
from pathlib import Path
SEED=21
NAMES=('original','final')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def variants():
    rows=[]
    for name,impl,m1,hybrid,warps in (('original',0,0,0,4),('final',1,1,1,16)):
        rows.append(dict(name=name,flags=(f'-O3 -std=c++17 -lineinfo -diag-suppress 177 -Xptxas=-v '
            f'-DASR_TASK3_GEMM_IMPL={impl} -DASR_TASK3_TILE=16 -DASR_TASK3_RM={4 if impl else 2} '
            f'-DASR_TASK3_RN={4 if impl else 2} -DASR_TASK3_M1_ENABLED={m1} -DASR_TASK3_M1_WARPS=4 '
            f'-DASR_TASK3_HYBRID_KV={hybrid} -DASR_ATTN_WARPS={warps} -DASR_ATTN_ONLINE_MODE=0')))
    return rows

def inspect_audio(data_dir,item):
    p=(data_dir/item['audio_file']).resolve()
    with wave.open(str(p),'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(16000,1,2,'NONE'):raise ValueError('Expected mono PCM16 16kHz')
        n=w.getnframes()
        if n<1 or len(w.readframes(n))!=2*n:raise ValueError('Empty or truncated WAV')
    return dict(query_id=str(item['query_id']),audio_path=str(p),duration_s=n/16000,split=item['split'],audio_sha256=sha(p))

def freeze(data_dir,allow_subset=False):
    data_dir=Path(data_dir).resolve();frame=[];practice=[];invalid=[];seen=set()
    manifest=data_dir/'student_audio_manifest.jsonl'
    for line in manifest.read_text().splitlines():
        if not line.strip():continue
        item=json.loads(line);key=str(item['query_id'])
        if key in seen:raise ValueError('Duplicate query ID')
        seen.add(key)
        if item['split'] not in ('dev','practice'):continue
        try:row=inspect_audio(data_dir,item)
        except (OSError,ValueError,EOFError,wave.Error) as e:
            invalid.append(dict(query_id=key,split=item['split'],reason=type(e).__name__));continue
        (frame if item['split']=='dev' else practice).append(row)
    frame.sort(key=lambda r:(r['duration_s'],r['query_id']))
    rng=random.Random(SEED);selected=[]
    for i,label in enumerate(('short','medium','long')):
        layer=frame[len(frame)*i//3:len(frame)*(i+1)//3]
        selected.extend(dict(r,stratum=label) for r in rng.sample(layer,min(10,len(layer))))
    limited=len(selected)<30
    if not selected or (limited and not allow_subset):raise ValueError(f'Valid dev frame has {len(frame)} files; need 30 or explicit --allow-subset')
    if len(practice)<2:raise ValueError('Need two valid practice warmup files')
    warm=random.Random(SEED).sample(sorted(practice,key=lambda r:r['query_id']),2)
    payload=dict(seed=SEED,split='dev',target=30,limited_subset=limited,items=selected,warmup=warm,
                 valid_frame=frame,invalid_files=invalid,source_manifest_sha256=sha(manifest))
    return dict(payload,manifest_sha256=hashlib.sha256(canonical(payload).encode()).hexdigest())

def schedule(manifest):
    rng=random.Random(SEED);plan=[]
    for repeat in range(1,4):
        items=list(manifest['items']);rng.shuffle(items)
        for item in items:
            names=list(NAMES);rng.shuffle(names)
            for order,name in enumerate(names):plan.append(dict(query_id=item['query_id'],round=repeat,variant=name,order=order,stratum=item['stratum'],audio_path=item['audio_path']))
    assert len(plan)<=180
    return plan
