#!/usr/bin/env python3
"""Frozen multi-audio experiment. Default mode prints a plan and makes no API call."""
import argparse,datetime as dt,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
from design import freeze,schedule,variants,sha,canonical
HERE=Path(__file__).resolve().parent;RAG=HERE.parents[1];REPO=RAG.parent;HW=REPO/'hw1-asr'
sys.path.insert(0,str(HW/'experiments/task3_gemm'))
import run_experiment as runtime

def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','freeze','check','run'),default='plan')
    p.add_argument('--data-dir',type=Path,default=RAG/'data/rag_corpus');p.add_argument('--encoder-dir',type=Path,default=RAG/'models/enc_a')
    p.add_argument('--env-file',type=Path,default=RAG/'.env');p.add_argument('--sample-manifest',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--sm');p.add_argument('--cuda-tmpdir',type=Path)
    p.add_argument('--asr-python',default=os.environ.get('RAG_ASR_PYTHON'));p.add_argument('--allow-subset',action='store_true')
    a=p.parse_args(argv)
    if a.sm and not a.sm.isdigit():p.error('--sm must be numeric')
    return a

def build_steps(out,vv,sm):
    S=runtime.Step;source=out/'source'
    steps=runtime.preflight_steps(out,sm,os.environ.get('NVCC_CCBIN','/usr/bin/g++'))
    for v in vv:
        cmd=['make',f'SM={sm}',f"BUILD={v['build']}",f'SRC={source}/csrc',f"NVCCFLAGS={v['flags']}"]
        steps.extend([S(f"build/{v['name']}.log",[*cmd,'-j2','all']),S(f"tests/{v['name']}.log",[*cmd,'test'])])
    exe=out/'build/benchmark_gemm';cuda_root=Path(shutil.which('nvcc') or '/opt/cuda-12.8.0/bin/nvcc').resolve().parent.parent
    steps.append(S('build/benchmark.log',['nvcc','-O3','-std=c++17',f'-arch=sm_{sm}',f'-I{source}/csrc',f'-I{source}/tests',str(source/'experiment/benchmark_gemm.cu'),'-o',str(exe),'-lcublas','-ldl','-Xlinker','-rpath','-Xlinker',str(cuda_root/'lib64')]))
    for tool in ('memcheck','racecheck','synccheck'):
        steps.append(S(f'sanitizer/hybrid_{tool}.log',['compute-sanitizer','--error-exitcode','99','--tool',tool,str(exe),'--cases',str(out/'hybrid_cases.csv'),'--correctness-only',*[f"{v['name']}={v['library']}" for v in vv]]))
    return steps

def snapshot(out):
    runtime.snapshot(out);source=out/'source'
    shutil.copytree(RAG/'src',source/'rag/src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(RAG/'src/asr_worker.py',source/'experiment/asr_worker_audio.py')
    for f in HERE.iterdir():
        if f.is_file():shutil.copy2(f,source/'experiment'/f.name)
    hashes={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')

def gpu_info(python):
    code="""import torch,json
if not torch.cuda.is_available(): raise RuntimeError('No CUDA GPU')
p=torch.cuda.get_device_properties(0)
print(json.dumps(dict(name=p.name,sm=str(p.major)+str(p.minor),multiprocessors=p.multi_processor_count,memory_bytes=p.total_memory,torch_version=torch.__version__,torch_cuda=torch.version.cuda)))"""
    return json.loads(subprocess.check_output([python,'-c',code],text=True))

def load_sample(a):
    if not a.sample_manifest:return freeze(a.data_dir,a.allow_subset)
    d=json.loads(a.sample_manifest.read_text());expected=d.pop('manifest_sha256')
    if hashlib.sha256(canonical(d).encode()).hexdigest()!=expected:raise ValueError('Sample manifest hash mismatch')
    d['manifest_sha256']=expected
    if d['seed']!=21 or d['split']!='dev' or len(d['items'])>30 or len(d['warmup'])!=2:raise ValueError('Invalid frozen cohort')
    if len({r['query_id'] for r in d['items']})!=len(d['items']):raise ValueError('Duplicate cohort ID')
    if d['limited_subset'] and not a.allow_subset:raise ValueError('Limited cohort requires --allow-subset')
    for r in d['items']+d['warmup']:
        if sha(r['audio_path'])!=r['audio_sha256']:raise ValueError('Frozen audio changed')
    return d

def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def main(argv=None):
    a=parse(argv);out=(a.output or HERE/'results'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    vv=[dict(v,build=str(out/'build'/v['name']),library=str(out/'build'/v['name']/'libasr_kernels.so')) for v in variants()]
    if a.mode=='plan':
        print('Frozen design: up to 30 dev queries, 3 paired rounds, 180 formal + 4 warmup API calls; no retries or judge. No commands executed.')
        print('Required assets:',a.data_dir,'and',a.encoder_dir,'; ASR interpreter:',a.asr_python or '<set RAG_ASR_PYTHON>')
        for s in build_steps(out,vv,a.sm or '90'):print(f'[{s.log}] {shlex.join(s.argv)}')
        return 0
    if out.exists():raise FileExistsError(out)
    if any(c.isspace() for c in str(out)):raise ValueError('Output path must not contain whitespace')
    out.mkdir(parents=True);env={k:os.environ.get(k) for k in ('TMPDIR','NVCC_CCBIN')};worker=None;meta=None;generator=None
    try:
        cohort=load_sample(a);plan=schedule(cohort)
        (out/'sample_manifest.json').write_text(json.dumps(cohort,indent=2)+'\n')
        (out/'request_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        if a.mode=='freeze':return 0
        if not a.asr_python or not Path(a.asr_python).is_file():raise RuntimeError('Set RAG_ASR_PYTHON to mls-asr Python')
        required=[a.data_dir/n for n in ('corpus.jsonl','qrels_dev.tsv','embeddings/corpus_enc_a.npy','embeddings/corpus_enc_a_ids.json')]+[a.encoder_dir/'modules.json']
        missing=[str(p) for p in required if not p.is_file()]
        if missing:raise FileNotFoundError('Missing assets: '+', '.join(missing))
        for name in ('nvcc','make','nvidia-smi','compute-sanitizer'):
            if shutil.which(name) is None:raise RuntimeError('Missing '+name)
        from dotenv import load_dotenv
        load_dotenv(a.env_file)
        key=os.environ.get('ELM_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not key:raise RuntimeError('ELM_API_KEY is missing')
        gpu=gpu_info(a.asr_python)
        if a.sm and a.sm!=gpu['sm']:raise ValueError('Requested SM differs from GPU')
        snapshot(out)
        meta=dict(schema=1,status='building',mode=a.mode,variants=vv,gpu=gpu,k=5,encoder_device='cpu',temperature=.1,
                  seed=21,rounds=3,diagnostics_expected=2*len({r['stratum'] for r in cohort['items']}),sample_manifest_sha256=cohort['manifest_sha256'],plan_sha256=hashlib.sha256(canonical(plan).encode()).hexdigest(),
                  api_budget=2 if a.mode=='check' else len(plan)+4,sdk_max_retries=0,
                  endpoint=os.environ.get('ELM_BASE_URL',os.environ.get('OPENAI_BASE_URL','https://elm.edina.ac.uk/api/v1')),
                  elm_model=os.environ.get('ELM_MODEL','utter-project/EuroLLM-22B-Instruct-2512'),asr_python=a.asr_python,
                  inputs={str(p.resolve()):sha(p) for p in required},
                  encoder_files={str(p.relative_to(a.encoder_dir)):sha(p) for p in sorted(a.encoder_dir.rglob('*')) if p.is_file()},
                  build_environment=runtime.prepare_build_environment(out,a.cuda_tmpdir))
        def save():
            if generator is not None:meta['api_client_calls']=generator.attempts
            (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        shutil.copy2(a.data_dir/'qrels_dev.tsv',out/'qrels_dev.tsv');meta['qrels_snapshot_sha256']=sha(out/'qrels_dev.tsv');save();(out/'hybrid_cases.csv').write_text('case,M,N,K,bias,act\nhybrid_m2,2,512,2048,0,0\n')
        for s in [runtime.Step('gpu.log',['nvidia-smi','-L']),runtime.Step('nvcc.log',['nvcc','--version']),*build_steps(out,vv,gpu['sm'])]:runtime.run_step(s,out)
        for v in vv:v['library_sha256']=sha(v['library'])
        save();sys.path.insert(0,str(out/'source/rag'))
        from src.encoder import LocalEncoder
        from src.retriever import ExactDenseRetriever
        from src.context_builder import SimpleContextBuilder
        from src.data import CourseData
        from openai import OpenAI
        from client import WorkerClient,MeteredGenerator,answer
        t=time.perf_counter();data=CourseData(a.data_dir);encoder=LocalEncoder(a.encoder_dir,device='cpu');retriever=ExactDenseRetriever(a.data_dir/'embeddings/corpus_enc_a.npy',a.data_dir/'embeddings/corpus_enc_a_ids.json');builder=SimpleContextBuilder()
        client=OpenAI(api_key=key,base_url=meta['endpoint'],timeout=180,max_retries=0)
        generator=MeteredGenerator(client,meta['elm_model'],meta['api_budget'])
        meta['rag_startup_ms']=1000*(time.perf_counter()-t)
        t=time.perf_counter();worker=WorkerClient(a.asr_python,out/'source',out/'metadata.json',out/'worker.log');meta['worker_startup_roundtrip_ms']=1000*(time.perf_counter()-t);meta['worker_ready']=worker.ready
        if worker.hashes!={v['name']:v['library_sha256'] for v in vv}:raise RuntimeError('Loaded libraries do not match manifest')
        meta['status']='running';save()
        warm=[dict(item,variant=name,round=0,order=i,stratum='practice') for i,item in enumerate(cohort['warmup']) for name in ('original','final')]
        if a.mode=='check':warm=[warm[0],warm[3]]
        for item in warm:
            row=answer(item,worker,encoder,retriever,data,builder,generator);append(out/'warmup.jsonl',row);save()
            if row['status']!='ok':raise RuntimeError('Practice connectivity/warmup failed; no formal run started')
        if a.mode=='check':
            meta['status']='check_complete';save();return 0
        for item in plan:
            row=answer(item,worker,encoder,retriever,data,builder,generator)
            append(out/'raw.jsonl',row);save()
        # Separate diagnostic generations: no ELM calls, not part of normal latency.
        for label in ('short','medium','long'):
            representatives=[r for r in cohort['items'] if r['stratum']==label]
            if not representatives:continue
            for name in ('original','final'):
                item=dict(representatives[0],variant=name)
                append(out/'diagnostics.jsonl',dict(query_id=item['query_id'],stratum=label,**worker.request(item,profile=True)))
        meta['status']='collected';save()
        from summarize_e2e import summarize
        report=summarize(out)
        meta['status']=report['status'];save()
        return 0 if report['status']=='complete' else 2
    except BaseException as e:
        (out/'FAILED.txt').write_text(type(e).__name__+': '+str(e)+'\n')
        if meta is not None:
            meta['status']='failed';meta['error_type']=type(e).__name__
            if generator is not None:meta['api_client_calls']=generator.attempts
            (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        raise
    finally:
        if worker is not None:worker.close()
        for k,v in env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
if __name__=='__main__':raise SystemExit(main())
