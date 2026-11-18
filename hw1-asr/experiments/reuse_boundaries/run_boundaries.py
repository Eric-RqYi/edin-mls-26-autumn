#!/usr/bin/env python3
"""Directed shape validation and paired ASR, using frozen existing GPU tools."""
import argparse,csv,datetime as dt,hashlib,json,os,shlex,shutil,sys
from pathlib import Path
from boundary_spec import HW, FLAGS, CASE_FIELDS, ATTENTION_FIELDS, cases, variants
import run_experiment as runtime
HERE=Path(__file__).resolve().parent

def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','gemm','m1','attention','asr','evidence'),default='all')
    p.add_argument('--output',type=Path);p.add_argument('--history-root',type=Path,default=HW/'experiments')
    p.add_argument('--sm');p.add_argument('--dry-run',action='store_true');p.add_argument('--cuda-tmpdir',type=Path)
    for k,v in (('jobs',2),('warmup',5),('iterations',20),('repeats',7),('seed',21),('model-blocks',3),('model-runs',10),('model-warmup',3)):
        p.add_argument('--'+k,type=int,default=v)
    a=p.parse_args(argv)
    if a.seed!=21:p.error('The frozen validation design uses seed 21')
    if min(a.jobs,a.iterations,a.repeats,a.model_blocks,a.model_runs)<1 or min(a.warmup,a.model_warmup)<0:p.error('Invalid counts')
    if (a.sm and not a.sm.isdigit()) or (a.dry_run and not a.sm):p.error('Dry-run requires numeric --sm')
    return a

def manifest(part,out,a,gpu):
    vv=[dict(v,build=str(out/'build'/v['name']),library=str(out/'build'/v['name']/'libasr_kernels.so')) for v in variants(part)]
    return dict(status='running',part=part,stage='factorial' if part=='asr' else part,baseline=vv[0]['name'],variants=vv,
                gpu=gpu,cases=cases('gemm' if part=='attention' else part),attention_cases=cases('attention'),
                repeats=a.repeats,warmup=a.warmup,iterations=a.iterations,seed=a.seed,
                skip_model=part!='asr',model_blocks=a.model_blocks,model_runs=a.model_runs,model_warmup=a.model_warmup)

def steps(part,out,a,meta,sm):
    S=runtime.Step;src=out/'source';exp=src/'experiment'
    result=runtime.preflight_steps(out,sm,os.environ.get('NVCC_CCBIN','/usr/bin/g++'))
    for v in meta['variants']:
        cmd=['make',f'SM={sm}',f"BUILD={v['build']}",f'SRC={src}/csrc',f"NVCCFLAGS={v['flags']}"]
        result += [S(f"build/{v['name']}.log",[*cmd,f'-j{a.jobs}','all']),S(f"tests/{v['name']}.log",[*cmd,'test'])]
    libs=[f"{v['name']}={v['library']}" for v in meta['variants']]
    kind='attention' if part=='attention' else 'gemm';exe=out/'build'/f'benchmark_{kind}';cuda_root=Path(shutil.which('nvcc') or '/opt/cuda-12.8.0/bin/nvcc').resolve().parent.parent
    result.append(S('build/benchmark.log',['nvcc',*FLAGS.split(),f'-arch=sm_{sm}',f'-I{src}/csrc',f'-I{src}/tests',str(exp/f'benchmark_{kind}.cu'),'-o',str(exe),'-lcublas','-ldl','-Xlinker','-rpath','-Xlinker',str(cuda_root/'lib64')]))
    for tool in ('memcheck','racecheck','synccheck'):
        result.append(S(f'sanitizer/{tool}.log',['compute-sanitizer','--error-exitcode','99','--tool',tool,str(exe),'--cases',str(out/'cases.csv'),'--small','--correctness-only',*libs]))
        if part in ('gemm','asr'):
            result.append(S(f'sanitizer/hybrid_{tool}.log',['compute-sanitizer','--error-exitcode','99','--tool',tool,str(exe),'--cases',str(out/'hybrid_sanitizer.csv'),'--correctness-only',*libs]))
    result.append(S('micro.log',[str(exe),'--cases',str(out/'cases.csv'),'--csv',str(out/('attention_micro.csv' if kind=='attention' else 'micro.csv')),'--warmup',str(a.warmup),'--iterations',str(a.iterations),'--repeats',str(a.repeats),'--seed',str(a.seed),*libs]))
    if part=='asr':
        for b in range(1,a.model_blocks+1):result.append(S(f'model/block{b}.log',[sys.executable,'-u',str(exp/'model_runner.py'),'--hw-root',str(src),'--manifest',str(out/'metadata.json'),'--block',str(b),'--output',str(out/'model'/f'block{b}.json')]))
    return result

def snapshot(out):
    runtime.snapshot(out)
    dst=out/'source'/'experiment'
    shutil.copy2(HW/'experiments/task4_attention/benchmark_attention.cu',dst)
    shutil.copy2(HW/'experiments/task4_attention/attention_spec.py',dst)
    shutil.copy2(HW/'experiments/task4_attention/attention_results.py',dst)
    for f in HERE.glob('*'):
        if f.is_file():shutil.copy2(f,dst)
    (out/'source_sha256.json').write_text(json.dumps({str(p.relative_to(out/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/'source').rglob('*')) if p.is_file()},indent=2)+'\n')

def main(argv=None):
    a=parse(argv);out=(a.output or HERE/'results'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    parts=('gemm','m1','attention','asr') if a.stage=='all' else (a.stage,)
    if a.dry_run:
        for part in parts:
            if part=='evidence':print('Read-only evidence import:',a.history_root);continue
            for s in steps(part,out/part,a,manifest(part,out/part,a,{}),a.sm):print(f'[{part}/{s.log}] {shlex.join(s.argv)}')
        return 0
    if out.exists():raise FileExistsError(out)
    if any(c.isspace() for c in str(out)):raise ValueError('Output path cannot contain whitespace')
    out.mkdir(parents=True);env={k:os.environ.get(k) for k in ('TMPDIR','NVCC_CCBIN')}
    try:
        from boundary_analysis import analyze,import_history
        import_history(a.history_root,out)
        if a.stage=='evidence':return 0
        gpu=runtime.gpu_info()
        if a.sm and a.sm!=gpu['sm']:raise ValueError('Requested architecture differs from visible GPU')
        for part in parts:
            job=out/part;job.mkdir();meta=manifest(part,job,a,gpu);snapshot(job)
            meta['build_environment']=runtime.prepare_build_environment(job,a.cuda_tmpdir)
            (job/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
            with (job/'cases.csv').open('w',newline='') as f:
                w=csv.writer(f,lineterminator='\n');w.writerow(ATTENTION_FIELDS if part=='attention' else CASE_FIELDS);w.writerows(cases(part))
            with (job/'hybrid_sanitizer.csv').open('w',newline='') as f:
                w=csv.writer(f,lineterminator='\n');w.writerow(CASE_FIELDS);w.writerow(('hybrid_m2',2,512,2048,0,0))
            for s in [runtime.Step('gpu.log',['nvidia-smi','-L']),runtime.Step('nvcc.log',['nvcc','--version']),*steps(part,job,a,meta,gpu['sm'])]:runtime.run_step(s,job)
            analyze(job)
            meta['status']='complete';(job/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        (out/'analysis.md').write_text('# Reuse validation\n\nNo samples removed. See history_manifest.json for existing evidence coverage.\n\n'+ '\n\n'.join((out/part/'analysis.md').read_text() for part in parts))
        return 0
    except BaseException as e:
        (out/'FAILED.txt').write_text(str(e)+'\n')
        for job in out.iterdir():
            m=job/'metadata.json'
            if job.is_dir() and m.exists():
                d=json.loads(m.read_text())
                if d['status']=='running':d.update(status='failed',error=str(e));m.write_text(json.dumps(d,indent=2)+'\n')
        raise
    finally:
        for k,v in env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
if __name__=='__main__':raise SystemExit(main())
