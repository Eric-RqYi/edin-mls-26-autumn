#!/usr/bin/env python3
"""Read-only audit and export of A/B results. No inference or network requests."""
import argparse,csv,hashlib,json,math,random,re,shutil,statistics,subprocess,sys,tempfile
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def jl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write(p,rr):
    if not rr:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def snapshot_audit(root):
    hs=json.loads((root/'source_sha256.json').read_text())
    assert all((root/'source'/n).is_file() and sha(root/'source'/n)==h for n,h in hs.items()),root
    m=json.loads((root/'metadata.json').read_text());assert m['status']=='complete'
    for v in m['variants']:
        log=(root/'tests'/f"{v['name']}.log").read_text()
        assert log.count('ALL PASS')==4 and not re.search(r'\bFAIL\b',log)
        build=(root/'build'/f"{v['name']}.log").read_text()
        assert all(flag in build for flag in v['flags'].split())
    logs=list((root/'sanitizer').glob('*.log'))
    for p in logs:
        s=p.read_text();assert ('0 hazards displayed (0 errors, 0 warnings)' if 'racecheck' in p.name else 'ERROR SUMMARY: 0 errors') in s
    return dict(source_files=len(hs),coursework_runs=4*len(m['variants']),sanitizer_logs=len(logs),gpu=m['gpu'])
def export_a(root):
    # Each independent process imports this part's own frozen validators.
    code='''import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);sys.path.insert(0,str(p/'source/experiment'));m=json.loads((p/'metadata.json').read_text())\nfrom results import micro_summary,model_summaries\nfrom attention_results import attention_micro_summary,validate_attention_profiles,factorial_effects\nif m['part']=='attention':d=dict(micro=attention_micro_summary(p,m))\nelse:d=dict(micro=micro_summary(p,m))\nif m['part']=='asr':\n samples,model,blocks,ops,linear=model_summaries(p,m);att,checks=validate_attention_profiles(p,m)\n d.update(samples=samples,model=model,blocks=blocks,operations=ops,linear=linear,attention=att,boundaries=checks,effects=factorial_effects(samples,m))\nprint(json.dumps(d))'''
    return json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(root)],text=True))
def export_b(root):
    # Frozen summarizer writes only into a disposable copy, never into the saved run.
    with tempfile.TemporaryDirectory() as td:
        target=Path(td)
        for name in ('metadata.json','sample_manifest.json','request_plan.json','raw.jsonl','warmup.jsonl','diagnostics.jsonl','qrels_dev.tsv'):
            shutil.copy2(root/name,target/name)
        shutil.copytree(root/'source',target/'source',ignore=shutil.ignore_patterns('__pycache__'))
        subprocess.run([sys.executable,'-B',str(root/'source/experiment/summarize_e2e.py'),str(target)],check=True)
        assert json.loads((target/'summary.json').read_text())['status']=='complete'
        result={name:read(target/name) for name in ('per_query.csv','summary.csv','output_consistency.csv','retrieval_metrics.csv','retrieval_summary.csv')}
        for name,rr in result.items():write(HERE/('b_'+name),rr)
        for name in ('summary.csv','retrieval_summary.csv'):
            before=read(root/name)
            assert len(before)==len(result[name])
            for a,b in zip(before,result[name]):
                for k in a:
                    try:assert math.isclose(float(a[k]),float(b[k]),rel_tol=1e-11,abs_tol=1e-7)
                    except ValueError:assert a[k]==b[k]
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--a-root',type=Path,default=REPO/'hw1-asr/experiments/reuse_boundaries/20261118T225304Z');ap.add_argument('--b-root',type=Path,default=REPO.parent/'task-b-rag-e2e/rag/extensions/reuse_e2e/results/20261118T235107Z');ap.add_argument('--plots',action='store_true');a=ap.parse_args()
    A,B=a.a_root.resolve(),a.b_root.resolve()
    input_hashes={str(p):sha(p) for root in (A,B) for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.log','.py','.cu','.cuh','.h')}
    result={};audit={}
    for part in ('gemm','m1','attention','asr'):
        audit['a_'+part]=snapshot_audit(A/part);result[part]=export_a(A/part)
        write(HERE/f'a_{part}_micro.csv',result[part]['micro'])
        if (A/part/'shape_comparison.csv').exists():shutil.copy2(A/part/'shape_comparison.csv',HERE/f'a_{part}_shape_details.csv')
    old_root=REPO/'hw1-asr/experiments/task3_gemm/results/20261117T192802Z-615737'
    historical_system=[]
    if old_root.exists():
        audit['historical_gemm']=snapshot_audit(old_root)
        code="import sys,json;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/'source/experiment'));from results import model_summaries;print(json.dumps(model_summaries(p,json.loads((p/'metadata.json').read_text()))[1]))"
        historical_system=json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(old_root)],text=True))
        write(HERE/'historical_gemm_asr.csv',historical_system)
        input_hashes.update({str(p):sha(p) for p in old_root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log','.py','.cu','.cuh','.h')})
    audit['b']=snapshot_audit(B)
    bm=json.loads((B/'metadata.json').read_text());cohort=json.loads((B/'sample_manifest.json').read_text());raw=jl(B/'raw.jsonl');warm=jl(B/'warmup.jsonl');diagnostics=jl(B/'diagnostics.jsonl')
    assert len(raw)==180 and len(warm)==4 and len(diagnostics)==6
    assert all(r['status']=='ok' for r in raw+warm)
    for v in bm['variants']:
        assert sha(B/'build'/v['name']/'libasr_kernels.so')==v['library_sha256']
        assert bm['worker_ready']['library_hashes'][v['name']]==v['library_sha256']
        assert all(r['asr']['library_sha256']==v['library_sha256'] for r in raw if r['variant']==v['name'])
    assert sha(B/'qrels_dev.tsv')==bm['qrels_snapshot_sha256']
    # Reconstruct count-balanced duration strata and selection from the frozen frame.
    frame=sorted(cohort['valid_frame'],key=lambda r:(r['duration_s'],r['query_id']));rng=random.Random(21);selection=[]
    for i,label in enumerate(('short','medium','long')):
        layer=frame[len(frame)*i//3:len(frame)*(i+1)//3]
        selection.extend((r['query_id'],label) for r in rng.sample(layer,10))
    assert selection==[(r['query_id'],r['stratum']) for r in cohort['items']]
    assert len(frame)==125 and not cohort['limited_subset'] and not cohort['invalid_files']
    assert all(r['split']=='dev' for r in frame) and all(r['split']=='practice' for r in cohort['warmup'])
    assert not {r['query_id'] for r in cohort['warmup']} & {r['query_id'] for r in cohort['items']}
    assert sorted(r['generation']['attempt'] for r in raw+warm)==list(range(1,185))
    assert all(r['generation']['retries']==0 for r in raw+warm)
    assert all(r['asr']['eos'] and not r['asr']['truncated'] for r in raw)
    # Confirm A and B use the same numerical kernels/model, despite different baselines.
    ha=json.loads((A/'asr/source_sha256.json').read_text());hb=json.loads((B/'source_sha256.json').read_text())
    common=[k for k in ha if k in hb and k.startswith(('csrc/','glm_asr_cuda/','tests/'))]
    assert all(ha[k]==hb[k] for k in common)
    audit['cross_run_common_sources']=len(common)
    br=export_b(B)
    for rr in br['output_consistency.csv']:
        assert all(rr[k]=='True' for k in ('tokens_equal','transcript_equal','embedding_close','scores_close','rank_ids_equal','context_equal','prompt_equal'))
    # ASR attribution uses per-operation medians; phase totals are summed within blocks first.
    model={r['variant']:r for r in result['asr']['model']};op={(r['variant'],r['operation']):r['median_gpu_event_ms'] for r in result['asr']['operations']}
    asr=[]
    for n in ('base','hybrid','attention','combined'):
        r=model[n];saved=model['base']['median_ms']-r['median_ms']
        asr.append(dict(variant=n,median_ms=r['median_ms'],mean_ms=r['mean_ms'],iqr_ms=r['iqr_ms'],max_ms=r['max_ms'],saved_ms=saved,latency_reduction_pct=100*(1-r['median_ms']/model['base']['median_ms']),paired_speedup=r['median_block_paired_speedup'],min_block_speedup=r['min_block_paired_speedup'],max_block_speedup=r['max_block_paired_speedup'],linear_profile_saved_ms=op['base','linear']-op[n,'linear'],attention_profile_saved_ms=op['base','attention']-op[n,'attention']))
    write(HERE/'a_asr_comparison.csv',asr);write(HERE/'a_factorial_blocks.csv',result['asr']['effects'])
    assert len(result['asr']['samples'])==150 and all(abs(r['difference_ms'])==0 for r in result['asr']['boundaries'])
    phase=defaultdict(list)
    for b in (1,2,3):
        x=defaultdict(float);d=json.loads((A/'asr/model'/f'block{b}.json').read_text())
        for r in d['attention_profiles']:x[r['variant'],r['phase']]+=r['gpu_event_ms']
        for key,t in x.items():phase[key].append(t)
    phase_rows=[dict(variant=n,phase=p,median_profile_ms=statistics.median(v)) for (n,p),v in phase.items()]
    write(HERE/'a_attention_phases.csv',phase_rows)
    # Compiler evidence and context remain machine-readable, without new hardware-counter claims.
    for part in ('gemm','attention'):
        shutil.copy2(A/part/'compiler_resources.csv',HERE/f'a_{part}_compiler_resources.csv')
    # Archive the already-validated historical shape table, with its provenance manifest.
    shutil.copy2(A/'history_gemm.csv',HERE/'a_historical_gemm.csv');shutil.copy2(A/'history_manifest.json',HERE/'a_history_manifest.json')
    stage=[]
    for n in ('original','final'):
        rr=[r for r in raw if r['variant']==n]
        means={k:statistics.mean(r['timing'][k] for r in rr) for k in ('asr_roundtrip_ms','embedding_ms','retrieval_ms','context_ms','generation_ms','end_to_end_ms','unattributed_ms')}
        means['core_asr_ms']=statistics.mean(r['asr']['core_asr_ms'] for r in rr)
        means['asr_other_ms']=means['asr_roundtrip_ms']-means['core_asr_ms']
        means['local_rag_ms']=means['embedding_ms']+means['retrieval_ms']+means['context_ms']+means['unattributed_ms']
        assert math.isclose(means['core_asr_ms']+means['asr_other_ms']+means['local_rag_ms']+means['generation_ms'],means['end_to_end_ms'],abs_tol=1e-7)
        stage.append(dict(variant=n,requests=len(rr),**means))
    write(HERE/'b_stage_means.csv',stage)
    sample=[]
    for q in cohort['items']:
        rr=[r for r in raw if r['query_id']==q['query_id']]
        sample.append(dict(query_id=q['query_id'],stratum=q['stratum'],duration_s=q['duration_s'],tokens=rr[0]['asr']['tokens'],requests=len(rr),eos_all=all(r['asr']['eos'] for r in rr)))
    write(HERE/'b_cohort.csv',sample)
    pq=[dict(r) for r in br['per_query.csv']]
    regression=[r for r in pq if r['metric']=='end_to_end_ms' and float(r['paired_saved_median_ms'])<0]
    local=[]
    for r in raw:
        t=r['timing'];noncore=t['asr_roundtrip_ms']-r['asr']['core_asr_ms']+t['embedding_ms']+t['retrieval_ms']+t['context_ms']+t['unattributed_ms']
        local.append(dict(query_id=r['query_id'],round=r['round'],variant=r['variant'],stratum=r['stratum'],non_core_local_ms=noncore,core_asr_ms=r['asr']['core_asr_ms'],audio_load_ms=r['asr']['audio_load_ms'],preprocess_ms=r['asr']['preprocess_ms'],embedding_ms=t['embedding_ms'],generation_ms=t['generation_ms'],end_to_end_ms=t['end_to_end_ms']))
    write(HERE/'b_largest_local_delays.csv',sorted(local,key=lambda r:-r['non_core_local_ms'])[:20])
    # Sensitivity check: preserve ten query clusters per duration stratum in resampling.
    sensitivity=[]
    for metric in ('core_asr_ms','end_to_end_ms'):
        layers=[[float(r['paired_saved_median_ms']) for r in pq if r['metric']==metric and r['stratum']==s] for s in ('short','medium','long')]
        rng=random.Random(21);values=sorted(statistics.median([v for group in layers for v in rng.choices(group,k=10)]) for _ in range(2000))
        def pct(p):
            z=(len(values)-1)*p;i=int(z);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(z-i)
        sensitivity.append(dict(metric=metric,method='stratified_query_cluster_bootstrap',seed=21,resamples=2000,ci_low_ms=pct(.025),ci_high_ms=pct(.975)))
    write(HERE/'b_ci_sensitivity.csv',sensitivity)
    # Request-level statistical units are retained, never promoted to independent audio samples.
    metrics=dict(historical_gemm_asr=historical_system,a_run=str(A),b_run=str(B),audit=audit,a_asr=asr,a_effects=result['asr']['effects'],a_attention_phases=phase_rows,
                 b_summary=br['summary.csv'],b_stage_means=stage,b_retrieval=br['retrieval_summary.csv'],b_regressed_queries=regression,
                 b_cohort=dict(selected=30,valid_frame=125,invalid_files=0,formal_requests=180,warmup_requests=4,diagnostic_generations=6,truncated=0,eos=180),
                 b_startup=dict(rag_startup_ms=bm['rag_startup_ms'],asr_worker_roundtrip_ms=bm['worker_startup_roundtrip_ms']),
                 limitations=['One allocation per experiment, same GPU model with different MIG UUIDs.','WAV/corpus/encoder assets are not copied here; their recorded hashes cannot all be independently recalculated locally.','No hardware counters, WER/CER or answer judging.','Cached token counts and server request IDs were not exposed.'])
    dump(HERE/'metrics.json',metrics);dump(HERE/'input_sha256.json',input_hashes)
    assert all(sha(Path(k))==h for k,h in input_hashes.items()),'Original input was modified'
    if a.plots:
        from make_figures import render
        render(HERE)
    print(json.dumps(dict(a_asr=asr,b_summary=[r for r in br['summary.csv'] if r['group']=='all'],audited_inputs=len(input_hashes),regressed_queries=len(regression)),indent=2))
if __name__=='__main__':main()
