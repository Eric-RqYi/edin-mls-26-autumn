"""Query-clustered statistics and input-consistency audit of real pipeline records."""
import csv,hashlib,json,math,random,statistics,sys
from collections import defaultdict
from pathlib import Path
from design import canonical,schedule,variants

def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def write(path,data,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)
def percentile(x,p):
    x=sorted(x);v=(len(x)-1)*p;i=int(v);j=min(i+1,len(x)-1);return x[i]+(x[j]-x[i])*(v-i)
def close(a,b):return len(a)==len(b) and all(math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,rel_tol=1e-5,abs_tol=1e-6) for x,y in zip(a,b))
def ci(values):
    rng=random.Random(21);est=[statistics.median(rng.choices(values,k=len(values))) for _ in range(2000)]
    return percentile(est,.025),percentile(est,.975)

def summarize(out):
    out=Path(out)
    for name in ('analysis.md','summary.json','summary.csv','per_query.csv','output_consistency.csv','retrieval_metrics.csv','retrieval_summary.csv','stage_means.png','query_savings.png','diagnostic_shapes.csv'):(out/name).unlink(missing_ok=True)
    meta=json.loads((out/'metadata.json').read_text());manifest=json.loads((out/'sample_manifest.json').read_text())
    if meta.get('schema')!=1 or len(meta['variants'])!=2:raise ValueError('Unsupported result schema')
    for v,expected in zip(meta['variants'],variants()):
        if v['name']!=expected['name'] or v['flags']!=expected['flags']:raise ValueError('Incorrect variant configuration')
    payload=dict(manifest);expected=payload.pop('manifest_sha256')
    if hashlib.sha256(canonical(payload).encode()).hexdigest()!=expected or expected!=meta['sample_manifest_sha256']:raise ValueError('Sample manifest changed')
    plan=schedule(manifest)
    if hashlib.sha256(canonical(plan).encode()).hexdigest()!=meta['plan_sha256']:raise ValueError('Request plan changed')
    data=rows(out/'raw.jsonl');by={};allowed={(r['query_id'],r['round'],r['variant']):r for r in plan}
    library_hashes={v['name']:v['library_sha256'] for v in meta['variants']}
    for r in data:
        key=r['query_id'],r['round'],r['variant']
        if key not in allowed or key in by:raise ValueError('Unknown/duplicate request')
        if any(r[f]!=allowed[key][f] for f in ('order','stratum','audio_path')):raise ValueError('Request order/input differs')
        if r['status'] not in ('ok','api_failed','failed'):raise ValueError('Unknown status')
        for field,t in r['timing'].items():
            if not math.isfinite(t) or (t<0 and field!='unattributed_ms'):raise ValueError('Invalid timing')
        t=r['timing']
        if t.get('unattributed_ms',0)<-1e-5 or not math.isclose(sum(v for k,v in t.items() if k!='end_to_end_ms'),t['end_to_end_ms'],abs_tol=1e-5):raise ValueError('Stage boundary mismatch')
        if 'asr' in r:
            a=r['asr']
            if a['variant']!=r['variant'] or a['library_sha256']!=library_hashes[r['variant']]:raise ValueError('Wrong ASR library')
            if len(a['token_ids'])!=a['tokens'] or not a['tokens'] or not math.isfinite(a['core_asr_ms']) or a['core_asr_ms']<=0:raise ValueError('Invalid ASR record')
        if r['status']=='ok' and ('generation' not in r or r['generation']['status']!='ok'):raise ValueError('Missing successful generation')
        by[key]=r
    warm=rows(out/'warmup.jsonl') if (out/'warmup.jsonl').exists() else []
    attempts=[r['generation']['attempt'] for r in data+warm if 'generation' in r]
    if sorted(attempts)!=list(range(1,len(attempts)+1)) or len(attempts)>meta['api_budget'] or any(r['generation']['retries'] for r in data+warm if 'generation' in r):raise ValueError('API attempt/budget mismatch')
    if meta.get('api_client_calls')!=len(attempts):raise ValueError('Unaccounted API attempt')
    consistency=[];per_query=[];retrieval=[]
    # Use unchanged coursework metrics, with public qrels copied before inference.
    sys.path.insert(0,str(out/'source/rag'))
    from src.metrics import recall_at_k,mrr_at_k,ndcg_at_k
    qrels=defaultdict(dict)
    if meta.get('qrels_snapshot_sha256') and hashlib.sha256((out/'qrels_dev.tsv').read_bytes()).hexdigest()!=meta['qrels_snapshot_sha256']:raise ValueError('Qrels snapshot changed')
    with (out/'qrels_dev.tsv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if float(r.get('score',1))>0:qrels[str(r['query_id'])][str(r['record_id'])]=float(r.get('score',1))
    for item in manifest['items']:
        qid=item['query_id'];records=[by.get((qid,i,n)) for i in range(1,4) for n in ('original','final')]
        usable=[r for r in records if r and all(k in r for k in ('asr','query_vector','retrieval','contexts','prompt'))]
        flags={k:len(usable)==6 for k in ('tokens_equal','transcript_equal','embedding_close','scores_close','rank_ids_equal','context_equal','prompt_equal')}
        if usable:
            ref=usable[0]
            for r in usable[1:]:
                flags['tokens_equal'] &= r['asr']['token_ids']==ref['asr']['token_ids']
                flags['transcript_equal'] &= r['asr']['transcript']==ref['asr']['transcript']
                flags['embedding_close'] &= close(r['query_vector'],ref['query_vector'])
                flags['scores_close'] &= close(r['retrieval']['scores'],ref['retrieval']['scores'])
                flags['rank_ids_equal'] &= r['retrieval']['record_ids']==ref['retrieval']['record_ids']
                flags['context_equal'] &= r['contexts']==ref['contexts']
                flags['prompt_equal'] &= r['prompt']==ref['prompt']
        consistency.append(dict(query_id=qid,stratum=item['stratum'],input_records=len(usable),**flags,all_inputs_consistent=all(flags.values())))
        for name in ('original','final'):
            rr=[r for r in usable if r['variant']==name]
            if rr and qid in qrels:
                retrieval.append(dict(query_id=qid,variant=name,observations=len(rr),
                    recall_at_5=statistics.mean(recall_at_k(r['retrieval']['record_ids'],qrels[qid],5) for r in rr),
                    mrr_at_5=statistics.mean(mrr_at_k(r['retrieval']['record_ids'],qrels[qid],5) for r in rr),
                    ndcg_at_5=statistics.mean(ndcg_at_k(r['retrieval']['record_ids'],qrels[qid],5) for r in rr)))
        for metric in ('core_asr_ms','end_to_end_ms'):
            pairs=[]
            for i in range(1,4):
                a,b=[by.get((qid,i,n)) for n in ('original','final')]
                if not a or not b:continue
                if metric=='core_asr_ms' and 'asr' in a and 'asr' in b:pairs.append((a['asr'][metric],b['asr'][metric]))
                if metric=='end_to_end_ms' and a['status']==b['status']=='ok':pairs.append((a['timing'][metric],b['timing'][metric]))
            per_query.append(dict(query_id=qid,stratum=item['stratum'],duration_s=item['duration_s'],metric=metric,complete_pairs=len(pairs),
                baseline_median_ms=statistics.median(a for a,b in pairs) if pairs else '',final_median_ms=statistics.median(b for a,b in pairs) if pairs else '',
                paired_saved_median_ms=statistics.median(a-b for a,b in pairs) if pairs else '',paired_speedup_median=statistics.median(a/b for a,b in pairs) if pairs else '',
                baseline_tokens_median=statistics.median(r['asr']['tokens'] for r in usable if r['variant']=='original') if any(r['variant']=='original' for r in usable) else '',
                final_tokens_median=statistics.median(r['asr']['tokens'] for r in usable if r['variant']=='final') if any(r['variant']=='final' for r in usable) else '',
                truncated_observations=sum(r['asr'].get('truncated',False) for r in usable)))
    summary=[]
    for group in ('all','short','medium','long'):
        for metric in ('core_asr_ms','end_to_end_ms'):
            rr=[r for r in per_query if r['metric']==metric and r['complete_pairs']==3 and (group=='all' or r['stratum']==group)]
            if not rr:continue
            savings=[r['paired_saved_median_ms'] for r in rr];lo,hi=ci(savings)
            summary.append(dict(group=group,metric=metric,queries=len(rr),baseline_median_ms=statistics.median(r['baseline_median_ms'] for r in rr),
                final_median_ms=statistics.median(r['final_median_ms'] for r in rr),median_saved_ms=statistics.median(savings),mean_saved_ms=statistics.mean(savings),
                iqr_saved_ms=percentile(savings,.75)-percentile(savings,.25),paired_speedup=statistics.median(r['paired_speedup_median'] for r in rr),
                query_bootstrap_ci_low_ms=lo,query_bootstrap_ci_high_ms=hi))
    fields=['group','metric','queries','baseline_median_ms','final_median_ms','median_saved_ms','mean_saved_ms','iqr_saved_ms','paired_speedup','query_bootstrap_ci_low_ms','query_bootstrap_ci_high_ms']
    write(out/'per_query.csv',per_query);write(out/'summary.csv',summary,fields);write(out/'output_consistency.csv',consistency)
    write(out/'retrieval_metrics.csv',retrieval,['query_id','variant','observations','recall_at_5','mrr_at_5','ndcg_at_5'])
    retrieval_summary=[]
    for name in ('original','final'):
        rr=[r for r in retrieval if r['variant']==name and r['observations']==3]
        if rr:retrieval_summary.append(dict(variant=name,queries=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in ('recall_at_5','mrr_at_5','ndcg_at_5')}))
    write(out/'retrieval_summary.csv',retrieval_summary,['variant','queries','recall_at_5','mrr_at_5','ndcg_at_5'])
    diagnostic_rows=[];diagnostics=[]
    if (out/'diagnostics.jsonl').exists():
        diagnostics=rows(out/'diagnostics.jsonl')
        for record in diagnostics:
            for shape in record.get('attention_shapes',[]):diagnostic_rows.append(dict(query_id=record['query_id'],stratum=record['stratum'],variant=record['variant'],**shape))
    if diagnostic_rows:write(out/'diagnostic_shapes.csv',diagnostic_rows)
    valid=len(diagnostics)==meta.get('diagnostics_expected',len(diagnostics)) and meta.get('status')!='failed' and len(data)==len(plan) and all(r['status']=='ok' for r in data)
    status='complete' if valid and all(r['all_inputs_consistent'] for r in consistency) else 'incomplete_or_mismatch'
    result=dict(status=status,expected_requests=len(plan),recorded_requests=len(data),failed_requests=sum(r['status']!='ok' for r in data),
                api_client_calls=len(attempts),diagnostic_generations=len(diagnostics),limited_subset=manifest['limited_subset'],qrels_query_coverage=len({r['query_id'] for r in retrieval}),
                consistency_rates={k:sum(bool(r[k]) for r in consistency)/len(consistency) for k in ('tokens_equal','transcript_equal','embedding_close','scores_close','rank_ids_equal','context_equal','prompt_equal')})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Multi-audio warm-inference results','',f"Status: {status}; queries={len(manifest['items'])}; formal requests={len(data)}/{len(plan)}; client API calls={len(attempts)}/{meta['api_budget']}.",'',
           '| Group | Metric | Complete queries | Median saving ms | Paired speedup | Query-bootstrap 95% CI of saving ms |','|---|---|---:|---:|---:|---|']
    lines += [f"| {r['group']} | {r['metric']} | {r['queries']} | {r['median_saved_ms']:.3f} | {r['paired_speedup']:.4f}x | [{r['query_bootstrap_ci_low_ms']:.3f}, {r['query_bootstrap_ci_high_ms']:.3f}] |" for r in summary]
    lines += ['',f'Input consistency rates: {result["consistency_rates"]}.','',
        'Query is the statistical unit: each query first reduces three paired rounds. Incomplete queries remain in raw/per-query files and are excluded only from complete-pair aggregate estimates, with coverage reported. All slow successful samples are retained. Startup and separate diagnostic generations are excluded from inference timing.',
        'The measured parent end-to-end clock covers the real audio request, ASR/IPC, encoding, retrieval, context construction and live API response. It is not reconstructed from stage medians. Core ASR timing is a nested interval and must not be added again to ASR roundtrip.',
        'No answer judge or human transcription reference was used: no WER/CER or answer-quality score is claimed. Answers may differ for identical prompts. Corpus/qrels retrieval metrics concern k=5 only. Service retries/cache not exposed by the endpoint cannot be inferred from local records.']
    if diagnostic_rows:
        lines+=['','Observed attention lengths by stratum (separate diagnostic generations):']
        for label in ('short','medium','long'):
            for phase in ('encoder','decoder'):
                lengths=sorted({(r['seq_q'],r['seq_k']) for r in diagnostic_rows if r['stratum']==label and r['phase']==phase})
                lines.append(f'- {label} {phase}: {lengths}')
        lines+=['Audio seconds do not imply a different padded encoder shape. Per-query token counts and truncations are saved in per_query.csv for interpreting decoder workload.']
    e=next((r for r in summary if r['group']=='all' and r['metric']=='end_to_end_ms'),None)
    if e and e['query_bootstrap_ci_low_ms']<=0<=e['query_bootstrap_ci_high_ms']:lines+=['','The end-to-end saving interval includes zero: full-system benefit is not resolved from API variability in this sample.']
    if status!='complete':lines+=['','Incomplete/mismatched inputs: do not publish a claim that outputs/quality were preserved. See raw.jsonl and output_consistency.csv.']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n');plot(out,data,per_query)
    return result

def plot(out,data,per_query):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    valid={r['query_id'] for r in per_query if r['metric']=='end_to_end_ms' and r['complete_pairs']==3}
    if not valid:return
    fig,ax=plt.subplots(figsize=(7,4));bottom=[0.,0.]
    for stage in ('asr_roundtrip_ms','embedding_ms','retrieval_ms','context_ms','generation_ms','unattributed_ms'):
        heights=[statistics.mean(r['timing'][stage] for r in data if r['query_id'] in valid and r['variant']==n) for n in ('original','final')]
        ax.bar(('original','final'),heights,bottom=bottom,label=stage);bottom=[x+y for x,y in zip(bottom,heights)]
    ax.set_ylabel('Mean measured request latency (ms)');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'stage_means.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for label in ('short','medium','long'):
        rr=sorted((r for r in per_query if r['metric']=='end_to_end_ms' and r['complete_pairs']==3 and r['stratum']==label),key=lambda r:r['duration_s'])
        ax.scatter([r['duration_s'] for r in rr],[r['paired_saved_median_ms'] for r in rr],label=label)
    ax.axhline(0,color='grey');ax.set_xlabel('Audio duration (s)');ax.set_ylabel('Query median paired end-to-end saving (ms)');ax.legend();fig.tight_layout();fig.savefig(out/'query_savings.png',dpi=180);plt.close(fig)
if __name__=='__main__':summarize(Path(sys.argv[1]).resolve())
