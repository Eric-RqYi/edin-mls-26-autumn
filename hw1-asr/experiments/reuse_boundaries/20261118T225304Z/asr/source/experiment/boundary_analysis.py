"""Coverage-checked tables and at most two report figures; never rewrite inputs."""
import csv,hashlib,json,re,subprocess,sys
from pathlib import Path
from boundary_spec import category,logical_gemm,logical_attention
from results import micro_summary,model_summaries,write_csv
from attention_results import attention_micro_summary,validate_attention_profiles,factorial_effects

HISTORY=(('task3_gemm','20261117T192802Z-615737','gemm'),
         ('task4_attention','20261118T001456Z-748564','attention'),
         ('task4_attention','20261118T082810Z-1093639','inventory_only'),
         ('task3_m1','20261117T154855Z-483031','inventory_only'))

def import_history(root,out):
    root=Path(root).resolve();inventory=[];tables={'gemm':[],'attention':[]}
    for folder,name,kind in HISTORY:
        run=root/folder/'results'/name
        if not (run/'metadata.json').exists():inventory.append(dict(run=str(run),status='missing'));continue
        meta=json.loads((run/'metadata.json').read_text())
        if not (run/'source_sha256.json').exists():
            inventory.append(dict(run=str(run),status='not_imported_missing_source_manifest'));continue
        hashes=json.loads((run/'source_sha256.json').read_text())
        if meta['status']!='complete':raise ValueError(f'Incomplete history: {run}')
        if not all(hashlib.sha256((run/'source'/p).read_bytes()).hexdigest()==v for p,v in hashes.items()):raise ValueError('Historical frozen source mismatch')
        inventory.append(dict(run=str(run),status='verified',kind=kind,gpu=meta['gpu'],source_files=len(hashes),
                              metadata_sha256=hashlib.sha256((run/'metadata.json').read_bytes()).hexdigest()))
        if kind=='inventory_only':continue
        # Isolate each run's frozen generic spec/results modules in a subprocess.
        code='''import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);sys.path.insert(0,str(p/'source/experiment'))\nm=json.loads((p/'metadata.json').read_text())\nif sys.argv[2]=='gemm':\n from results import micro_summary\n rows=micro_summary(p,m)\nelse:\n from attention_results import attention_micro_summary\n rows=attention_micro_summary(p,m)\n rows=[r for r in rows if next(v for v in m['variants'] if v['name']==r['variant'])['online']==0]\nprint(json.dumps(rows))'''
        rows=json.loads(subprocess.check_output([sys.executable,'-c',code,str(run),kind],text=True))
        for row in rows:
            if kind=='gemm':
                v=next(v for v in meta['variants'] if v['name']==row['variant'])
                row['kernel_path']='m1_w4' if row['M']==1 else ('rb'+str(v['rm'])+'x'+str(v['rn']) if v['impl'] else 'tiled'+str(v['tile']))
                row['family']=gemm_family(row)
            row['run']=name;row['hardware']=meta['gpu']['name']
        tables[kind].extend(rows)
    (out/'history_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    for kind,rows in tables.items():
        if rows:write_csv(out/f'history_{kind}.csv',rows)
    (out/'history_analysis.md').write_text('# Existing evidence\n\n'+
        '\n'.join(f"- {r['run']}: {r['status']}" for r in inventory)+
        '\n\nM=1 in the previous six-configuration GEMM comparison uses the same specialized path. It is not a raw register-blocking comparison. Hardware-specific timings are not pooled across allocations.\n')

def gemm_family(row):
    if row['M']==1:return 'M1'
    if row['N']==512:return 'narrow_KV'
    c=row['case']
    if c.startswith('encoder'):return 'encoder'
    if 'gate' in c:return 'decoder_gate_up'
    if 'down' in c:return 'decoder_down'
    if c.startswith('conv') or c.startswith('projector'):return c.split('_')[0]
    return 'decoder_Q_O_or_boundary'

def analyze(job):
    meta=json.loads((job/'metadata.json').read_text());part=meta['part']
    for name in ('analysis.md','shape_comparison.csv','report_candidate.png','compiler_resources.csv'):(job/name).unlink(missing_ok=True)
    if meta['status'] not in ('running','complete'):raise ValueError('Cannot summarize failed job')
    hashes=json.loads((job/'source_sha256.json').read_text())
    if not all(hashlib.sha256((job/'source'/p).read_bytes()).hexdigest()==h for p,h in hashes.items()):raise ValueError('Frozen hash mismatch')
    if part=='asr':
        samples,pooled,blocks,ops,shapes=model_summaries(job,meta)
        attention,checks=validate_attention_profiles(job,meta)
        effects=factorial_effects(samples,meta)
        for n,rows in (('model_samples',samples),('model_summary',pooled),('model_blocks',blocks),('operation_profiles',ops),('linear_shapes',shapes),('attention_shapes',attention),('profile_boundaries',checks),('factorial_effects',effects)):
            write_csv(job/(n+'.csv'),rows)
        lines=['# ASR controlled comparison','','| Configuration | Median ms | Mean ms | IQR ms | Paired speedup |','|---|---:|---:|---:|---:|']
        lines += [f"| {r['variant']} | {r['median_ms']:.3f} | {r['mean_ms']:.3f} | {r['iqr_ms']:.3f} | {r['median_block_paired_speedup']:.5f}x |" for r in pooled]
        lines += ['','Attention factor varies only block query/warp count (4/16). Hybrid factor varies only the fixed narrow-N dispatch. Every timed token sequence matches the reference. All samples retained; one supplied audio, three processes in one allocation. Profiles are separate diagnostics and include original ops allocation/launch boundaries.']
    else:
        rows=attention_micro_summary(job,meta) if part=='attention' else micro_summary(job,meta)
        vv={v['name']:v for v in meta['variants']}
        for r in rows:
            v=vv[r['variant']];r['coverage_class']=category(r['case'])
            if part=='attention':
                r.update(warps=v['warps'],**logical_attention(v,*(r[f] for f in ('batch','num_heads','num_kv_heads','seq_q','seq_k','head_dim','causal'))))
            else:r.update(family=gemm_family(r),**logical_gemm(v,r['M'],r['N'],r['K']))
        write_csv(job/'shape_comparison.csv',rows)
        lines=[f'# {part} shape boundaries','','Coverage and finite timing/correctness checks passed. Full machine-readable table: shape_comparison.csv.','',
               'Logical load counts are valid-element requests from the source algorithm, not measured DRAM traffic. Output slot utilization includes tile tails. Register blocking changes both per-thread work and block tile. These are directed validation shapes, not an unseen independent test set.']
        if part=='m1':lines+=['','The first two configurations explicitly disable the M1 path; the third enables it.']
        if part=='gemm':
            rb=[r for r in rows if r['variant']=='rb4x4'];fast=[r for r in rb if r['paired_speedup_median']>1];slow=[r for r in rb if r['paired_speedup_median']<1]
            lines += ['',f'Against tiled16, rb4x4 improves {len(fast)}/{len(rb)} directed cases and regresses on {len(slow)}. This count is not weighted by model call frequency.']
            lines += ['','| Case | M,N,K | rb4x4 paired speedup | Valid output slots |','|---|---|---:|---:|']
            chosen=sorted(rb,key=lambda r:r['paired_speedup_median'])[:3]+sorted(rb,key=lambda r:-r['paired_speedup_median'])[:3]
            lines += [f"| {r['case']} | {r['M']},{r['N']},{r['K']} | {r['paired_speedup_median']:.4f}x | {r['valid_slot_fraction']:.3f} |" for r in chosen]
            lines += ['','Use M=31/32/33 and 63/64/65 rows to check discontinuities. The slot-fraction and request-count columns quantify tile waste and logical reuse. Narrow-N and adjacent N/K rows test the fixed dispatch boundary; they do not modify it. Actual GPU traffic, occupancy and cache behavior require separate measurements.']
        if part=='attention':
            lines += ['','| Case | Coverage | Fastest warp configuration | Paired speedup vs w4 |','|---|---|---|---:|']
            for case in dict.fromkeys(r['case'] for r in rows):
                r=min((r for r in rows if r['case']==case),key=lambda r:r['median_ms'])
                lines.append(f"| {case} | {r['coverage_class']} | {r['variant']} | {r['paired_speedup_median']:.4f}x |")
            lines += ['','Length/GQA/causal conditions are compared separately; stress cases do not count as additional model workloads. This table tests whether the long-encoder choice transfers to short or low-parallelism inputs.']
        resources=[]
        for v in meta['variants']:
            log=(job/'build'/f"{v['name']}.log").read_text()
            for kernel,body in re.findall(r'Function properties for ([^\n]+)\n(.*?)(?=ptxas info    : Compile time|ptxas info    : Function properties|$)',log,re.S):
                regs=re.search(r'Used (\d+) registers',body);spill=re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',body)
                resources.append(dict(variant=v['name'],kernel=kernel,registers=int(regs[1]) if regs else '',spill_store_bytes=int(spill[1]) if spill else '',spill_load_bytes=int(spill[2]) if spill else '',static_shared_bytes=(re.search(r'(\d+) bytes smem',body)[1] if re.search(r'(\d+) bytes smem',body) else 0)))
        if resources:write_csv(job/'compiler_resources.csv',resources)
        plot(job,part,rows)
    (job/'analysis.md').write_text('\n'.join(lines)+'\n')

def plot(job,part,rows):
    if part not in ('gemm','attention'):return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        (job/'plot_unavailable.txt').write_text('Install matplotlib to render the candidate figure; numerical tables are available.\n');return
    fig,ax=plt.subplots(figsize=(7,4))
    if part=='gemm':
        for name in dict.fromkeys(r['variant'] for r in rows):
            rr=sorted((r for r in rows if r['variant']==name and r['case'].startswith('m_tail_')),key=lambda r:r['M'])
            ax.plot([r['M'] for r in rr],[r['median_ms'] for r in rr],marker='.',label=name)
        ax.set_xlabel('M (N=K=2048)')
    else:
        for t in (32,128,512,1500):
            rr=sorted((r for r in rows if r['seq_q']==t and r['seq_k']==t and r['num_heads']==20 and r['head_dim']==64 and not r['causal']),key=lambda r:r['warps'])
            ax.plot([r['warps'] for r in rr],[r['paired_speedup_median'] for r in rr],marker='.',label=f'T={t}')
        ax.set_xlabel('Warps per block')
    ax.set_ylabel('Kernel latency (ms)' if part=='gemm' else 'Paired speedup vs 4 warps')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(job/'report_candidate.png',dpi=180);plt.close(fig)
if __name__=='__main__':analyze(Path(sys.argv[1]).resolve())
