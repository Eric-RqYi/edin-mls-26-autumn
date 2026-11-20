"""Four static report candidates from exported data; no inference performed."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
COLORS=['#64748b','#0f766e','#2563eb','#c2410c','#7c3aed']

def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def save(fig,root,name):
    fig.savefig(root/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(root/(name+'.png'),dpi=200,bbox_inches='tight')
    plt.close(fig)
def render(root):
    root=Path(root);plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'figure.dpi':110})
    old=read(root/'a_historical_gemm.csv');gemm=read(root/'a_gemm_micro.csv');m1=read(root/'a_m1_micro.csv')
    fig,axs=plt.subplots(1,3,figsize=(13.2,3.5),gridspec_kw={'width_ratios':[1.45,1.3,1]})
    names=('tiled32','rb2x4','rb4x4');cases=('encoder_q','encoder_fc1','encoder_fc2','decoder_q64','decoder_gate64','decoder_down64','decoder_kv64')
    index={(r['case'],r['variant']):r for r in old};x=np.arange(len(cases))
    for j,n in enumerate(names):axs[0].bar(x+(j-1)*.23,[float(index[c,n]['paired_speedup_median']) for c in cases],.23,label=n,color=COLORS[j+1])
    axs[0].axhline(1,color='#64748b',ls='--',lw=.8);axs[0].set_xticks(x,['Enc Q','Enc FC1','Enc FC2','Dec Q','Gate/up','Down','KV'],rotation=45,ha='right');axs[0].set_ylabel('Paired speedup vs tiled16');axs[0].set_title('(a) Recorded model shapes');axs[0].legend(fontsize=7,ncol=1)
    for j,n in enumerate(('tiled16','tiled32','rb2x4','rb4x4')):
        rr=sorted((r for r in gemm if r['variant']==n and r['case'].startswith('m_tail_')),key=lambda r:int(r['M']))
        axs[1].plot([int(r['M']) for r in rr],[float(r['median_ms']) for r in rr],marker='.',ms=4,label=n,color=COLORS[j])
    axs[1].axvline(64,color='#aaa',lw=.8,ls=':');axs[1].set_xlabel('M (N=K=2048)');axs[1].set_ylabel('Kernel latency (ms)');axs[1].set_title('(b) Directed tile boundaries');axs[1].legend(fontsize=7)
    index={(r['case'],r['variant']):float(r['median_ms']) for r in m1};x=np.arange(2)
    for j,(n,label) in enumerate((('tiled16_raw','Tiled16, no M1'),('rb4x4_raw','rb4x4, no M1'),('m1_w4','M1 specialization'))):
        axs[2].bar(x+(j-1)*.23,[index[c,n] for c in ('kv','lm_head')],.23,color=COLORS[j],label=label)
    axs[2].set_yscale('log');axs[2].set_xticks(x,['KV, N=512','LM head']);axs[2].set_ylabel('Kernel latency (ms, log scale)');axs[2].set_title('(c) Explicit M=1 paths');axs[2].legend(fontsize=7)
    for ax in axs:ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2.2);save(fig,root,'fig1_gemm_boundaries')
    att=read(root/'a_attention_micro.csv');fig,axs=plt.subplots(1,2,figsize=(9.3,3.35))
    for j,(c,label) in enumerate((('length_32','T=32'),('length_128','T=128'),('length_512','T=512'),('encoder','T=1500'))):
        rr=sorted((r for r in att if r['case']==c),key=lambda r:int(r['variant'][1:]))
        axs[0].plot([int(r['variant'][1:]) for r in rr],[float(r['paired_speedup_median']) for r in rr],marker='o',ms=4,label=label,color=COLORS[j])
    for j,(c,label) in enumerate((('decoder','Causal, Q=K=64'),('causal_tail65','Causal, Q=K=65'),('cache_q1','Diagnostic: Q=1, K=71'))):
        rr=sorted((r for r in att if r['case']==c),key=lambda r:int(r['variant'][1:]))
        axs[1].plot([int(r['variant'][1:]) for r in rr],[float(r['paired_speedup_median']) for r in rr],marker='o',ms=4,label=label,color=COLORS[j+1])
    for ax in axs:
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.set_xlabel('Warps per block');ax.set_ylabel('Paired speedup vs 4 warps');ax.axhline(1,lw=.7,ls='--',color='#64748b');ax.legend(fontsize=8);ax.grid(alpha=.16)
    axs[0].set_title('(a) Encoder-style, D=64, noncausal');axs[1].set_title('(b) GQA, D=128, causal / cache diagnostics');fig.tight_layout(w_pad=2);save(fig,root,'fig2_attention_boundaries')
    stage=read(root/'b_stage_means.csv');fig,ax=plt.subplots(figsize=(6.7,3.7));bottom=np.zeros(2)
    for j,(field,label) in enumerate((('core_asr_ms','Core ASR'),('asr_other_ms','Audio/preprocess/IPC'),('local_rag_ms','Local RAG + residual'),('generation_ms','ELM response'))):
        height=np.array([float(r[field])/1000 for r in stage]);ax.bar(['Original','Final'],height,bottom=bottom,label=label,color=COLORS[j],width=.55);bottom+=height
    for i,y in enumerate(bottom):ax.text(i,y+.17,f'{y:.2f} s',ha='center',fontweight='bold')
    ax.set_ylim(0,max(bottom)*1.17);ax.set_ylabel('Mean measured request latency (s)');ax.legend(loc='upper right',fontsize=8);ax.set_title('30 audio queries × 3 rounds per configuration');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);fig.tight_layout();save(fig,root,'fig3_rag_stage_means')
    pq=read(root/'b_per_query.csv');cohort={r['query_id']:r for r in read(root/'b_cohort.csv')};fig,axs=plt.subplots(1,2,figsize=(9.3,3.4))
    for j,label in enumerate(('short','medium','long')):
        rr=[r for r in pq if r['metric']=='core_asr_ms' and r['stratum']==label]
        axs[0].scatter([int(cohort[r['query_id']]['tokens']) for r in rr],[float(r['paired_speedup_median']) for r in rr],s=30,label=label,color=COLORS[j+1],alpha=.85)
        rr=[r for r in pq if r['metric']=='end_to_end_ms' and r['stratum']==label]
        axs[1].scatter([float(r['duration_s']) for r in rr],[float(r['paired_saved_median_ms'])/1000 for r in rr],s=30,label=label,color=COLORS[j+1],alpha=.85)
    axs[0].set_xlabel('Generated ASR tokens');axs[0].set_ylabel('Query paired ASR speedup');axs[0].set_title('(a) Decoder work varies across audio')
    axs[1].set_xlabel('Audio duration (s)');axs[1].set_ylabel('Query paired RAG saving (s)');axs[1].set_title('(b) Real complete-request savings');axs[1].axhline(0,color='#64748b',ls='--',lw=.8)
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.16)
    fig.tight_layout(w_pad=2);save(fig,root,'fig4_query_results')
if __name__=='__main__':render(Path(__file__).resolve().parent)
