"""Frozen, directed validation matrix. No adaptive configuration search."""
from pathlib import Path
import sys
HW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW / 'experiments/task3_gemm'))
from spec import FLAGS, CASE_FIELDS, CASES, VARIANTS
sys.path.insert(0, str(HW / 'experiments/task4_attention'))
from attention_spec import ATTENTION_CASES, ATTENTION_FIELDS

M_VALUES = (16,31,32,33,48,59,63,64,65,71,80,96,128)

def variants(part):
    def v(name, impl=1, tile=16, rm=4, rn=4, m1=1, hybrid=0, warps=4):
        flags = (f'{FLAGS} -DASR_TASK3_GEMM_IMPL={impl} -DASR_TASK3_TILE={tile} '
                 f'-DASR_TASK3_RM={rm} -DASR_TASK3_RN={rn} -DASR_TASK3_M1_ENABLED={m1} '
                 f'-DASR_TASK3_M1_WARPS=4 -DASR_TASK3_HYBRID_KV={hybrid} '
                 f'-DASR_ATTN_WARPS={warps} -DASR_ATTN_ONLINE_MODE=0')
        return dict(name=name,impl=impl,tile=tile,rm=rm,rn=rn,m1=m1,hybrid=hybrid,warps=warps,flags=flags)
    if part == 'gemm':
        return [v(n,**{k:c[k] for k in ('impl','tile','rm','rn')})
                for n,c in VARIANTS.items() if n != 'tiled8'] + [v('hybrid',hybrid=1)]
    if part == 'm1':
        return [v('tiled16_raw',impl=0,m1=0),v('rb4x4_raw',m1=0),v('m1_w4')]
    if part == 'attention':
        return [v(f'w{w}',warps=w) for w in (4,1,2,8,16)]
    if part == 'asr':
        return [v('base'),v('hybrid',hybrid=1),v('attention',warps=16),v('combined',hybrid=1,warps=16)]
    raise ValueError(part)

def cases(part):
    if part == 'm1': return [('kv',1,512,2048,0,0),('lm_head',1,59264,2048,0,0),('m1_tail',1,33,33,1,1)]
    if part == 'attention':
        return list(ATTENTION_CASES) + [(f'length_{t}',1,20,20,t,t,64,0,4,0) for t in (32,128,512)] + [
            ('causal_tail65',1,16,4,65,65,128,1,4,0)]
    if part == 'asr': return list(CASES)
    if part != 'gemm': raise ValueError(part)
    return ([(f'm_tail_{m}',m,2048,2048,0,0) for m in M_VALUES] +
            [(f'n_scan_m{m}_n{n}',m,n,2048,0,0) for m in (32,64,65,96) for n in (256,512,1024)] +
            [('k1024',64,512,1024,0,0),('k4096',64,512,4096,0,0)] +
            [c for c in CASES if c[0] in ('encoder_q','encoder_fc1','encoder_fc2')] +
            [c for c in CASES if c[0].startswith('edge_')])

def category(case):
    if case.startswith(('encoder_','decoder_','conv','projector')) or case in ('encoder','decoder'): return 'model_representative'
    if case.startswith('m_tail_') or case.startswith('n_scan_'): return 'directed_boundary'
    return 'diagnostic'

def path_for(v,m,n,k):
    if m == 1 and v['m1']: return 'm1_w4'
    if v['hybrid'] and m>1 and n==512 and k==2048: return 'tiled32'
    return f"tiled{v['tile']}" if v['impl']==0 else f"rb{v['rm']}x{v['rn']}"

def logical_gemm(v,m,n,k):
    path=path_for(v,m,n,k)
    if path=='m1_w4':
        return dict(kernel_path=path,threads=128,grid_x=(n+3)//4,grid_y=1,valid_slot_fraction=n/(((n+3)//4)*4),logical_load_bytes=8*n*k,static_shared_bytes=0)
    bm=bn=int(path[5:]) if path.startswith('tiled') else 0
    if not bm: bm,bn=16*v['rm'],16*v['rn']
    gx,gy=(n+bn-1)//bn,(m+bm-1)//bm
    return dict(kernel_path=path,threads=bm*bn if path.startswith('tiled') else 256,
                grid_x=gx,grid_y=gy,valid_slot_fraction=m*n/(gx*gy*bm*bn),logical_load_bytes=4*(m*k*gx+n*k*gy),static_shared_bytes=4*(bm*bm+bm*(bm+1)) if path.startswith('tiled') else 4*(bm*16+bn*17))

def logical_attention(v,b,h,hkv,sq,sk,d,causal):
    w=v['warps']; blocks=(sq+w-1)//w
    keys=sum(max(0,min(sk,((min(sq-1,(i+1)*w-1)+sk-sq)//32+1)*32)) if causal else sk for i in range(blocks))
    return dict(threads=32*w,blocks=blocks*b*h,valid_slot_fraction=sq/(blocks*w),logical_kv_load_bytes=8*b*h*d*keys,dynamic_shared_bytes=4*(32*(d+1)+32*d+w*d))
