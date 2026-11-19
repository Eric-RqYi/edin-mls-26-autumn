#!/usr/bin/env python3
"""One loaded ASR model; retained ctypes handles; per-request JSONL protocol."""
import argparse,contextlib,hashlib,json,os,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hw-root',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);args=ap.parse_args()
    sys.path.insert(0,str(args.hw_root));meta=json.loads(args.manifest.read_text())
    old_env=os.environ.get('ASR_LIB');original_lib=None;original_groups=None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            start=time.perf_counter()
            import torch
            from glm_asr_cuda import ops
            from glm_asr_cuda.model import MODEL_NAME,load_model
            from benchmark import use_cache_if_present
            from model_runner import ShapeProfiler
            from asr_worker_audio import read_audio
            if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU')
            original_lib=ops._lib;original_groups=ops.USE_CUDA.copy();libraries={};hashes={}
            for v in meta['variants']:
                actual=hashlib.sha256(Path(v['library']).read_bytes()).hexdigest()
                if actual!=v['library_sha256']:raise ValueError('Library SHA256 mismatch')
                os.environ['ASR_LIB']=v['library'];ops._lib=None;libraries[v['name']]=ops._library();hashes[v['name']]=actual
            ops.set_cuda_groups('all');ops._lib=libraries['original'];use_cache_if_present(MODEL_NAME)
            model,processor=load_model();torch.cuda.synchronize();startup_ms=1000*(time.perf_counter()-start)
        print(json.dumps(dict(ready=True,startup_ms=startup_ms,library_hashes=hashes)),flush=True)
        for line in sys.stdin:
            try:
                request=json.loads(line);name=request['variant']
                if name not in libraries:raise ValueError('Unknown variant')
                with contextlib.redirect_stdout(sys.stderr):
                    ops._lib=libraries[name];t=time.perf_counter();audio=read_audio(request['audio_path']);audio_ms=1000*(time.perf_counter()-t)
                    t=time.perf_counter();inputs=processor.apply_transcription_request(audio)
                    features=inputs['input_features'].to('cuda',torch.float32);ids=inputs['input_ids'].to('cuda');mask=inputs['input_features_mask'].to('cuda')
                    torch.cuda.synchronize();preprocess_ms=1000*(time.perf_counter()-t)
                    torch.cuda.reset_peak_memory_stats();profile=request.get('profile',False);extra={}
                    if profile:
                        with ShapeProfiler(model,ops,torch) as profiler:
                            ops.profile_start()
                            try:result=model.generate(features,ids,mask,max_new_tokens=100)
                            finally:table=ops.profile_stop()
                        shapes=profiler.rows();attention=profiler.attention_rows()
                        for operation,rows in (('linear',shapes),('attention',attention)):
                            actual=[v for (op,_),v in table.items() if op==operation]
                            if len(actual)!=1 or sum(r['calls'] for r in rows)!=actual[0][0] or abs(sum(r['gpu_event_ms'] for r in rows)-actual[0][1])>1e-5:raise ValueError('Profile boundary mismatch')
                        extra=dict(linear_shapes=shapes,attention_shapes=attention,operation_profiles=[dict(operation=op,backend=backend,calls=n,gpu_event_ms=ms) for (op,backend),(n,ms) in table.items()])
                        core_ms=None
                    else:result,core_ms=ops.wall_clock_ms(model.generate,features,ids,mask,max_new_tokens=100)
                    tokens,probs=result;tokens=list(map(int,tokens));probs=list(map(float,probs))
                    text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                    eos=bool(tokens and tokens[-1] in model.c.eos_token_ids)
                    response=dict(request_id=request['request_id'],variant=name,library_sha256=hashes[name],token_ids=tokens,selected_probs=probs,transcript=text,
                        tokens=len(tokens),eos=eos,truncated=len(tokens)>=100 and not eos,feature_shape=list(features.shape),prompt_input_tokens=ids.numel(),
                        audio_load_ms=audio_ms,preprocess_ms=preprocess_ms,core_asr_ms=core_ms,peak_allocated_bytes=torch.cuda.max_memory_allocated(),**extra)
                print(json.dumps(response,ensure_ascii=False,allow_nan=False),flush=True)
            except Exception as e:print(json.dumps(dict(error=type(e).__name__,request_id=request.get('request_id') if 'request' in locals() else None)),flush=True)
    except Exception as e:
        print(json.dumps(dict(error=type(e).__name__)),flush=True);return 1
    finally:
        if original_groups is not None:ops.USE_CUDA=original_groups;ops._lib=original_lib
        if old_env is None:os.environ.pop('ASR_LIB',None)
        else:os.environ['ASR_LIB']=old_env
    return 0
if __name__=='__main__':raise SystemExit(main())
