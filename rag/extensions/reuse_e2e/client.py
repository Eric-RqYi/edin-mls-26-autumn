"""Experiment-local IPC client and a bounded, non-retrying ELM generator."""
import datetime as dt
import json,os,subprocess,time
from pathlib import Path

class WorkerClient:
    def __init__(self,python,source,manifest,log):
        self.log=Path(log).open('w');self.counter=0
        self.process=subprocess.Popen([str(python),'-B','-u',str(source/'experiment/worker.py'),'--hw-root',str(source),'--manifest',str(manifest)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',env=dict(os.environ,HF_HUB_OFFLINE='1',ASR_DEBUG='0',CUDA_LAUNCH_BLOCKING='0'))
        try:
            self.ready=self.read()
            if not self.ready.get('ready'):raise RuntimeError('Worker not ready')
            self.hashes=self.ready['library_hashes']
        except BaseException:self.close();raise
    def read(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Worker exited; see worker.log')
        row=json.loads(line)
        if 'error' in row:raise RuntimeError('ASR worker: '+row['error'])
        return row
    def request(self,item,profile=False):
        self.counter+=1;request=dict(item,request_id=self.counter,profile=profile)
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush();row=self.read()
        if row['request_id']!=self.counter or row['variant']!=item['variant'] or row['library_sha256']!=self.hashes[item['variant']]:raise RuntimeError('Worker response/variant mismatch')
        return row
    def close(self):
        p=self.process
        if p.stdin:p.stdin.close()
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        if p.stdout:p.stdout.close()
        self.log.close()

class MeteredGenerator:
    SYSTEM="Use the supplied evidence to answer the user's question. Do not invent facts that are unsupported by the evidence."
    def __init__(self,client,model,budget):self.client,self.model,self.budget,self.attempts=client,model,budget,0
    def generate(self,prompt):
        if self.attempts>=self.budget:raise RuntimeError('Frozen API budget exhausted')
        self.attempts+=1;start=time.perf_counter()
        try:
            response=self.client.chat.completions.create(model=self.model,messages=[dict(role='system',content=self.SYSTEM),dict(role='user',content=prompt)],temperature=.1)
            usage=response.usage.model_dump() if getattr(response,'usage',None) else None
            cached=(usage or {}).get('prompt_tokens_details') or {}
            return dict(status='ok',answer=(response.choices[0].message.content or '').strip(),latency_ms=1000*(time.perf_counter()-start),attempt=self.attempts,retries=0,usage=usage,request_id=getattr(response,'_request_id',None),cached_tokens=cached.get('cached_tokens'))
        except Exception as e:
            return dict(status='failed',error_type=type(e).__name__,http_status=getattr(e,'status_code',None),request_id=getattr(e,'request_id',None),latency_ms=1000*(time.perf_counter()-start),attempt=self.attempts,retries=0,usage=None,cached_tokens=None)

def answer(item,worker,encoder,retriever,data,builder,generator):
    """Only the actual request is timed; raw serialization occurs after return."""
    row=dict(item,status='failed',started_utc=dt.datetime.now(dt.timezone.utc).isoformat());start=time.perf_counter();timing={}
    try:
        t=time.perf_counter();asr=worker.request(item);timing['asr_roundtrip_ms']=1000*(time.perf_counter()-t)
        row['asr']=asr
        t=time.perf_counter();vector=encoder.encode(asr['transcript']);timing['embedding_ms']=1000*(time.perf_counter()-t)
        t=time.perf_counter();retrieval=retriever.retrieve(vector,k=5);timing['retrieval_ms']=1000*(time.perf_counter()-t)
        t=time.perf_counter();built=builder.build(asr['transcript'],retrieval['record_ids'],data);timing['context_ms']=1000*(time.perf_counter()-t)
        generated=generator.generate(built['prompt']);timing['generation_ms']=generated['latency_ms']
        elapsed=1000*(time.perf_counter()-start)
        # Conversion and output/collection work is deliberately outside this timer.
        row.update(status='ok' if generated['status']=='ok' else 'api_failed',query_vector=vector.tolist() if hasattr(vector,'tolist') else list(vector),retrieval=retrieval,contexts=built['contexts'],prompt=built['prompt'],generation=generated)
    except Exception as e:
        elapsed=1000*(time.perf_counter()-start);row.update(error_type=type(e).__name__)
    timing['end_to_end_ms']=elapsed
    timing['unattributed_ms']=elapsed-sum(v for k,v in timing.items() if k!='end_to_end_ms')
    row['timing']=timing;row['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    return row
