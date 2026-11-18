"""Synthetic offline fixtures; none of these timings is a performance result."""
import contextlib,csv,hashlib,io,json,sys,tempfile,unittest,wave
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock
from design import freeze,schedule,variants,canonical
from client import MeteredGenerator,answer,WorkerClient
import run_e2e as run
import summarize_e2e as summary

def wav(path,frames):
    with wave.open(str(path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*frames)

def dataset(root,n=60):
    rr=[]
    for split,count in (('dev',n),('practice',2)):
        for i in range(count):
            name=f'{split}{i:03d}.wav';wav(root/name,32+i*8);rr.append(dict(query_id=name,split=split,audio_file=name))
    (root/'student_audio_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))

def fixture(out):
    data=out/'data';data.mkdir();dataset(data)
    cohort=freeze(data);plan=schedule(cohort);vv=[dict(v,library_sha256=v['name']*8) for v in variants()]
    meta=dict(schema=1,variants=vv,sample_manifest_sha256=cohort['manifest_sha256'],plan_sha256=hashlib.sha256(canonical(plan).encode()).hexdigest(),api_budget=184,api_client_calls=184)
    (out/'sample_manifest.json').write_text(json.dumps(cohort));(out/'metadata.json').write_text(json.dumps(meta))
    src=out/'source/rag/src';src.mkdir(parents=True)
    (src/'metrics.py').write_text((run.RAG/'src/metrics.py').read_text());(src/'__init__.py').touch()
    (out/'qrels_dev.tsv').write_text('query_id\trecord_id\tscore\n'+''.join(f"{r['query_id']}\tdoc\t1\n" for r in cohort['items']))
    rows=[]
    for j,item in enumerate(plan,5):
        name=item['variant'];v=next(v for v in vv if v['name']==name)
        asr=dict(variant=name,library_sha256=v['library_sha256'],token_ids=[1,2],tokens=2,transcript='question',core_asr_ms=100 if name=='original' else 50)
        r=dict(item,status='ok',asr=asr,query_vector=[.1,.2],retrieval=dict(record_ids=['doc'],scores=[.9]),contexts=['evidence'],prompt='same prompt',
               generation=dict(status='ok',attempt=j,retries=0,answer='answer'),timing=dict(end_to_end_ms=1124 if name=='original' else 1074,asr_roundtrip_ms=120 if name=='original' else 70,embedding_ms=1,retrieval_ms=1,context_ms=1,generation_ms=1000,unattributed_ms=1))
        rows.append(r)
    (out/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'warmup.jsonl').write_text(''.join(json.dumps(dict(generation=dict(attempt=i,retries=0)))+'\n' for i in range(1,5)))
    return meta,cohort,rows

class E2ETest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
    def test_sampling_and_schedule_are_frozen_and_stratified(self):
        dataset(self.out);a=freeze(self.out);b=freeze(self.out)
        self.assertEqual(a,b);self.assertEqual(len(a['items']),30)
        self.assertEqual({s:sum(r['stratum']==s for r in a['items']) for s in ('short','medium','long')},dict(short=10,medium=10,long=10))
        p=schedule(a);self.assertEqual(len(p),180)
        self.assertEqual(sum(r['variant']=='original' for r in p),90)
        self.assertEqual({r['order'] for r in p if r['variant']=='original'},{0,1})
    def test_limited_and_corrupt_frame_not_silently_replaced(self):
        dataset(self.out,12)
        with self.assertRaises(ValueError):freeze(self.out)
        d=freeze(self.out,True);self.assertTrue(d['limited_subset']);self.assertEqual(len(d['items']),12)
        (self.out/'dev000.wav').write_bytes(b'broken')
        d=freeze(self.out,True);self.assertEqual(len(d['items']),11);self.assertEqual(len(d['invalid_files']),1)
    def test_explicit_baseline_and_sanitizer_path(self):
        vv=variants();self.assertIn('-DASR_TASK3_M1_ENABLED=0',vv[0]['flags']);self.assertIn('-DASR_TASK3_M1_ENABLED=1',vv[1]['flags'])
        steps=run.build_steps(self.out,[dict(v,build='/tmp/b/'+v['name'],library='/tmp/b/'+v['name']+'/lib.so') for v in vv],'90')
        sanitizer=[s for s in steps if s.log.startswith('sanitizer/')]
        self.assertEqual(len(sanitizer),3)
        for s in sanitizer:self.assertNotIn('--small',s.argv)
    def test_plan_is_offline_side_effect_free(self):
        target=self.out/'never'
        with contextlib.redirect_stdout(io.StringIO()) as text:self.assertEqual(run.main(['--output',str(target)]),0)
        self.assertFalse(target.exists());self.assertIn('180 formal + 4 warmup',text.getvalue())
    def test_api_budget_failure_and_no_hidden_retry(self):
        create=mock.Mock(side_effect=RuntimeError('fixture failure'))
        g=MeteredGenerator(NS(chat=NS(completions=NS(create=create))),'model',2)
        for _ in range(2):self.assertEqual(g.generate('prompt')['status'],'failed')
        with self.assertRaises(RuntimeError):g.generate('prompt')
        self.assertEqual(create.call_count,2);self.assertEqual(g.attempts,2)
    def test_success_usage_and_same_generation_settings(self):
        create=mock.Mock(return_value=NS(choices=[NS(message=NS(content=' answer '))],usage=NS(model_dump=lambda:dict(prompt_tokens=10,prompt_tokens_details=dict(cached_tokens=5))),_request_id='id'))
        g=MeteredGenerator(NS(chat=NS(completions=NS(create=create))),'model',1);r=g.generate('prompt')
        self.assertEqual(r['cached_tokens'],5);self.assertEqual(r['retries'],0)
        self.assertEqual(create.call_args.kwargs['temperature'],.1)
    def test_real_parent_boundary_calls_each_stage_once(self):
        w=NS(request=mock.Mock(return_value=dict(transcript='question')))
        encoder=NS(encode=mock.Mock(return_value=[.1,.2]));ret=NS(retrieve=mock.Mock(return_value=dict(record_ids=['doc'],scores=[.9])))
        builder=NS(build=mock.Mock(return_value=dict(prompt='prompt',contexts=['evidence'])))
        generator=NS(generate=mock.Mock(return_value=dict(status='ok',latency_ms=0.0,answer='answer')))
        row=answer(dict(query_id='q'),w,encoder,ret,None,builder,generator)
        self.assertEqual(row['status'],'ok');w.request.assert_called_once();generator.generate.assert_called_once()
        total=sum(v for k,v in row['timing'].items() if k!='end_to_end_ms')
        self.assertAlmostEqual(total,row['timing']['end_to_end_ms'])
    def test_query_cluster_summary_and_token_mismatch(self):
        _,_,rr=fixture(self.out)
        with mock.patch.object(summary,'plot'):r=summary.summarize(self.out)
        self.assertEqual(r['status'],'complete');self.assertEqual(r['api_client_calls'],184)
        tables=list(csv.DictReader(io.StringIO((self.out/'summary.csv').read_text())))
        row=next(r for r in tables if r['group']=='all' and r['metric']=='core_asr_ms')
        self.assertEqual(int(row['queries']),30);self.assertEqual(float(row['paired_speedup']),2)
        rr[0]['asr']['token_ids']=[2,1];(self.out/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
        with mock.patch.object(summary,'plot'):self.assertEqual(summary.summarize(self.out)['status'],'incomplete_or_mismatch')
    def test_duplicate_wrong_library_and_bad_order_rejected(self):
        _,_,rr=fixture(self.out)
        for changed in ('duplicate','library','order'):
            data=json.loads(json.dumps(rr))
            if changed=='duplicate':data.append(data[0])
            if changed=='library':data[0]['asr']['library_sha256']='wrong'
            if changed=='order':data[0]['order']=99
            (self.out/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in data))
            with self.subTest(changed=changed),self.assertRaises(ValueError):summary.summarize(self.out)
    def test_api_failures_remain_and_reduce_only_complete_e2e_coverage(self):
        _,_,rr=fixture(self.out);rr[-1]['status']='api_failed';rr[-1]['generation']['status']='failed'
        (self.out/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
        with mock.patch.object(summary,'plot'):r=summary.summarize(self.out)
        self.assertEqual(r['failed_requests'],1);self.assertEqual(r['recorded_requests'],180)
        table=list(csv.DictReader(io.StringIO((self.out/'summary.csv').read_text())))
        self.assertEqual(int(next(x for x in table if x['group']=='all' and x['metric']=='end_to_end_ms')['queries']),29)
        self.assertEqual(int(next(x for x in table if x['group']=='all' and x['metric']=='core_asr_ms')['queries']),30)
    def test_worker_protocol_retains_identity_and_closes(self):
        src=self.out/'source';(src/'experiment').mkdir(parents=True)
        (src/'experiment/worker.py').write_text("import json,sys\nprint(json.dumps(dict(ready=True,library_hashes=dict(original='hash'))),flush=True)\nfor line in sys.stdin:\n r=json.loads(line);print(json.dumps(dict(request_id=r['request_id'],variant=r['variant'],library_sha256='hash')),flush=True)\n")
        w=WorkerClient(sys.executable,src,self.out/'manifest',self.out/'worker.log')
        try:self.assertEqual(w.request(dict(variant='original',audio_path='fixture'))['library_sha256'],'hash')
        finally:w.close()
        self.assertIsNotNone(w.process.returncode)
    def test_freeze_main_and_audio_hash_guard(self):
        data=self.out/'assets';data.mkdir();dataset(data)
        out=self.out/'cohort'
        self.assertEqual(run.main(['--mode','freeze','--data-dir',str(data),'--output',str(out)]),0)
        args=run.parse(['--sample-manifest',str(out/'sample_manifest.json')])
        cohort=run.load_sample(args)
        Path(cohort['items'][0]['audio_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'audio changed'):run.load_sample(args)
    def test_failed_resummary_clears_stale_success(self):
        _,_,rr=fixture(self.out)
        with mock.patch.object(summary,'plot'):summary.summarize(self.out)
        rr.append(rr[0]);(self.out/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        with self.assertRaises(ValueError):summary.summarize(self.out)
        self.assertFalse((self.out/'analysis.md').exists())
    def test_boundary_corruption_cannot_publish_success(self):
        _,_,rr=fixture(self.out);rr[0]['timing']['end_to_end_ms']+=100
        (self.out/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        with self.assertRaisesRegex(ValueError,'boundary'):summary.summarize(self.out)
    def test_missing_interpreter_records_failure_without_api(self):
        assets=self.out/'assets';assets.mkdir();dataset(assets);target=self.out/'failed'
        with self.assertRaises(RuntimeError):run.main(['--mode','run','--data-dir',str(assets),'--output',str(target),'--asr-python','/not/an/interpreter'])
        self.assertTrue((target/'FAILED.txt').exists());self.assertFalse((target/'raw.jsonl').exists())
if __name__=='__main__':unittest.main()
