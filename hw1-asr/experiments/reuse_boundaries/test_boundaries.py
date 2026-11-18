import contextlib,csv,io,json,os,tempfile,unittest
from unittest import mock
from pathlib import Path
from boundary_spec import variants,cases,path_for,logical_gemm,logical_attention
import run_boundaries as run
from boundary_analysis import import_history

class BoundariesTest(unittest.TestCase):
    def test_frozen_matrix_and_paths(self):
        self.assertEqual(len(variants('gemm')),6)
        for part in ('gemm','m1','attention','asr'):
            for v in variants(part):
                self.assertIn('-DASR_ATTN_ONLINE_MODE=0',v['flags'])
                self.assertNotIn('fast_math',v['flags'])
        v=variants('gemm')[-1]
        self.assertEqual(path_for(v,2,512,2048),'tiled32')
        self.assertEqual(path_for(v,1,512,2048),'m1_w4')
        self.assertEqual(path_for(v,64,512,1024),'rb4x4')
        self.assertEqual([path_for(v,1,512,2048) for v in variants('m1')],['tiled16','rb4x4','m1_w4'])
    def test_required_shapes_and_logical_counts(self):
        self.assertEqual(len(cases('attention')),18)
        self.assertEqual(len(cases('gemm')),37)
        r=logical_gemm(variants('gemm')[4],65,2048,2048)
        self.assertAlmostEqual(r['valid_slot_fraction'],65/128)
        a,b=[logical_attention(v,1,20,20,1500,1500,64,0) for v in (variants('attention')[0],variants('attention')[-1])]
        self.assertEqual(a['blocks'],375*20);self.assertEqual(b['blocks'],94*20)
        self.assertAlmostEqual(a['logical_kv_load_bytes']/b['logical_kv_load_bytes'],375/94)
    def test_sanitizer_route_and_model_order(self):
        a=run.parse(['--stage','asr']);out=Path('/tmp/new-job');m=run.manifest('asr',out,a,{})
        steps=run.steps('asr',out,a,m,'90')
        hybrid=[s for s in steps if s.log.startswith('sanitizer/hybrid_')]
        self.assertEqual(len(hybrid),3)
        for s in hybrid:self.assertNotIn('--small',s.argv)
        self.assertLess(max(steps.index(s) for s in hybrid),next(i for i,s in enumerate(steps) if s.log.startswith('model/')))
    def test_dryrun_no_gpu_or_directory(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'absent'
            with contextlib.redirect_stdout(io.StringIO()) as f:self.assertEqual(run.main(['--dry-run','--sm','90','--output',str(target)]),0)
            self.assertFalse(target.exists());self.assertIn('block3.json',f.getvalue())
    def test_missing_evidence_not_invented(self):
        with tempfile.TemporaryDirectory() as d:
            import_history(Path(d)/'missing',Path(d))
            self.assertTrue((Path(d)/'history_manifest.json').exists())
            self.assertFalse((Path(d)/'history_gemm.csv').exists())
    def test_snapshot_has_reference_and_shared_model_tool(self):
        with tempfile.TemporaryDirectory() as d:
            run.snapshot(Path(d))
            for n in ('benchmark_gemm.cu','benchmark_attention.cu','model_runner.py','boundary_spec.py'):
                self.assertTrue((Path(d)/'source/experiment'/n).exists())
    def test_sanitizer_failure_stops_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'job';logs=[];before={k:os.environ.get(k) for k in ('TMPDIR','NVCC_CCBIN')}
            def step(s,job):
                logs.append(s.log)
                if s.log=='sanitizer/hybrid_memcheck.log':raise RuntimeError('fixture sanitizer failure')
            with mock.patch.object(run.runtime,'gpu_info',return_value=dict(name='fixture',sm='90')), mock.patch.object(run.runtime,'run_step',side_effect=step):
                with self.assertRaisesRegex(RuntimeError,'sanitizer'):
                    run.main(['--stage','asr','--output',str(out),'--history-root',str(Path(d)/'missing'),'--cuda-tmpdir',str(Path(d)/'scratch')])
            self.assertFalse(any(n.startswith('model/') for n in logs))
            self.assertEqual(json.loads((out/'asr/metadata.json').read_text())['status'],'failed')
            self.assertEqual(before,{k:os.environ.get(k) for k in before})
    def test_raw_m1_summary_has_three_distinct_paths_and_rejects_missing_samples(self):
        from boundary_analysis import analyze
        from results import write_csv
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);a=run.parse(['--repeats','2']);m=run.manifest('m1',out,a,{});run.snapshot(out)
            (out/'metadata.json').write_text(json.dumps(m));rr=[]
            for c,M,N,K,bias,act in m['cases']:
                for rep in (1,2):
                    for order,v in enumerate(m['variants']):
                        rr.append(dict(case=c,M=M,N=N,K=K,bias=bias,act=act,variant=v['name'],repeat=rep,order=order,ms=1,gflops=1,max_abs_error=0,wrong=0,reference='CPU_double' if M*N*K<=2e7 else 'cuBLAS_FP32',seed=21,warmup=5,iterations=20))
            (out/'build').mkdir()
            for v in m['variants']:(out/'build'/f"{v['name']}.log").write_text('fixture')
            write_csv(out/'micro.csv',rr);analyze(out)
            table=list(csv.DictReader(io.StringIO((out/'shape_comparison.csv').read_text())))
            self.assertEqual({r['kernel_path'] for r in table},{'tiled16','rb4x4','m1_w4'})
            write_csv(out/'micro.csv',rr[:-1])
            with self.assertRaises(ValueError):analyze(out)
            self.assertFalse((out/'analysis.md').exists())
if __name__=='__main__':unittest.main()
