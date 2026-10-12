"""ASR adapter and audio-protocol tests using a lightweight worker fixture."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.asr_adapter import ASRAdapter
from src.asr_worker import read_audio

def wav(path, rate=16000):
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.array([0,16384,-16384],dtype=np.int16).tobytes())

class AdapterTests(unittest.TestCase):
    def test_pcm_conversion_and_format(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            p=Path(temp)/'audio.wav'
            wav(p)
            np.testing.assert_allclose(read_audio(p),[0,.5,-.5])
            wav(p,8000)
            with self.assertRaises(ValueError):read_audio(p)

    def test_persistent_worker_preprocessing_and_decoding(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp)
            pkg=root/'glm_asr_cuda'
            pkg.mkdir()
            (pkg/'__init__.py').write_text('')
            (root/'torch.py').write_text('''float32="float32"
class cuda:
    @staticmethod
    def is_available():return True
    @staticmethod
    def synchronize():pass
''')
            (pkg/'ops.py').write_text('''USE_CUDA=set()
def set_cuda_groups(spec):
    if spec!="none":raise ValueError("fixture expects reference mode")
''')
            (pkg/'model.py').write_text('''from pathlib import Path
class Tensor:
    def __init__(self,name):self.name=name
    def to(self,*args):
        assert args[0]=="cuda"
        if self.name=="features":assert args[1]=="float32"
        return self
class Tokenizer:
    def decode(self,tokens,skip_special_tokens=False):
        assert tokens==[7,8] and skip_special_tokens
        return "fixture transcript"
class Processor:
    tokenizer=Tokenizer()
    def apply_transcription_request(self,audio):
        assert list(audio)==[0,.5,-.5]
        return {"input_features":Tensor("features"),"input_ids":Tensor("ids"),"input_features_mask":Tensor("mask")}
class Model:
    def generate(self,features,ids,mask,max_new_tokens):
        assert max_new_tokens==42
        assert (features.name,ids.name,mask.name)==("features","ids","mask")
        print("generation log on stdout")
        return [7,8],[1,1]
def load_model():
    p=Path(__file__).parent/"loads.txt"
    with p.open("a") as f:f.write("loaded\\n")
    print("loading log on stdout")
    return Model(),Processor()
''')
            audio=root/'query.wav'
            wav(audio)
            with ASRAdapter(asr_root=root,python=sys.executable,cuda_groups='none',max_new_tokens=42) as asr:
                asr.prepare()
                pid=asr.process.pid
                self.assertEqual(asr.transcribe(audio),'fixture transcript')
                self.assertEqual(asr.transcribe(audio),'fixture transcript')
                self.assertEqual(asr.process.pid,pid)
                self.assertEqual((pkg/'loads.txt').read_text().splitlines(),['loaded'])
                wav(audio,8000)
                with self.assertRaisesRegex(RuntimeError,'16 kHz'):asr.transcribe(audio)
            self.assertIsNone(asr.process)

    def test_unavailable_source_reports_configuration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            with ASRAdapter(asr_root=temp) as asr:
                with self.assertRaisesRegex(RuntimeError,'RAG_ASR_ROOT'):asr.prepare()

if __name__=='__main__':
    unittest.main(verbosity=2)
