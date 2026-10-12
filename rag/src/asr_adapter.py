"""File-to-transcript interface for the sibling ASR coursework module."""
from __future__ import annotations
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys


class ASRNotIntegratedError(RuntimeError):
    """The ASR runtime or source directory is unavailable."""


class ASRAdapter:
    """Reuse one ASR worker and its loaded model across audio queries.

    RAG_ASR_PYTHON selects the ASR environment's interpreter. ASR_CUDA uses
    the kernel-group syntax documented in hw1-asr. The worker owns ASR imports
    and CUDA state, independently of the RAG encoder's environment.
    """

    def __init__(self, asr_root=None, python=None, cuda_groups=None, max_new_tokens=100):
        root = Path(__file__).resolve().parents[1]
        self.asr_root = Path(asr_root or os.getenv('RAG_ASR_ROOT') or root.parent / 'hw1-asr').resolve()
        self.python = python or os.getenv('RAG_ASR_PYTHON') or sys.executable
        self.cuda_groups = cuda_groups or os.getenv('ASR_CUDA', 'all')
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        self.max_new_tokens = max_new_tokens
        self.process = None
        atexit.register(self.close)

    def prepare(self):
        """Load ASR once before measuring a query's inference time."""
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError('ASR worker exited; see its error output')
            return
        if not (self.asr_root / 'glm_asr_cuda/model.py').is_file():
            raise ASRNotIntegratedError('Set RAG_ASR_ROOT to the hw1-asr directory')
        worker = Path(__file__).with_name('asr_worker.py')
        env = os.environ.copy()
        env['HF_HUB_OFFLINE'] = '1'
        self.process = subprocess.Popen(
            [self.python, '-B', '-u', str(worker), '--asr-root', str(self.asr_root),
             '--cuda', self.cuda_groups, '--max-new-tokens', str(self.max_new_tokens)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding='utf-8', env=env,
        )
        try:
            ready = self._read_response()
            if ready.get('ready') is not True:
                raise RuntimeError('ASR worker did not report readiness')
        except BaseException:
            self.close()
            raise

    def _read_response(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('ASR worker stopped; check its interpreter, GPU and model setup')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError('ASR: ' + result['error'])
        return result

    def transcribe(self, audio_path):
        audio = Path(audio_path).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        self.prepare()
        self.process.stdin.write(json.dumps({'audio_path': str(audio)}) + '\n')
        self.process.stdin.flush()
        text = self._read_response()['transcript'].strip()
        if not text:
            raise RuntimeError(f'ASR returned an empty transcript for {audio}')
        return text

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout:
            process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

