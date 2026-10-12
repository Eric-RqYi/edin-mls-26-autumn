"""Run the ASR model in its own environment using a JSON-lines protocol."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import wave


def read_audio(path):
    import numpy as np
    with wave.open(str(path), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError('Expected a 16 kHz mono PCM16 WAV file')
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32)
    if not len(samples):
        raise ValueError('Audio file is empty')
    return samples / 32768.0


def transcribe(model, processor, audio, torch, max_new_tokens):
    inputs = processor.apply_transcription_request(audio)
    features = inputs['input_features'].to('cuda', torch.float32)
    input_ids = inputs['input_ids'].to('cuda')
    features_mask = inputs['input_features_mask'].to('cuda')
    tokens, _ = model.generate(features, input_ids, features_mask, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    return processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asr-root', type=Path, required=True)
    ap.add_argument('--cuda', default='all')
    ap.add_argument('--max-new-tokens', type=int, default=100)
    args = ap.parse_args()
    sys.path.insert(0, str(args.asr_root.resolve()))
    try:
        with redirect_stdout(sys.stderr):
            import torch
            from glm_asr_cuda import ops
            from glm_asr_cuda.model import load_model
            if not torch.cuda.is_available():
                raise RuntimeError('Run ASR on a CUDA GPU node')
            ops.set_cuda_groups(args.cuda)
            if ops.USE_CUDA and not ops.library_path().is_file():
                raise RuntimeError('Run make lib in hw1-asr on this GPU node first')
            model, processor = load_model()
        print(json.dumps({'ready': True}), flush=True)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                audio = read_audio(request['audio_path'])
                text = transcribe(model, processor, audio, torch, args.max_new_tokens)
            response = {'transcript': text}
        except Exception as exc:
            response = {'error': str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
