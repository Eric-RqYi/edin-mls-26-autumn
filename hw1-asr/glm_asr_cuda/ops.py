"""The bridge between the model (PyTorch tensors) and your CUDA kernels.

Every operation the model needs is a function here. Each one has two versions:
your CUDA kernel from libasr_kernels.so, called through ctypes, and a PyTorch
reference. ASR_CUDA (or ./run.sh --cuda ...) chooses which version runs:

    all                   every kernel group uses CUDA (default)
    none                  everything uses the PyTorch reference
    rope,reduce,...       only these groups use CUDA (task1..task4 also work)

The groups: rope (task 1), reduce (task 2), gemm (task 3), attention (task 4),
and provided (kernels in provided.cu; on unless ASR_CUDA=none).

You do not need to change this file.
"""

import ctypes
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

HW_DIR = Path(__file__).resolve().parent.parent
TASK_GROUPS = ("rope", "reduce", "gemm", "attention")
ALIASES = {"task1": "rope", "task2": "reduce", "task3": "gemm", "task4": "attention"}


def _parse_groups(spec):
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return set(TASK_GROUPS) | {"provided"}
    if spec == "none":
        return set()
    groups = {"provided"}
    for name in spec.split(","):
        name = ALIASES.get(name.strip(), name.strip())
        if not name:
            continue
        if name not in TASK_GROUPS + ("provided",):
            raise ValueError(f"Unknown kernel group '{name}'. Use all, none, or a list of "
                             f"{', '.join(TASK_GROUPS)} (or task1..task4).")
        groups.add(name)
    return groups


USE_CUDA = _parse_groups(os.environ.get("ASR_CUDA"))
DEBUG = os.environ.get("ASR_DEBUG", "0") not in ("", "0")


def set_cuda_groups(spec):
    """Choose which kernel groups run on your CUDA code (same syntax as ASR_CUDA)."""
    global USE_CUDA
    USE_CUDA = _parse_groups(spec)


# ----------------------------------------------------------------------------
# Loading libasr_kernels.so
# ----------------------------------------------------------------------------

_lib = None
_P, _I, _F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
_SIGNATURES = {
    "launch_rope": [_P, _P, _P, _I, _I, _I, _I, _I, _I, _P],
    "launch_rmsnorm": [_P, _P, _P, _I, _I, _F, _P],
    "launch_layernorm": [_P, _P, _P, _P, _I, _I, _F, _P],
    "launch_softmax": [_P, _P, _I, _I, _P],
    "launch_linear": [_P, _P, _P, _P, _I, _I, _I, _I, _P],
    "launch_attention": [_P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _F, _I, _P],
    "launch_silu_mul": [_P, _P, _P, _I, _P],
    "launch_embedding": [_P, _P, _P, _I, _I, _P],
    "launch_im2col_1d": [_P, _P, _I, _I, _I, _I, _I, _I, _I, _P],
}


def library_path():
    if os.environ.get("ASR_LIB"):
        return Path(os.environ["ASR_LIB"])
    major, minor = torch.cuda.get_device_capability()
    return HW_DIR / "build" / f"sm_{major}{minor}" / "libasr_kernels.so"


def _library():
    global _lib
    if _lib is None:
        path = library_path()
        if not path.exists():
            raise RuntimeError(f"{path} does not exist. Run `make` in {HW_DIR} on this GPU node "
                               "first (./run.sh does it for you).")
        lib = ctypes.CDLL(str(path))
        for name, argtypes in _SIGNATURES.items():
            getattr(lib, name).argtypes = argtypes
            getattr(lib, name).restype = ctypes.c_int
        lib.asr_error_string.argtypes = [ctypes.c_int]
        lib.asr_error_string.restype = ctypes.c_char_p
        _lib = lib
    return _lib


def _ptr(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _launch(op, fn_name, *args):
    lib = _library()
    code = getattr(lib, fn_name)(*args, _stream())
    if code != 0:
        msg = lib.asr_error_string(code).decode()
        raise RuntimeError(f"{op}: launching the kernel failed with CUDA error {code} ({msg})")
    if DEBUG:
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            raise RuntimeError(f"{op}: the kernel failed while running: {e}") from None


def _check_float32(op, *tensors):
    for t in tensors:
        if t is not None and (t.dtype != torch.float32 or not t.is_cuda or not t.is_contiguous()):
            raise ValueError(f"{op}: expected contiguous float32 CUDA tensors, got "
                             f"{t.dtype} on {t.device}, contiguous={t.is_contiguous()}")


# ----------------------------------------------------------------------------
# Timing (./run.sh --profile)
# ----------------------------------------------------------------------------

_profile = None  # list of (op, backend, start event, end event) while profiling


@contextmanager
def _timed(op, group):
    if _profile is None:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    _profile.append((op, "cuda" if group in USE_CUDA else "torch", start, end))


def profile_start():
    global _profile
    _profile = []


def profile_stop():
    """Stop profiling; return {(op, backend): (calls, total ms)}."""
    global _profile
    torch.cuda.synchronize()
    totals = defaultdict(lambda: [0, 0.0])
    for op, backend, start, end in _profile:
        totals[(op, backend)][0] += 1
        totals[(op, backend)][1] += start.elapsed_time(end)
    _profile = None
    return {k: tuple(v) for k, v in totals.items()}


# ----------------------------------------------------------------------------
# Task 1: RoPE
# ----------------------------------------------------------------------------

def rope_(q, k, inv_freq, rotary_dim):
    """Rotate q [B, S, Hq, D] and k [B, S, Hk, D] in place; positions are 0..S-1."""
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_k_heads = k.shape[2]
    with _timed("rope", "rope"):
        if "rope" in USE_CUDA:
            _check_float32("rope", q, k, inv_freq)
            _launch("rope", "launch_rope", _ptr(q), _ptr(k), _ptr(inv_freq), batch, seq_len,
                    num_q_heads, num_k_heads, head_dim, rotary_dim)
        else:
            half = rotary_dim // 2
            pos = torch.arange(seq_len, device=q.device, dtype=torch.float32)
            angle = pos[:, None] * inv_freq[None, :]                # [S, half]
            cos = angle.cos()[None, :, None, :]
            sin = angle.sin()[None, :, None, :]
            for x in (q, k):
                a = x[..., :half].clone()
                b = x[..., half:rotary_dim].clone()
                x[..., :half] = a * cos - b * sin
                x[..., half:rotary_dim] = b * cos + a * sin
    return q, k


# ----------------------------------------------------------------------------
# Task 2: reductions
# ----------------------------------------------------------------------------

def rmsnorm(x, weight, eps):
    cols = x.shape[-1]
    with _timed("rmsnorm", "reduce"):
        if "reduce" in USE_CUDA:
            x = x.contiguous()
            y = torch.empty_like(x)
            _check_float32("rmsnorm", x, weight)
            _launch("rmsnorm", "launch_rmsnorm", _ptr(x), _ptr(weight), _ptr(y),
                    x.numel() // cols, cols, eps)
            return y
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def layernorm(x, weight, bias, eps):
    cols = x.shape[-1]
    with _timed("layernorm", "reduce"):
        if "reduce" in USE_CUDA:
            x = x.contiguous()
            y = torch.empty_like(x)
            _check_float32("layernorm", x, weight, bias)
            _launch("layernorm", "launch_layernorm", _ptr(x), _ptr(weight), _ptr(bias), _ptr(y),
                    x.numel() // cols, cols, eps)
            return y
        return F.layer_norm(x, (cols,), weight, bias, eps)


def softmax(x):
    """Softmax over the last dimension."""
    cols = x.shape[-1]
    with _timed("softmax", "reduce"):
        if "reduce" in USE_CUDA:
            x = x.contiguous()
            y = torch.empty_like(x)
            _check_float32("softmax", x)
            _launch("softmax", "launch_softmax", _ptr(x), _ptr(y), x.numel() // cols, cols)
            return y
        return torch.softmax(x, dim=-1)


# ----------------------------------------------------------------------------
# Task 3: linear layer (tiled GEMM)
# ----------------------------------------------------------------------------

def linear(x, weight, bias=None, act=None):
    """y = act(x @ weight^T + bias). x: [..., K], weight: [N, K], act: None or "gelu"."""
    K = x.shape[-1]
    N = weight.shape[0]
    x2 = x.reshape(-1, K)
    with _timed("linear", "gemm"):
        if "gemm" in USE_CUDA:
            x2 = x2.contiguous()
            _check_float32("linear", x2, weight, bias)
            y = torch.empty((x2.shape[0], N), device=x.device, dtype=torch.float32)
            _launch("linear", "launch_linear", _ptr(x2), _ptr(weight), _ptr(bias), _ptr(y),
                    x2.shape[0], N, K, 1 if act == "gelu" else 0)
        else:
            y = F.linear(x2, weight, bias)
            if act == "gelu":
                y = F.gelu(y)
    return y.reshape(*x.shape[:-1], N)


# ----------------------------------------------------------------------------
# Task 4: attention
# ----------------------------------------------------------------------------

def attention(q, k, v, causal):
    """q: [B, Sq, H, D]; k, v: [B, Sk, Hkv, D]. Returns [B, Sq, H, D]."""
    batch, seq_q, num_heads, head_dim = q.shape
    seq_k, num_kv_heads = k.shape[1], k.shape[2]
    scale = head_dim ** -0.5
    with _timed("attention", "attention"):
        if "attention" in USE_CUDA:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            _check_float32("attention", q, k, v)
            out = torch.empty_like(q)
            _launch("attention", "launch_attention", _ptr(q), _ptr(k), _ptr(v), _ptr(out), batch,
                    num_heads, num_kv_heads, seq_q, seq_k, head_dim, scale, int(causal))
            return out
        group = num_heads // num_kv_heads
        qh = q.transpose(1, 2)                                   # [B, H, Sq, D]
        kh = k.transpose(1, 2).repeat_interleave(group, dim=1)   # [B, H, Sk, D]
        vh = v.transpose(1, 2).repeat_interleave(group, dim=1)
        scores = (qh @ kh.transpose(-1, -2)) * scale             # [B, H, Sq, Sk]
        if causal:
            qpos = torch.arange(seq_q, device=q.device)[:, None] + (seq_k - seq_q)
            kpos = torch.arange(seq_k, device=q.device)[None, :]
            scores = scores.masked_fill(kpos > qpos, float("-inf"))
        return (torch.softmax(scores, dim=-1) @ vh).transpose(1, 2).contiguous()


# ----------------------------------------------------------------------------
# Provided kernels
# ----------------------------------------------------------------------------

def silu_mul(gate, up):
    """silu(gate) * up, element-wise."""
    with _timed("silu_mul", "provided"):
        if "provided" in USE_CUDA:
            gate, up = gate.contiguous(), up.contiguous()
            y = torch.empty_like(gate)
            _launch("silu_mul", "launch_silu_mul", _ptr(gate), _ptr(up), _ptr(y), gate.numel())
            return y
        return F.silu(gate) * up


def embedding(ids, table):
    """table[ids]. ids: int64 [...]; table: [vocab, dim]. Returns [..., dim]."""
    dim = table.shape[1]
    with _timed("embedding", "provided"):
        if "provided" in USE_CUDA:
            ids = ids.contiguous()
            out = torch.empty((*ids.shape, dim), device=table.device, dtype=torch.float32)
            _launch("embedding", "launch_embedding", _ptr(ids), _ptr(table), _ptr(out),
                    ids.numel(), dim)
            return out
        return table[ids]


def im2col_1d(x, kernel_size, stride, padding):
    """x: [B, L, C] -> windows [B * L_out, kernel_size * C] (see provided.cu)."""
    batch, length, channels = x.shape
    out_length = (length + 2 * padding - kernel_size) // stride + 1
    with _timed("im2col", "provided"):
        if "provided" in USE_CUDA:
            x = x.contiguous()
            cols = torch.empty((batch * out_length, kernel_size * channels), device=x.device,
                               dtype=torch.float32)
            _launch("im2col", "launch_im2col_1d", _ptr(x), _ptr(cols), batch, length, channels,
                    kernel_size, stride, padding, out_length)
            return cols, out_length
        xp = F.pad(x, (0, 0, padding, padding))                       # pad the time axis
        win = xp.unfold(1, kernel_size, stride)                       # [B, L_out, C, k]
        cols = win.permute(0, 1, 3, 2).reshape(batch * out_length, kernel_size * channels)
        return cols.contiguous(), out_length


def describe():
    """One line saying which groups use CUDA."""
    parts = [f"{g}={'cuda' if g in USE_CUDA else 'torch'}" for g in TASK_GROUPS + ("provided",)]
    return "  ".join(parts)


def wall_clock_ms(fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000
