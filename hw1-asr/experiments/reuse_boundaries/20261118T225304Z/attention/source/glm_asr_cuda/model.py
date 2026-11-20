"""GLM-ASR-Nano, written so that every heavy operation goes through ops.py.

PyTorch only holds the tensors and does the glue: reshapes, residual additions,
concatenation and picking the most likely token. The arithmetic runs in your
kernels (or in the PyTorch reference, if you switch a group off).

Activations keep the layout the linear layers produce, [batch, seq, features];
for attention that is viewed as [batch, seq, heads, head_dim], so no transposes
are needed. See docs/model-tour.md for a walk through the model.

You do not need to change this file.
"""

import json
from dataclasses import dataclass

import torch

from . import ops

MODEL_NAME = "zai-org/GLM-ASR-Nano-2512"


@dataclass
class Config:
    # audio encoder
    audio_hidden: int = 1280
    audio_heads: int = 20
    audio_layers: int = 32
    audio_ffn: int = 5120
    audio_rotary_factor: float = 0.5
    audio_rope_base: float = 10000.0
    # projector: 4 audio frames are concatenated, then 5120 -> 4096 -> 2048
    pool: int = 4
    # text decoder (a Llama-style language model)
    text_hidden: int = 2048
    text_heads: int = 16
    text_kv_heads: int = 4
    text_layers: int = 28
    text_ffn: int = 6144
    text_rope_base: float = 10000.0
    vocab: int = 59264
    rms_eps: float = 1e-5
    ln_eps: float = 1e-5
    eos_token_ids: tuple = (59246, 59253, 59255)
    audio_token_id: int = 59260

    @classmethod
    def from_hf(cls, hf):
        a, t = hf.audio_config, hf.text_config
        eos = t.eos_token_id if isinstance(t.eos_token_id, (list, tuple)) else [t.eos_token_id]
        return cls(
            audio_hidden=a.hidden_size, audio_heads=a.num_attention_heads,
            audio_layers=a.num_hidden_layers, audio_ffn=a.intermediate_size,
            audio_rotary_factor=a.rope_parameters.get("partial_rotary_factor", 1.0),
            audio_rope_base=a.rope_parameters["rope_theta"],
            text_hidden=t.hidden_size, text_heads=t.num_attention_heads,
            text_kv_heads=t.num_key_value_heads, text_layers=t.num_hidden_layers,
            text_ffn=t.intermediate_size, text_rope_base=t.rope_parameters["rope_theta"],
            vocab=t.vocab_size, rms_eps=t.rms_norm_eps, eos_token_ids=tuple(eos),
            audio_token_id=hf.audio_token_id,
        )


def inv_freq(head_dim, rotary_factor, base, device):
    rotary_dim = int(head_dim * rotary_factor)
    exps = torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(torch.float32) / rotary_dim
    return (1.0 / (base ** exps)).to(device), rotary_dim


class GlmAsr:
    def __init__(self, config, weights, device):
        self.c = config
        self.w = weights  # name -> float32 CUDA tensor, names as in the HF checkpoint
        c = config
        self.audio_head_dim = c.audio_hidden // c.audio_heads
        self.text_head_dim = c.text_hidden // c.text_heads
        self.audio_inv_freq, self.audio_rotary_dim = inv_freq(
            self.audio_head_dim, c.audio_rotary_factor, c.audio_rope_base, device)
        self.text_inv_freq, self.text_rotary_dim = inv_freq(
            self.text_head_dim, 1.0, c.text_rope_base, device)

    # ---------------- audio encoder ----------------

    def conv1d_gelu(self, x, name, stride):
        """GELU(Conv1d(kernel 3, padding 1)) on x: [B, L, C_in] -> [B, L_out, C_out]."""
        cols, out_len = ops.im2col_1d(x, kernel_size=3, stride=stride, padding=1)
        y = ops.linear(cols, self.w[name + ".weight"], self.w[name + ".bias"], act="gelu")
        return y.reshape(x.shape[0], out_len, -1)

    def encoder_layer(self, h, i):
        c, w, p = self.c, self.w, f"audio_tower.layers.{i}."
        B, S, _ = h.shape
        x = ops.layernorm(h, w[p + "input_layernorm.weight"], w[p + "input_layernorm.bias"], c.ln_eps)
        q = ops.linear(x, w[p + "self_attn.q_proj.weight"], w[p + "self_attn.q_proj.bias"])
        k = ops.linear(x, w[p + "self_attn.k_proj.weight"])
        v = ops.linear(x, w[p + "self_attn.v_proj.weight"], w[p + "self_attn.v_proj.bias"])
        q = q.view(B, S, c.audio_heads, self.audio_head_dim)
        k = k.view(B, S, c.audio_heads, self.audio_head_dim)
        v = v.view(B, S, c.audio_heads, self.audio_head_dim)
        ops.rope_(q, k, self.audio_inv_freq, self.audio_rotary_dim)
        a = ops.attention(q, k, v, causal=False).view(B, S, c.audio_hidden)
        h = h + ops.linear(a, w[p + "self_attn.o_proj.weight"], w[p + "self_attn.o_proj.bias"])

        x = ops.layernorm(h, w[p + "post_attention_layernorm.weight"],
                          w[p + "post_attention_layernorm.bias"], c.ln_eps)
        x = ops.linear(x, w[p + "mlp.fc1.weight"], w[p + "mlp.fc1.bias"], act="gelu")
        return h + ops.linear(x, w[p + "mlp.fc2.weight"], w[p + "mlp.fc2.bias"])

    def encode_audio(self, features, features_mask):
        """features: [B, 128 mel bins, T] -> audio embeddings [num_audio_tokens, text_hidden]."""
        x = features.transpose(1, 2).contiguous()               # [B, T, 128]
        x = self.conv1d_gelu(x, "audio_tower.conv1", stride=1)  # [B, T, 1280]
        x = self.conv1d_gelu(x, "audio_tower.conv2", stride=2)  # [B, T/2, 1280]
        for i in range(self.c.audio_layers):
            x = self.encoder_layer(x, i)
        x = ops.layernorm(x, self.w["audio_tower.norm.weight"], self.w["audio_tower.norm.bias"],
                          self.c.ln_eps)

        # Projector: concatenate groups of 4 frames, then a two-layer MLP.
        B, S, H = x.shape
        x = x[:, : S // self.c.pool * self.c.pool].reshape(B, S // self.c.pool, H * self.c.pool)
        x = ops.linear(x, self.w["multi_modal_projector.linear_1.weight"],
                       self.w["multi_modal_projector.linear_1.bias"], act="gelu")
        x = ops.linear(x, self.w["multi_modal_projector.linear_2.weight"],
                       self.w["multi_modal_projector.linear_2.bias"])

        # Keep only the frames that came from real audio, not from padding.
        length = features_mask.sum(-1)
        for stride in (1, 2):  # the two convolutions
            length = (length + 2 - 2 - 1) // stride + 1
        n_tokens = (length - self.c.pool) // self.c.pool + 1
        keep = torch.arange(x.shape[1], device=x.device)[None, :] < n_tokens[:, None]
        return x[keep]

    # ---------------- text decoder ----------------

    def decoder_layer(self, h, i):
        c, w, p = self.c, self.w, f"language_model.model.layers.{i}."
        B, S, _ = h.shape
        x = ops.rmsnorm(h, w[p + "input_layernorm.weight"], c.rms_eps)
        q = ops.linear(x, w[p + "self_attn.q_proj.weight"]).view(B, S, c.text_heads, self.text_head_dim)
        k = ops.linear(x, w[p + "self_attn.k_proj.weight"]).view(B, S, c.text_kv_heads, self.text_head_dim)
        v = ops.linear(x, w[p + "self_attn.v_proj.weight"]).view(B, S, c.text_kv_heads, self.text_head_dim)
        ops.rope_(q, k, self.text_inv_freq, self.text_rotary_dim)
        a = ops.attention(q, k, v, causal=True).view(B, S, c.text_hidden)
        h = h + ops.linear(a, w[p + "self_attn.o_proj.weight"])

        x = ops.rmsnorm(h, w[p + "post_attention_layernorm.weight"], c.rms_eps)
        gate = ops.linear(x, w[p + "mlp.gate_proj.weight"])
        up = ops.linear(x, w[p + "mlp.up_proj.weight"])
        return h + ops.linear(ops.silu_mul(gate, up), w[p + "mlp.down_proj.weight"])

    def next_token_probs(self, embeds):
        """embeds: [B, S, hidden] -> probabilities of the next token [B, vocab]."""
        h = embeds
        for i in range(self.c.text_layers):
            h = self.decoder_layer(h, i)
        last = ops.rmsnorm(h[:, -1], self.w["language_model.model.norm.weight"], self.c.rms_eps)
        logits = ops.linear(last, self.w["language_model.lm_head.weight"])  # only the last position
        return ops.softmax(logits)

    # ---------------- generation ----------------

    @torch.no_grad()
    def generate(self, features, input_ids, features_mask, max_new_tokens=100):
        """Greedy decoding. Returns (new token ids, probability of each chosen token)."""
        audio = self.encode_audio(features, features_mask)          # [n_audio, hidden]
        embeds = ops.embedding(input_ids, self.w["language_model.model.embed_tokens.weight"])
        is_audio = (input_ids[0] == self.c.audio_token_id).nonzero().flatten()
        first, last = int(is_audio[0]), int(is_audio[-1])
        # The prompt has one <|pad|> placeholder per audio embedding; swap them in.
        embeds = torch.cat([embeds[:, :first], audio[None], embeds[:, last + 1:]], dim=1)

        tokens, probs = [], []
        for _ in range(max_new_tokens):
            # No KV cache: the whole sequence is recomputed for every new token.
            p = self.next_token_probs(embeds)
            best = torch.argmax(p, dim=-1)                          # greedy choice
            tokens.append(int(best[0]))
            probs.append(float(p[0, best[0]]))
            if tokens[-1] in self.c.eos_token_ids:
                break
            new = ops.embedding(best[:, None], self.w["language_model.model.embed_tokens.weight"])
            embeds = torch.cat([embeds, new], dim=1)
        return tokens, probs


SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def read_safetensors(path):
    """Yield (name, CPU tensor) for every tensor of a .safetensors file, in file order.

    The safetensors library maps the file into memory and reads it in small pieces in
    name order. On the cluster's network file system that took more than 20 minutes on
    the H200 node, against about a minute for reading the file front to back in large
    pieces, as this function does.
    """
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
        header.pop("__metadata__", None)
        data_start = 8 + header_size
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            data = bytearray(end - begin)
            f.readinto(data)
            dtype = SAFETENSORS_DTYPES[info["dtype"]]
            yield name, torch.frombuffer(data, dtype=dtype).reshape(info["shape"])


def load_model(device="cuda"):
    """Read the checkpoint and the processor. Weights are converted to float32 on the GPU."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoProcessor

    config = Config.from_hf(AutoConfig.from_pretrained(MODEL_NAME))
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    path = hf_hub_download(MODEL_NAME, "model.safetensors")
    weights = {}
    for name, t in read_safetensors(path):
        t = t.to(device=device, dtype=torch.float32)
        if name.endswith(("conv1.weight", "conv2.weight")):
            # [out, in, kernel] -> [out, kernel * in], the column order of im2col_1d
            t = t.permute(0, 2, 1).reshape(t.shape[0], -1)
        weights[name] = t.contiguous()
    return GlmAsr(config, weights, torch.device(device)), processor
