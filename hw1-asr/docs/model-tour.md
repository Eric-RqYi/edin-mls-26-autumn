# A tour of the model

What GLM-ASR-Nano does with the recording, and where your kernels run. The code is in `glm_asr_cuda/model.py`.

## The whole model

```
+------------------------------+
| recording, 3.5 s             |
+------------------------------+
               |  Hugging Face processor, on the CPU
               v
+------------------------------+
| mel spectrogram              |   128 x 3000
+------------------------------+
               |
               v
+------------------------------+
| 2 convolutions               |  1500 x 1280   im2col + task 3
+------------------------------+
| audio encoder, 32 layers     |  1500 x 1280   tasks 1-4
+------------------------------+
| projector                    |    44 x 2048   task 3
+------------------------------+
               |  44 audio positions go into the text prompt
               v
+------------------------------+
| text decoder, 28 layers      |    59 x 2048   tasks 1-4   <-----+
+------------------------------+                                  |
| lm_head, last position only  |  1 x 59264     task 3            |
+------------------------------+                                  |
| softmax                      |  1 x 59264     task 2            |
+------------------------------+                                  |
               |  the most likely token                           |
               +-----> append it to the text and run again -------+
                       (13 times; the 13th token means "end")
```

The numbers are the array shapes: positions x features.

- The processor pads every recording to 30 s, so the encoder sees 1500 positions, although only about 175 hold speech.
- The decoder recomputes the whole text on every run (there is no KV cache), so its input grows from 59 to 71 positions.

## One layer

Encoder and decoder layers have the same shape. `(+)` adds the layer's input back to its output.

```
            h
            |
            +-------------------+
            v                   |
    +---------------+           |
    | norm          |  task 2   |
    +---------------+           |
    | q, k, v       |  task 3   |
    +---------------+           |
    | RoPE on q, k  |  task 1   |
    +---------------+           |
    | attention     |  task 4   |
    +---------------+           |
    | out linear    |  task 3   |
    +---------------+           |
            |                   |
            v                   |
           (+) <----------------+
            |
            +-------------------+
            v                   |
    +---------------+           |
    | norm          |  task 2   |
    +---------------+           |
    | MLP           |  task 3   |
    +---------------+           |
            |                   |
            v                   |
           (+) <----------------+
            |
            v
     to the next layer
```

| Operation | What it does | Kernel |
|-----------|--------------|--------|
| linear | `y = x W^T + b`, with `W` stored as `[out_features, in_features]`. Almost all of the arithmetic. | task 3 |
| norm | Rescales each row to a standard size, then multiplies by a weight per column. LayerNorm in the encoder, RMSNorm in the decoder. | task 2 |
| RoPE | Rotates pairs of numbers in `q` and `k` by angles that grow with the position. | task 1 |
| attention | Each position takes an average of the `v` vectors, weighted by the softmax of `q . k`. | task 4 |
| MLP | Encoder: `fc2(GELU(fc1(x)))`. Decoder: `down(SiLU(gate(x)) * up(x))`; `silu_mul` is provided. | task 3 |
| softmax | `exp(x_i - max) / sum_j exp(x_j - max)` | task 2 |

## Data layout

Every array is float32 and row-major, with a batch dimension of 1 in front: `[batch, seq, hidden]`. Element `(s, c)` of a `[seq, hidden]` array is at `s * hidden + c`.

For attention the same memory is read as `[batch, seq, heads, head_dim]`, without moving data: 20 heads of 64 in the encoder, 16 heads of 128 in the decoder.

- GQA: the decoder has only 4 key/value heads. Query heads 0 to 3 use key/value head 0, heads 4 to 7 use head 1, and so on.
- Causal: in the decoder, a token only looks at itself and the tokens before it. The encoder is not causal.

## Where your kernels run

Calls for one transcription of the test recording:

| Operation | Kernel | Calls | Typical shapes |
|-----------|--------|------:|----------------|
| `linear` | task 3 | 2757 | encoder `1500 x 1280 x 1280`, `1500 x 5120 x 1280`; decoder `59 x 2048 x 2048`, `59 x 6144 x 2048`; `lm_head` `1 x 59264 x 2048` |
| `attention` | task 4 | 396 | encoder 20 heads, 1500 x 1500, head_dim 64; decoder 16/4 heads, causal, about 64 x 64, head_dim 128 |
| `rope` | task 1 | 396 | encoder rotates 32 of 64 numbers per head, decoder all 128 |
| `rmsnorm` | task 2 | 741 | 59 to 71 rows of 2048 |
| `layernorm` | task 2 (given) | 65 | 1500 rows of 1280 |
| `softmax` | task 2 | 13 | 1 row of 59,264 |
| `silu_mul`, `embedding`, `im2col` | provided | | |

The linear layers take most of the GPU time: they do almost all of the roughly 4.5 trillion floating-point operations.

## The convolutions

Each convolution slides a window of 3 time steps along its input. `im2col` (provided) copies every window into one row of a matrix, so the convolution becomes one matrix multiply that your task 3 GEMM computes.
