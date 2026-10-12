# A tour of the model

This page explains what GLM-ASR-Nano does with the recording, step by step, and where each of your kernels is used. You do not need any background in machine learning. The code is in `glm_asr_cuda/model.py` and is short enough to read alongside this page.

## Numbers in, numbers out

A neural network is a long chain of simple operations on arrays of floats. The model's "knowledge" is 2.3 billion fixed numbers, the parameters or weights, which were learned before we downloaded them. Running the model means pushing the input through the chain. Almost every step is one of a handful of operations, and those are the kernels you write.

```
recording (56,080 samples, 3.5 s)
    |  feature extraction (on the CPU, by the Hugging Face processor)
    v
mel spectrogram [128 x 3000]          how loud each of 128 frequency bands is, every 10 ms
    |  2 convolutions (im2col + task 3 GEMM)
    v
audio encoder, 32 layers               [1500 positions x 1280]     tasks 1, 2, 3, 4
    |  projector: join 4 positions, 2 linear layers (task 3)
    v
44 audio embeddings                    [44 x 2048]
    |  placed into a text prompt: "<audio> Please transcribe this audio into text"
    v
text decoder, 28 layers                [59 tokens x 2048]           tasks 1, 2, 3, 4
    |  last position -> lm_head (task 3) -> softmax (task 2)
    v
probability of each of 59,264 tokens; the most likely one is the next piece of text
```

The decoder produces one token (a word or a piece of a word) at a time: "Con", "cord", " returned", ... After each token, the token is appended to the input and the decoder runs again. For this recording it runs 13 times; the 13th token is an end marker. Our implementation keeps nothing from one run to the next (it has no "KV cache"), so every run recomputes all 59 to 71 positions.

The Hugging Face processor pads every recording with silence to 30 seconds, which is why the encoder works on 1500 positions although only about 175 of them hold speech. We run the model on exactly what the processor produces.

## The shape of the data

Between the steps, the data is a 2D array: one row per position (an audio frame or a text token), one column per feature. In the code it has a leading batch dimension of 1: `[batch, seq, hidden]`, for example `[1, 1500, 1280]` in the encoder. Every array is float32 and stored row by row (row-major), so element `(s, c)` of a `[seq, hidden]` array is at index `s * hidden + c`.

## One transformer layer

The encoder and the decoder are both stacks of the same kind of layer. One layer does this to the `[seq, hidden]` array `h`:

```
x = norm(h)                         LayerNorm (encoder) or RMSNorm (decoder)     task 2
q, k, v = x Wq^T, x Wk^T, x Wv^T    three linear layers                          task 3
rotate q and k by position          RoPE                                         task 1
a = attention(q, k, v)                                                           task 4
h = h + a Wo^T                      a linear layer, then add to the input        task 3
x = norm(h)                                                                      task 2
h = h + MLP(x)                      linear, activation, linear                   task 3
```

What each operation is for, in one or two sentences:

- A linear layer multiplies every row by a weight matrix: `y = x W^T + b`. It is where nearly all the arithmetic is. W has one row per output feature, so `W` is `[out_features, in_features]`. Task 3.
- A normalisation rescales each row so its values have a standard size. LayerNorm subtracts the row's mean and divides by its standard deviation; RMSNorm only divides by the root mean square. Both then multiply by a learned weight per column. Task 2.
- Attention lets each position collect information from the other positions. The query vector `q` of a position is compared with the key vectors `k` of all positions; the scores go through a softmax; the output is the weighted average of the value vectors `v`. Task 4.
- RoPE rotates pairs of numbers inside `q` and `k` by angles that grow with the position, so attention can tell which tokens are near each other. Task 1.
- The MLP is two or three linear layers with an activation function between them. The encoder uses `fc2(GELU(fc1(x)))`. The decoder uses `down(SiLU(gate(x)) * up(x))`, called SwiGLU; the `SiLU(g) * u` part is the provided `silu_mul` kernel.
- Softmax turns a row of scores into probabilities: `exp(x_i - max) / sum_j exp(x_j - max)`. It is used inside attention and, at the very end, over the 59,264 possible next tokens. Task 2.

Attention is split into heads: the 1280 features of an encoder position are viewed as 20 heads of 64 numbers each, and each head does its own attention. The `[seq, 1280]` array is viewed as `[seq, 20, 64]` (`[batch, seq, heads, head_dim]`) without moving any data, which is why your kernels see that layout. The decoder has 16 query heads but only 4 key/value heads: heads 0 to 3 share key/value head 0, heads 4 to 7 share head 1, and so on. This is grouped-query attention (GQA), and it saves memory.

The decoder's attention is causal: a token may only look at itself and the tokens before it, since the text after it does not exist yet when it is generated. The encoder sees the whole recording at once, so its attention is not causal.

## Where your kernels run

Counts for one transcription of the test recording:

| Operation | Kernel | Where | Calls | Typical shapes |
|-----------|--------|-------|------:|----------------|
| `linear` | task 3 | every linear layer, both convolutions, the projector, `lm_head` | 2757 | encoder `1500 x 1280 x 1280` and `1500 x 5120 x 1280`; decoder `59 x 2048 x 2048`, `59 x 6144 x 2048`; `lm_head` `1 x 59264 x 2048` |
| `attention` | task 4 | every layer | 396 | encoder 20 heads, 1500 x 1500, head_dim 64; decoder 16/4 heads, causal, about 64 x 64, head_dim 128 |
| `rope` | task 1 | every layer | 396 | encoder rotates 32 of 64 numbers per head; decoder all 128 |
| `rmsnorm` | task 2 | decoder | 741 | 59 to 71 rows of 2048 |
| `layernorm` | task 2 (given) | encoder | 65 | 1500 rows of 1280 |
| `softmax` | task 2 | end of each decoder run | 13 | 1 row of 59,264 |
| `silu_mul`, `embedding`, `im2col` | provided | | | |

The linear layers take most of the GPU time, because they do almost all of the roughly 4.5 trillion floating-point operations of a transcription. Attention is most of the rest, mainly the encoder's 1500 x 1500 attention in 32 layers.

## The convolutions

The two convolutions at the start of the encoder slide a window of 3 time steps along the spectrogram. `im2col` (in `provided.cu`) copies every window into one row of a matrix, so the whole convolution becomes one matrix multiply, `windows x W^T + b`, which your task 3 GEMM computes. GPU libraries often handle convolutions this way.
