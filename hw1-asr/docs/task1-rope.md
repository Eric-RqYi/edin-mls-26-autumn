# Task 1: rotary position embedding (RoPE)

File: `csrc/task1_rope.cu`. Test: `make test1`. Model check: `./run.sh --cuda task1`.

This is the warm-up. It is an element-wise kernel like `vector_add`: every thread does a few independent multiplications on its own data. The work is in the index arithmetic: working out which numbers a thread owns when the data is a 4D array.

## What RoPE computes

Attention compares a query vector with key vectors (see [model-tour.md](model-tour.md)). On its own, that comparison cannot tell whether two tokens are next to each other or far apart. RoPE fixes this by rotating the vectors before the comparison, by an angle proportional to the token's position.

A vector has `head_dim` numbers. RoPE treats the first `rotary_dim` of them as `half = rotary_dim / 2` pairs: element `i` is paired with element `i + half`. Each pair is a point in the plane, and it is rotated by

```
angle = pos * inv_freq[i]
a' = a * cos(angle) - b * sin(angle)        a = v[i]
b' = b * cos(angle) + a * sin(angle)        b = v[i + half]
```

`pos` is the token's position (0, 1, 2, ...) and `inv_freq[i]` is a fixed frequency per pair, computed on the CPU and passed in. Elements from `rotary_dim` to `head_dim - 1` are not touched. The decoder rotates the whole vector (`rotary_dim = head_dim = 128`); the audio encoder rotates only the first half (`rotary_dim = 32` of `head_dim = 64`).

A small example: `head_dim = rotary_dim = 4`, so `half = 2` and the pairs are `(v[0], v[2])` and `(v[1], v[3])`. With `inv_freq = [1.0, 0.01]`, a token at position 3 has its first pair rotated by 3 radians and its second pair by 0.03 radians. The vector `[1, 0, 0, 0]` becomes `[cos 3, 0, sin 3, 0] = [-0.990, 0, 0.141, 0]`.

## The layout, and which thread does what

`q` comes straight out of a linear layer, so it is `[batch, seq_len, num_q_heads, head_dim]` in row-major order. Treat it as a 2D array with `head_dim` columns: each row is one head of one token, and the rows go head by head, token by token:

```
row 0: token 0, head 0
row 1: token 0, head 1
row 2: token 1, head 0          (here num_heads = 2)
row 3: token 1, head 1
...
```

So row `r` belongs to the token at position `(r / num_heads) % seq_len`. The `% seq_len` matters when `batch > 1`: the positions start again at 0 for every sequence in the batch.

`k` has the same layout with `num_k_heads` heads. In the decoder `q` has 16 heads and `k` has 4.

The kernel rotates `q` and `k` in one launch, with one thread per pair. The pairs of `q` come first and then the pairs of `k`. For `batch = 1, seq_len = 3, num_q_heads = 2, num_k_heads = 1, half = 2`:

```
thread (idx)   0  1 |  2  3 |  4  5 |  6  7 |  8  9 | 10 11 | 12 13 | 14 15 | 16 17
tensor         q    |  q    |  q    |  q    |  q    |  q    |  k    |  k    |  k
row            0    |  1    |  2    |  3    |  4    |  5    |  0    |  1    |  2
pair i         0  1 |  0  1 |  0  1 |  0  1 |  0  1 |  0  1 |  0  1 |  0  1 |  0  1
```

The launcher starts `ceil(total_pairs / 256)` blocks of `ROPE_THREADS = 256` threads, as in `vector_add`.

## The blanks

The variables each blank sets are declared just above it, so the file compiles while blanks are empty.

### 1.1: thread index and bounds check

Compute the global index of the thread and stop threads that have no pair. The launcher rounds the number of threads up to a multiple of the block size, so the last block has spare threads.

<details><summary>Hint</summary>

Exactly as in `vector_add`: `blockIdx.x * blockDim.x + threadIdx.x`, and `return` if it is not smaller than `q_pairs + k_pairs`. An early `return` is fine here because this kernel has no `__syncthreads()`.

</details>

### 1.2: q or k?

Threads with `idx < q_pairs` work on `q`. The others work on `k`: point `x` at `k`, use `num_k_heads`, and subtract `q_pairs` so `idx` counts the pairs of `k` from 0.

### 1.3: row, pair and position

Split `idx` into the row and the pair inside the row (each row holds `half` pairs), then get the token position from the row.

<details><summary>Hint</summary>

Division and remainder by `half` give the row and the pair. For the position, use the formula from the layout section with the `num_heads` of the tensor you are working on.

</details>

### 1.4: rotate

`v` already points at the start of the row. Read the pair, compute the angle, and write both rotated values back. Read both values before you write either one: `b'` needs the old `a`.

## Test

```
make test1
```

Every case prints one line. When the kernel is right, the line says `PASS` and shows the largest error, the time of one call and the speed in GB/s, and the test ends with `Task 1 (RoPE): ALL PASS`.

The GB/s column is the number of bytes the kernel reads and writes, divided by its time. RoPE does very little arithmetic per byte, so the best it can do is move its data at the speed of GPU memory; the "peak" is that speed. A kernel like this is called memory bound.

Then run `./run.sh --cuda task1`; the transcription should still pass.

Common mistakes:

- Using `num_q_heads` for `k`. The decoder test catches this (16 heads against 4), the encoder test does not.
- Forgetting `% seq_len`. Only the cases with `batch > 1` catch this.
- Rotating `rotary_dim` pairs instead of `rotary_dim / 2`.
- Writing `v[i]` before reading it for `b'`.

## Try this

1. Change `ROPE_THREADS` to 32, 64, 128, 512 and 1024 and compare the GB/s of the audio encoder case. Which sizes are slow, and why might very small blocks hurt?
2. The tiny cases reach almost none of the peak, and their time barely changes with the size. What is that time made of?
3. Build with fast but less accurate maths: `make clean && make test1 NVCCFLAGS="-O3 -std=c++17 --use_fast_math"`. `cosf` and `sinf` become fast hardware approximations. Which case fails, and why that one? (Look at the largest angle each case uses.) Run `make clean` afterwards to go back to the normal flags.

## Questions for the report

1. For the audio encoder case, how many bytes does the kernel read and write in total? Which fraction of your GPU's peak bandwidth did you reach?
2. The kernel computes `cos` and `sin` for every pair. The alternative is to compute a `[positions, half]` table of them once and read it in the kernel. Would the table make the kernel faster or slower? Count the bytes.
3. Why is it worth rotating `q` and `k` in one launch instead of two? The model calls this function 396 times per transcription.
