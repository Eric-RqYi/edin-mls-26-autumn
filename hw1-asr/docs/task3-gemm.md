# Task 3: the linear layer as a tiled GEMM

File: `csrc/task3_gemm.cu`. Test: `make test3`. Model check: `./run.sh --cuda task1,task2,task3`.

Every linear layer of the model computes

```
y[M, N] = act( x[M, K] * W[N, K]^T + bias[N] )
```

where `M` is the number of positions, `K` the number of input features and `N` the number of output features. The two convolutions and the projector also run through this kernel. Together they take most of the model's GPU time, so this is the kernel that decides how fast the transcription is.

| Layer | M | N | K |
|-------|--:|--:|--:|
| encoder `q_proj`, `k_proj`, `v_proj`, `o_proj` | 1500 | 1280 | 1280 |
| encoder `fc1` (+ GELU) | 1500 | 5120 | 1280 |
| encoder `fc2` | 1500 | 1280 | 5120 |
| decoder `q_proj`, `o_proj` | 59 to 71 | 2048 | 2048 |
| decoder `gate_proj`, `up_proj` | 59 to 71 | 6144 | 2048 |
| decoder `down_proj` | 59 to 71 | 2048 | 6144 |
| `lm_head` (last position only) | 1 | 59264 | 2048 |

## Start from the lecture's tiled GEMM

Open [cuda-tutorial/5-tiled-gemm/gemm.cu](../../cuda-tutorial/5-tiled-gemm/gemm.cu) next to this page. Its kernel is the starting point:

- A block of `TILE x TILE` threads (16 x 16 = 256) computes a `TILE x TILE` piece of the output. Thread `(ty, tx)` computes `y[row][col]` with `row = blockIdx.y * TILE + ty` and `col = blockIdx.x * TILE + tx`.
- The block walks along K in steps of `TILE`. In each step every thread loads one element of the x tile and one element of the weight tile into shared memory, the block waits at `__syncthreads()`, every thread adds `TILE` products to its sum, and the block waits again before the next step overwrites the tiles.
- Elements outside the matrices are loaded as 0, so the edge tiles need no special case, and no thread returns early because every thread must reach both barriers.

Why this is faster than one thread per output reading everything from global memory: every value loaded into shared memory is used by `TILE` threads. With `TILE = 16`, the block reads x and W from global memory 16 times less often.

## What is different here

### The weight is stored transposed

The lecture multiplies `A[M, K]` by `B[K, N]`. PyTorch stores a linear layer's weight as `W[N, K]`: one row per output feature. So output `y[m][n]` is the dot product of row `m` of x with row `n` of W:

```
y[m][n] = sum over k of  x[m][k] * W[n][k]
```

A block needs rows `blockIdx.y*TILE ...` of x and rows `blockIdx.x*TILE ...` of W, and it reads both along K. That changes how the W tile should be loaded. In memory, a row of W is contiguous, so the 16 threads that share a `ty` (half a warp) should read 16 neighbouring `k` of the same row of W. That is a coalesced read: the GPU fetches them as one 64-byte transaction. If instead neighbouring threads read neighbouring rows of W, their addresses are `K` floats apart and each needs its own transaction.

So the W tile is stored as `ws[n][k]` (row `n` of the tile holds part of row `n` of W), and the inner loop reads `ws[tx][q]`.

### Shared memory has banks

Shared memory is split into 32 banks; float number `i` lives in bank `i % 32`. The lanes of a warp can read 32 different banks in one go, but two lanes that need different addresses in the same bank have to wait for each other (a bank conflict). In the inner loop, the lanes read `ws[tx][q]` for `tx = 0..15`: same column, different rows. If a row is 16 floats long, those addresses are 16 apart and fall into only 2 banks. That is why the code declares `ws[TILE][TILE + 1]`: with rows of 17 floats, the 16 addresses land in 16 different banks. The `+1` is already in the code; Try this shows what it is worth.

### The epilogue

Before the result is stored, the kernel adds `bias[col]` (when `bias` is not `nullptr`) and applies GELU when `act == ACT_GELU`. The encoder's `fc1`, both convolutions and the projector's first layer are followed by GELU. Doing it here, on the value still in a register, saves a separate kernel that would read and write the whole output again. Combining operations into one kernel like this is called fusion. `gelu()` is in `common.cuh`.

## The blanks

3.1 Load one element of the x tile: `xs[ty][tx] = x[row][k0 + tx]`, or 0 if that is outside x. x is row-major, so `x[row][k]` is `x[row * K + k]`.

3.2 Load one element of the W tile so that afterwards `ws[i][j] = W[blockIdx.x*TILE + i][k0 + j]`. Choose which thread loads which element so that threads with neighbouring `tx` read neighbouring addresses.

<details><summary>Hint for 3.2</summary>

Thread `(ty, tx)` loads `ws[ty][tx]`. Which row of W is that, and which column? The row does not depend on `col`.

</details>

3.3 Wait for the tiles, add this step's `TILE` products to `acc`, and wait again. Which element of `ws` pairs with `xs[ty][q]`?

3.4 The epilogue: if `(row, col)` is inside y, add the bias, apply GELU if asked, and store.

## Test

```
make test3
```

Small cases are checked against a CPU loop, large ones against cuBLAS. For each case the test also times the naive kernel (one thread per output, no shared memory, in `tests/test_task3_gemm.cu`) and cuBLAS. When a case passes, a second line shows their speed on the same case, so you can see where your kernel stands:

```
encoder fc1 + GELU (1500x5120x1280)  PASS  max_err=...   ... ms   ... GFLOP/s
    for comparison: naive ... GFLOP/s, cuBLAS ... GFLOP/s (cuBLAS reference)
```

GFLOP/s counts `2 * M * N * K` floating-point operations (one multiply and one add per term) per second. Your tiled kernel should be several times faster than the naive one. cuBLAS is NVIDIA's hand-tuned library and is still several times faster than that; see the questions.

Common mistakes:

- Using `col` as the W row in blank 3.2. `col` follows `tx`, but the row of `ws` a thread fills is `ty`.
- Comparing `k0 + tx` with `M` or `N` instead of `K`.
- Returning early for threads outside y before the loop. The block then hangs or computes garbage, because those threads never reach `__syncthreads()`.
- Adding the bias inside the K loop.

## Try this

1. Set `TILE` to 8 and to 32. Which size is fastest for the encoder cases on your GPU? If you can, try a second kind of GPU as well; the answer is not the same everywhere. Then look at `lm_head` with `TILE = 32`. How many of the 32 rows of threads in a block have a real output row when `M = 1`, and what does that do to its time?
2. Change `ws[TILE][TILE + 1]` to `ws[TILE][TILE]`. How much slower are the large cases?
3. Load W the way the lecture loads B: thread `(ty, tx)` loads W row `blockIdx.x*TILE + tx`, column `k0 + ty`, into `ws[tx][ty]`. The result is the same; compare the time, especially for `lm_head`.
4. Delete the second `__syncthreads()`. Run `make test3`, then `compute-sanitizer --tool racecheck build/sm_XX/test_task3_gemm small`, which checks only the small cases. Which cases fail in the normal test, and which does the sanitizer complain about?
5. `lm_head` has `M = 1`, so only one of the 16 rows of threads in each block computes a real output. For a case like this, GB/s tells you more than GFLOP/s: the kernel must read the 485 MB weight matrix. Compute the GB/s of your kernel and of cuBLAS from the times the test prints.

## Questions for the report

1. For `M = N = K = 1024`, how many floats does the naive kernel load from global memory in total, and how many does your tiled kernel load? (Count requests, ignoring caches.)
2. Your kernel reaches only a fraction of cuBLAS's speed on the large cases. Name one thing cuBLAS does differently that explains part of the gap. You do not have to implement it.
3. How many bytes of global memory traffic does fusing GELU into the epilogue save for the encoder's `fc1`, compared with a separate GELU kernel that reads y and writes it back?
4. Why does every thread have to reach both `__syncthreads()`, even a thread whose `(row, col)` is outside y?
