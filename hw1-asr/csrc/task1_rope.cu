// Task 1: rotary position embedding (RoPE). Read docs/task1-rope.md first.
//
// Before attention, every query and key vector is rotated by an angle that depends
// on the token's position, so attention can tell tokens apart by where they are.
// One vector has head_dim numbers. RoPE pairs element i with element i + half,
// for i = 0 .. half-1 (half = rotary_dim / 2), and rotates each pair:
//
//     angle  = pos * inv_freq[i]
//     new_a  = a * cos(angle) - b * sin(angle)       a = x[i]
//     new_b  = b * cos(angle) + a * sin(angle)       b = x[i + half]
//
// Elements rotary_dim .. head_dim-1 are left unchanged (the audio encoder only
// rotates the first half of each head; the text decoder rotates all of it).
//
// Memory layout, straight out of the q/k projections (no transposes):
//     q: [batch, seq_len, num_q_heads, head_dim]
//     k: [batch, seq_len, num_k_heads, head_dim]
// Seen as a 2D array, each row is one head of one token: row r of q belongs to the
// token at position (r / num_q_heads) % seq_len.
//
// One launch rotates q AND k: one thread per pair, q's pairs first, then k's.
#include "asr_kernels.h"
#include "common.cuh"

constexpr int ROPE_THREADS = 256;

__global__ void rope_kernel(float* q, float* k, const float* inv_freq, int seq_len,
                            int num_q_heads, int num_k_heads, int q_rows, int k_rows,
                            int head_dim, int rotary_dim) {
  const int half = rotary_dim / 2;
  const int q_pairs = q_rows * half;
  const int k_pairs = k_rows * half;

  int idx = 0;  // this thread's pair number, counting q's pairs first, then k's
  // ======================== BLANK 1.1 =========================
  // Compute the global thread index (as in vector_add), store it in idx, and
  // return if it is past the last pair (q_pairs + k_pairs pairs in total).
  // YOUR CODE HERE
  // ============================================================

  float* x = q;               // the tensor this thread works on
  int num_heads = num_q_heads;
  // ======================== BLANK 1.2 =========================
  // Threads with idx >= q_pairs work on k: point x at k, use num_k_heads,
  // and subtract q_pairs from idx so it counts pairs inside k from 0.
  // YOUR CODE HERE
  // ============================================================

  int row = 0;  // which row (one head of one token) of x
  int i = 0;    // which pair inside that row, 0 .. half-1
  int pos = 0;  // position of the token this row belongs to
  // ======================== BLANK 1.3 =========================
  // Split idx into row and i (each row has `half` pairs), then compute pos
  // from row (see the layout notes at the top of the file).
  // YOUR CODE HERE
  // ============================================================

  float* v = x + static_cast<size_t>(row) * head_dim;  // start of this row
  // ======================== BLANK 1.4 =========================
  // Read the pair (v[i], v[i + half]), compute the angle, and write the
  // rotated pair back in place. Use cosf and sinf.
  // YOUR CODE HERE
  // ============================================================
}

ASR_API int launch_rope(float* q, float* k, const float* inv_freq, int batch, int seq_len,
                        int num_q_heads, int num_k_heads, int head_dim, int rotary_dim,
                        cudaStream_t stream) {
  int q_rows = batch * seq_len * num_q_heads;
  int k_rows = batch * seq_len * num_k_heads;
  int total_pairs = (q_rows + k_rows) * (rotary_dim / 2);
  int blocks = ceil_div(total_pairs, ROPE_THREADS);
  rope_kernel<<<blocks, ROPE_THREADS, 0, stream>>>(q, k, inv_freq, seq_len, num_q_heads,
                                                    num_k_heads, q_rows, k_rows, head_dim,
                                                    rotary_dim);
  RETURN_LAUNCH_ERROR();
}
