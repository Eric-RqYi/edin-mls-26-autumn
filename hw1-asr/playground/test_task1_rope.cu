// Task 1 test: RoPE on the shapes the model uses, checked against a CPU version.
// Build and run with:  make test1
#include "test_util.h"

struct RopeCase {
  const char* name;
  int batch, seq_len, num_q_heads, num_k_heads, head_dim, rotary_dim;
};

static std::vector<float> make_inv_freq(int rotary_dim, double base) {
  std::vector<float> inv(rotary_dim / 2);
  for (int i = 0; i < rotary_dim / 2; ++i)
    inv[i] = static_cast<float>(1.0 / std::pow(base, 2.0 * i / rotary_dim));
  return inv;
}

// CPU reference, written as plainly as possible.
static std::vector<double> rope_cpu(const std::vector<float>& x, const std::vector<float>& inv,
                                    int rows, int num_heads, int seq_len, int head_dim,
                                    int rotary_dim) {
  std::vector<double> y(x.begin(), x.end());
  int half = rotary_dim / 2;
  for (int r = 0; r < rows; ++r) {
    int pos = (r / num_heads) % seq_len;
    for (int i = 0; i < half; ++i) {
      float angle = pos * inv[i];  // same float rounding as the GPU
      double a = x[static_cast<size_t>(r) * head_dim + i];
      double b = x[static_cast<size_t>(r) * head_dim + i + half];
      y[static_cast<size_t>(r) * head_dim + i] = a * std::cos(angle) - b * std::sin(angle);
      y[static_cast<size_t>(r) * head_dim + i + half] = b * std::cos(angle) + a * std::sin(angle);
    }
  }
  return y;
}

static bool run_case(const RopeCase& t, double peak_gbs) {
  int q_rows = t.batch * t.seq_len * t.num_q_heads;
  int k_rows = t.batch * t.seq_len * t.num_k_heads;
  auto q = random_vector(static_cast<size_t>(q_rows) * t.head_dim, 1);
  auto k = random_vector(static_cast<size_t>(k_rows) * t.head_dim, 2);
  auto inv = make_inv_freq(t.rotary_dim, 10000.0);

  DeviceArray dq(q), dk(k), dinv(inv);
  CHECK(launch_rope(dq.ptr, dk.ptr, dinv.ptr, t.batch, t.seq_len, t.num_q_heads,
                    t.num_k_heads, t.head_dim, t.rotary_dim, 0));
  CHECK(cudaDeviceSynchronize());

  auto ref_q = rope_cpu(q, inv, q_rows, t.num_q_heads, t.seq_len, t.head_dim, t.rotary_dim);
  auto ref_k = rope_cpu(k, inv, k_rows, t.num_k_heads, t.seq_len, t.head_dim, t.rotary_dim);
  Comparison c = compare(dq.download(), ref_q, 1e-5, 1e-4);
  Comparison ck = compare(dk.download(), ref_k, 1e-5, 1e-4);
  if (ck.wrong && !c.wrong) c.first_wrong = q.size() + ck.first_wrong;  // index into q then k
  c.wrong += ck.wrong;
  c.checked += ck.checked;
  c.max_err = std::max(c.max_err, ck.max_err);

  // Timing. RoPE works in place, so the data keeps rotating; that is fine for timing.
  float ms = time_ms([&] {
    return launch_rope(dq.ptr, dk.ptr, dinv.ptr, t.batch, t.seq_len, t.num_q_heads,
                       t.num_k_heads, t.head_dim, t.rotary_dim, 0);
  });
  double bytes = 2.0 * (q_rows + k_rows) * t.rotary_dim * sizeof(float);  // read + write
  double gbs = bytes / (ms * 1e-3) / 1e9;
  return report(t.name, c, ms, fmt("%7.1f GB/s (%2.0f%% of peak)", gbs, 100.0 * gbs / peak_gbs));
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  double peak = print_gpu();
  const RopeCase cases[] = {
      // name                            B  seq  Hq  Hk  dim  rotary
      {"tiny (2 x 3 tokens, Hq=3, Hk=1)", 2, 3, 3, 1, 8, 8},
      {"odd sizes, partial rotation", 3, 7, 5, 1, 32, 16},
      {"audio encoder (partial, 32/64)", 1, 1500, 20, 20, 64, 32},
      {"text decoder (GQA 16/4, full)", 1, 64, 16, 4, 128, 128},
  };
  bool ok = true;
  for (const auto& t : cases) {
    if (g_small_only && double(t.batch) * t.seq_len * t.num_q_heads * t.head_dim > 2e5) continue;
    ok = run_case(t, peak) && ok;
  }
  return finish("Task 1 (RoPE)", ok);
}
