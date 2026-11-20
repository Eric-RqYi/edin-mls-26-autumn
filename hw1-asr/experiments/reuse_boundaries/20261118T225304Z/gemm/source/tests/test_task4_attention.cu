// Task 4 test: attention on the model's shapes, checked against a CPU version.
// For the big cases the CPU only checks a sample of the query rows.
// Build and run with:  make test4
#include "test_util.h"

struct AttnCase {
  const char* name;
  int batch, num_heads, num_kv_heads, seq_q, seq_k, head_dim;
  bool causal;
  float q_range;  // queries are drawn from [-q_range, q_range]; larger means peakier softmax
};

// Output of one query row (b, h, i), computed in double on the CPU.
static void attention_row_cpu(const AttnCase& t, const std::vector<float>& q,
                              const std::vector<float>& k, const std::vector<float>& v, int b,
                              int h, int i, std::vector<double>& out) {
  int D = t.head_dim, kvh = h / (t.num_heads / t.num_kv_heads);
  double scale = 1.0 / std::sqrt(static_cast<double>(D));
  int last = t.causal ? std::min(t.seq_k - 1, i + t.seq_k - t.seq_q) : t.seq_k - 1;
  const float* qr = &q[((static_cast<size_t>(b) * t.seq_q + i) * t.num_heads + h) * D];
  std::vector<double> s(last + 1);
  double m = -INFINITY, sum = 0;
  for (int j = 0; j <= last; ++j) {
    const float* kr = &k[((static_cast<size_t>(b) * t.seq_k + j) * t.num_kv_heads + kvh) * D];
    double dot = 0;
    for (int d = 0; d < D; ++d) dot += double(qr[d]) * kr[d];
    s[j] = dot * scale;
    m = std::max(m, s[j]);
  }
  for (int j = 0; j <= last; ++j) sum += (s[j] = std::exp(s[j] - m));
  out.assign(D, 0.0);
  for (int j = 0; j <= last; ++j) {
    const float* vr = &v[((static_cast<size_t>(b) * t.seq_k + j) * t.num_kv_heads + kvh) * D];
    for (int d = 0; d < D; ++d) out[d] += s[j] / sum * vr[d];
  }
}

static bool run_case(const AttnCase& t) {
  size_t nq = static_cast<size_t>(t.batch) * t.seq_q * t.num_heads * t.head_dim;
  size_t nk = static_cast<size_t>(t.batch) * t.seq_k * t.num_kv_heads * t.head_dim;
  auto q = random_vector(nq, 31, -t.q_range, t.q_range);
  auto k = random_vector(nk, 32);
  auto v = random_vector(nk, 33);
  DeviceArray dq(q), dk(k), dv(v), dout(nq);
  float scale = 1.0f / std::sqrt(static_cast<float>(t.head_dim));

  auto launch = [&] {
    return launch_attention(dq.ptr, dk.ptr, dv.ptr, dout.ptr, t.batch, t.num_heads,
                            t.num_kv_heads, t.seq_q, t.seq_k, t.head_dim, scale,
                            t.causal ? 1 : 0, 0);
  };
  CHECK(launch());
  CHECK(cudaDeviceSynchronize());
  auto got = dout.download();

  // Check every row of small cases, and about 300 rows (always including the
  // first and last query) of large ones.
  Comparison c;
  std::vector<double> ref;
  int rows = t.batch * t.num_heads * t.seq_q;
  int step = std::max(1, rows / 300);
  std::vector<int> sample;
  for (int r = 0; r < rows; r += step) sample.push_back(r);
  if (sample.back() != rows - 1) sample.push_back(rows - 1);
  for (int r : sample) {
    int i = r % t.seq_q, h = (r / t.seq_q) % t.num_heads, b = r / (t.seq_q * t.num_heads);
    attention_row_cpu(t, q, k, v, b, h, i, ref);
    size_t base = ((static_cast<size_t>(b) * t.seq_q + i) * t.num_heads + h) * t.head_dim;
    for (int d = 0; d < t.head_dim; ++d)
      compare_into(c, got[base + d], ref[d], base + d, 1e-4, 1e-3);
  }

  float ms = time_ms(launch);
  double pairs = static_cast<double>(t.seq_q) * t.seq_k;
  if (t.causal && t.seq_q == t.seq_k) pairs = pairs / 2;  // about half the pairs are masked
  double flops = 4.0 * t.batch * t.num_heads * pairs * t.head_dim;  // q.k and p*v
  return report(t.name, c, ms, fmt("%7.0f GFLOP/s", flops / (ms * 1e-3) / 1e9));
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  print_gpu();
  const AttnCase cases[] = {
      // name                                   B   H  Hkv  seq_q seq_k dim causal q_range
      {"tiny, one tile (4 queries, 5 keys)", 1, 1, 1, 4, 5, 32, false, 1.0f},
      {"several tiles, peaky scores", 1, 2, 2, 9, 100, 32, false, 8.0f},
      {"GQA 4 heads / 2 kv heads", 2, 4, 2, 17, 17, 64, false, 2.0f},
      {"causal, seq_q = seq_k", 1, 2, 1, 70, 70, 128, true, 4.0f},
      {"causal, 5 new queries, 37 keys", 2, 4, 2, 5, 37, 32, true, 4.0f},
      {"causal, one new query", 1, 16, 4, 1, 71, 128, true, 4.0f},
      {"text decoder (GQA 16/4, causal)", 1, 16, 4, 64, 64, 128, true, 4.0f},
      {"audio encoder (20 heads, 1500)", 1, 20, 20, 1500, 1500, 64, false, 4.0f},
  };
  bool ok = true;
  for (const auto& t : cases) {
    if (g_small_only && double(t.batch) * t.num_heads * t.seq_q * t.seq_k * t.head_dim > 1e7) continue;
    ok = run_case(t) && ok;
  }
  return finish("Task 4 (attention)", ok);
}
