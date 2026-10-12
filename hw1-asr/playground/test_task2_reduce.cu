// Task 2 test: RMSNorm, LayerNorm and softmax, checked against CPU versions.
// Build and run with:  make test2
#include "test_util.h"

enum class Op { RMSNorm, LayerNorm, Softmax };

struct ReduceCase {
  const char* name;
  Op op;
  int rows, cols;
  float lo, hi;  // range of the random inputs
};

static std::vector<double> cpu_reference(Op op, const std::vector<float>& x,
                                         const std::vector<float>& w,
                                         const std::vector<float>& b, int rows, int cols,
                                         double eps) {
  std::vector<double> y(x.size());
  for (int r = 0; r < rows; ++r) {
    const float* xr = &x[static_cast<size_t>(r) * cols];
    double* yr = &y[static_cast<size_t>(r) * cols];
    if (op == Op::RMSNorm) {
      double sq = 0;
      for (int c = 0; c < cols; ++c) sq += double(xr[c]) * xr[c];
      double inv = 1.0 / std::sqrt(sq / cols + eps);
      for (int c = 0; c < cols; ++c) yr[c] = xr[c] * inv * w[c];
    } else if (op == Op::LayerNorm) {
      double mean = 0, var = 0;
      for (int c = 0; c < cols; ++c) mean += xr[c];
      mean /= cols;
      for (int c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
      double inv = 1.0 / std::sqrt(var / cols + eps);
      for (int c = 0; c < cols; ++c) yr[c] = (xr[c] - mean) * inv * w[c] + b[c];
    } else {
      double m = -INFINITY, s = 0;
      for (int c = 0; c < cols; ++c) m = std::max(m, double(xr[c]));
      for (int c = 0; c < cols; ++c) s += std::exp(xr[c] - m);
      for (int c = 0; c < cols; ++c) yr[c] = std::exp(xr[c] - m) / s;
    }
  }
  return y;
}

static bool run_case(const ReduceCase& t, double peak_gbs) {
  const float eps = 1e-5f;
  size_t n = static_cast<size_t>(t.rows) * t.cols;
  auto x = random_vector(n, 11, t.lo, t.hi);
  auto w = random_vector(t.cols, 12, 0.5f, 1.5f);
  auto b = random_vector(t.cols, 13);
  DeviceArray dx(x), dw(w), db(b), dy(n);

  auto launch = [&] {
    switch (t.op) {
      case Op::RMSNorm: return launch_rmsnorm(dx.ptr, dw.ptr, dy.ptr, t.rows, t.cols, eps, 0);
      case Op::LayerNorm:
        return launch_layernorm(dx.ptr, dw.ptr, db.ptr, dy.ptr, t.rows, t.cols, eps, 0);
      default: return launch_softmax(dx.ptr, dy.ptr, t.rows, t.cols, 0);
    }
  };
  CHECK(launch());
  CHECK(cudaDeviceSynchronize());

  auto ref = cpu_reference(t.op, x, w, b, t.rows, t.cols, eps);
  // Softmax outputs are tiny (about 1/cols), so they get a tighter absolute tolerance.
  double atol = t.op == Op::Softmax ? 1e-7 : 1e-4;
  Comparison c = compare(dy.download(), ref, atol, 1e-4);

  float ms = time_ms(launch);
  double bytes = 2.0 * n * sizeof(float);  // read x once, write y once (weights are small)
  double gbs = bytes / (ms * 1e-3) / 1e9;
  return report(t.name, c, ms, fmt("%7.1f GB/s (%2.0f%% of peak)", gbs, 100.0 * gbs / peak_gbs));
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  double peak = print_gpu();
  const ReduceCase cases[] = {
      // name                                op              rows   cols    input range
      {"rmsnorm, one short row", Op::RMSNorm, 1, 40, -1, 1},
      {"rmsnorm, odd cols", Op::RMSNorm, 5, 1001, -1, 1},
      {"rmsnorm, text decoder", Op::RMSNorm, 64, 2048, -3, 3},
      {"layernorm, odd cols (example)", Op::LayerNorm, 5, 1001, -1, 1},
      {"layernorm, audio encoder", Op::LayerNorm, 1500, 1280, -3, 3},
      {"softmax, one short row", Op::Softmax, 1, 7, -1, 1},
      {"softmax, odd cols", Op::Softmax, 3, 1001, -5, 5},
      {"softmax, large inputs (+-100)", Op::Softmax, 4, 513, -100, 100},
      {"softmax, all negative (-200..-150)", Op::Softmax, 2, 300, -200, -150},
      {"softmax, vocabulary row (59264)", Op::Softmax, 1, 59264, -10, 10},
      {"softmax, 64 x 4096", Op::Softmax, 64, 4096, -10, 10},
  };
  bool ok = true;
  for (const auto& t : cases) {
    if (g_small_only && double(t.rows) * t.cols > 2e5) continue;
    ok = run_case(t, peak) && ok;
  }
  return finish("Task 2 (reductions)", ok);
}
