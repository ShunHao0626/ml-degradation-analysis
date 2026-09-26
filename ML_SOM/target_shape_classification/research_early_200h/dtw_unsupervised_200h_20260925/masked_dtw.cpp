#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Each pair is compared only at grid hours supported by both observed curves.
// The three distances share the same input, but each computes its own DTW path.
static double run_dtw(const std::vector<int>& shared, const double* level,
                      const double* derivative, int a, int b, int grid,
                      int warp_steps, int mode) {
    const double inf = std::numeric_limits<double>::infinity();
    const int n = static_cast<int>(shared.size());
    std::vector<double> prev(n + 1, inf), cur(n + 1, inf);
    std::vector<int> prev_len(n + 1, 0), cur_len(n + 1, 0);
    prev[0] = 0.0;
    for (int i = 1; i <= n; ++i) {
        std::fill(cur.begin(), cur.end(), inf);
        std::fill(cur_len.begin(), cur_len.end(), 0);
        for (int j = std::max(1, i - warp_steps); j <= std::min(n, i + warp_steps); ++j) {
            int ti = shared[i - 1], tj = shared[j - 1];
            if (std::abs(ti - tj) > warp_steps) continue;
            double yl = level[a * grid + ti] - level[b * grid + tj];
            double dl = derivative[a * grid + ti] - derivative[b * grid + tj];
            double cost = mode == 0 ? std::abs(yl) : mode == 1 ? std::abs(dl)
                            : std::sqrt(0.5 * yl * yl + 0.5 * dl * dl);
            double best = prev[j - 1];
            int length = prev_len[j - 1];
            if (prev[j] < best) { best = prev[j]; length = prev_len[j]; }
            if (cur[j - 1] < best) { best = cur[j - 1]; length = cur_len[j - 1]; }
            if (!std::isfinite(best)) continue;
            cur[j] = best + cost;
            cur_len[j] = length + 1;
        }
        prev.swap(cur);
        prev_len.swap(cur_len);
    }
    return std::isfinite(prev[n]) && prev_len[n] > 0 ? prev[n] / prev_len[n] : inf;
}

extern "C" void masked_pair_dtw(const double* level, const double* derivative,
                                 const uint8_t* valid, int grid,
                                 const int32_t* pairs, int pair_count,
                                 int warp_steps, int minimum_shared,
                                 int minimum_span_steps, double* output) {
    const double inf = std::numeric_limits<double>::infinity();
    for (int k = 0; k < pair_count; ++k) {
        int a = pairs[2 * k], b = pairs[2 * k + 1];
        std::vector<int> shared;
        shared.reserve(grid);
        int union_count = 0;
        for (int t = 0; t < grid; ++t) {
            bool va = valid[a * grid + t] != 0, vb = valid[b * grid + t] != 0;
            if (va || vb) ++union_count;
            if (va && vb) shared.push_back(t);
        }
        if (static_cast<int>(shared.size()) < minimum_shared ||
            shared.back() - shared.front() < minimum_span_steps) {
            output[3 * k] = output[3 * k + 1] = output[3 * k + 2] = inf;
            continue;
        }
        // Penalize matching a short shared prefix of a much longer curve.
        double missing_fraction = 1.0 - double(shared.size()) / union_count;
        for (int mode = 0; mode < 3; ++mode) {
            double d = run_dtw(shared, level, derivative, a, b, grid,
                               warp_steps, mode);
            output[3 * k + mode] = d * (1.0 + 0.4 * missing_fraction)
                                   + 0.03 * missing_fraction;
        }
    }
}
