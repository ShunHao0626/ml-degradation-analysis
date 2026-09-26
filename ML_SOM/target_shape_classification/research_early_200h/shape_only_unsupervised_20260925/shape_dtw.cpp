#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" void shape_pair_dtw(const double* level, const double* short_change,
                                const double* long_change, int grid,
                                const int32_t* pairs, int pair_count, int warp_steps,
                                const double* weights, double* output) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> prev(grid + 1), cur(grid + 1);
    std::vector<int> prev_len(grid + 1), cur_len(grid + 1);
    for (int k = 0; k < pair_count; ++k) {
        int a = pairs[2 * k], b = pairs[2 * k + 1];
        std::fill(prev.begin(), prev.end(), inf);
        std::fill(prev_len.begin(), prev_len.end(), 0);
        prev[0] = 0.0;
        for (int i = 1; i <= grid; ++i) {
            std::fill(cur.begin(), cur.end(), inf);
            std::fill(cur_len.begin(), cur_len.end(), 0);
            for (int j = std::max(1, i - warp_steps); j <= std::min(grid, i + warp_steps); ++j) {
                int ai = a * grid + i - 1, bj = b * grid + j - 1;
                double cost = weights[0] * std::abs(level[ai] - level[bj])
                            + weights[1] * std::abs(short_change[ai] - short_change[bj])
                            + weights[2] * std::abs(long_change[ai] - long_change[bj]);
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
        output[k] = std::isfinite(prev[grid]) && prev_len[grid] > 0
                  ? prev[grid] / prev_len[grid] : inf;
    }
}
