#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// C-contiguous arrays: [curve, time, channel]. Deterministic tie order matches
// the existing Python implementations: diagonal, vertical, horizontal.
extern "C" void pairwise_dtw(const double* data, int count, int points,
                             int channels, const double* weights, int radius,
                             int normalize_path, double* output) {
    const double inf = std::numeric_limits<double>::infinity();
    std::fill(output, output + count * count, 0.0);
    for (int a = 0; a < count; ++a) {
        for (int b = a + 1; b < count; ++b) {
            std::vector<double> prev(points + 1, inf), cur(points + 1, inf);
            std::vector<int> plen(points + 1, 0), clen(points + 1, 0);
            prev[0] = 0.;
            for (int i = 1; i <= points; ++i) {
                std::fill(cur.begin(), cur.end(), inf);
                std::fill(clen.begin(), clen.end(), 0);
                for (int j = std::max(1, i-radius); j <= std::min(points, i+radius); ++j) {
                    double cost = 0.;
                    for (int c = 0; c < channels; ++c) {
                        double delta = data[(a*points+i-1)*channels+c]
                                     - data[(b*points+j-1)*channels+c];
                        cost += weights[c]*delta*delta;
                    }
                    double best = prev[j-1];
                    int length = plen[j-1];
                    if (prev[j] < best) { best = prev[j]; length = plen[j]; }
                    if (cur[j-1] < best) { best = cur[j-1]; length = clen[j-1]; }
                    cur[j] = best + std::sqrt(cost);
                    clen[j] = length + 1;
                }
                prev.swap(cur); plen.swap(clen);
            }
            double result = normalize_path ? prev[points]/plen[points] : prev[points];
            output[a*count+b] = output[b*count+a] = result;
        }
    }
}
