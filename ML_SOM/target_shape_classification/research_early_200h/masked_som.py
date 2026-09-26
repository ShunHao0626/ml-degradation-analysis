"""Single-map SOM with support masks and group-normalized distances."""
from __future__ import annotations

import numpy as np


class MaskedSOM:
    def __init__(self, rows, columns, sizes, weights, seed):
        self.rows, self.columns = rows, columns
        self.sizes = tuple(sizes)
        self.weights = np.asarray(weights, dtype=float)
        self.seed = seed
        self.slices = []
        start = 0
        for size in sizes:
            self.slices.append(slice(start, start + size))
            start += size
        self.coordinates = np.array([(i, j) for i in range(rows) for j in range(columns)])

    def fit(self, data, mask, epochs=25):
        if not len(data):
            raise ValueError("no train data")
        rng = np.random.default_rng(self.seed)
        self.supported = np.any(mask, axis=0)
        if not np.all(self.supported):
            # Unsupported dimensions are excluded from every distance/update.
            mask = mask & self.supported
        medians = np.zeros(data.shape[1])
        medians[self.supported] = np.nanmedian(np.where(mask[:, self.supported],
                                                        data[:, self.supported], np.nan), axis=0)
        self.prototypes = np.tile(medians, (self.rows * self.columns, 1))
        scale = np.zeros(data.shape[1])
        scale[self.supported] = np.nanstd(np.where(mask[:, self.supported],
                                                   data[:, self.supported], np.nan), axis=0)
        scale[~np.isfinite(scale)] = 0
        self.prototypes += rng.normal(0, .05, self.prototypes.shape) * scale
        self.prototypes[:, ~self.supported] = 0
        trace = []
        for epoch in range(epochs):
            fraction = epoch / max(epochs - 1, 1)
            lr = .3 * (.03 / .3) ** fraction
            radius = max(self.rows, self.columns) / 2 * (.5 / (max(self.rows, self.columns) / 2)) ** fraction
            for index in rng.permutation(len(data)):
                m = mask[index] & self.supported
                if not m.any():
                    continue
                distances = self.distance_one(data[index], m)
                bmu = int(np.argmin(distances))
                squared = np.sum((self.coordinates - self.coordinates[bmu]) ** 2, axis=1)
                neighborhood = (lr * np.exp(-squared / (2 * radius ** 2)))[:, None]
                self.prototypes[:, m] += neighborhood * (data[index, m] - self.prototypes[:, m])
            if epoch in (0, epochs // 2, epochs - 1):
                q = self.assign(data, mask)[1]
                trace.append(dict(epoch=epoch + 1, train_quantization_error=float(np.mean(q))))
        return trace

    def distance_one(self, x, mask):
        numerator = np.zeros(len(self.prototypes))
        denominator = 0.
        for weight, sl in zip(self.weights, self.slices):
            m = mask[sl] & self.supported[sl]
            if not m.any():
                continue
            delta = self.prototypes[:, sl][:, m] - x[sl][m]
            numerator += weight * np.mean(delta * delta, axis=1)
            denominator += weight
        return np.sqrt(numerator / denominator) if denominator else np.full(len(self.prototypes), np.inf)

    def assign(self, data, masks):
        bmus = np.empty(len(data), dtype=int)
        q = np.empty(len(data))
        second = np.empty(len(data), dtype=int)
        for i, (x, mask) in enumerate(zip(data, masks)):
            d = self.distance_one(x, mask)
            pair = np.argpartition(d, 2)[:2]
            pair = pair[np.argsort(d[pair])]
            bmus[i], second[i], q[i] = pair[0], pair[1], d[pair[0]]
        return bmus, q, second

    def metrics(self, data, mask):
        if not len(data):
            return {}
        bmu, q, second = self.assign(data, mask)
        adjacent = np.max(np.abs(self.coordinates[bmu] - self.coordinates[second]), axis=1) <= 1
        return dict(quantization_error=float(q.mean()), topology_error=float(np.mean(~adjacent)),
                    occupied_nodes=int(len(np.unique(bmu))), sample_count=int(len(data)))
