"""Keep every >=4-point curve for comparison with the small QC exclusion."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from run import CONFIG, HERE, OUT, encode, fit, read_rows, write_rows


def main() -> None:
    rows = read_rows((HERE / CONFIG["input_index"]).resolve())
    with np.load((HERE / CONFIG["input_shapes"]).resolve()) as data:
        raw = data["raw_rank"]
        assert data["curve_id"].tolist() == [r["curve_id"] for r in rows]
    embedding, _, _, _ = encode(raw)
    labels, _, _ = fit(embedding)
    output = [dict(curve_id=r["curve_id"], anonymous_cluster=int(labels[i]))
              for i, r in enumerate(rows)]
    write_rows(OUT / "all_curves_variant_assignments.csv", output, list(output[0]))
    summary = dict(included=len(rows), excluded=0,
                   cluster_sizes=dict(Counter(map(int, labels))),
                   note="Same label-free shape fit without roughness exclusion; one group is dominated by repeated spikes/near-zero jumps.")
    (OUT / "all_curves_variant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
