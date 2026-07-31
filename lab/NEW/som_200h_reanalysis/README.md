# 200 h SOM reanalysis

This directory is an independent, reproducible SOM reanalysis of the 2,151
literature-mined ageing curves in `lab/data_all`.

It does not read from or overwrite `lab/RESULTS_>200h`.

The complete analysis is generated with:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python3 code/run_pipeline.py
python3 code/verify_outputs.py
```

If the computational results already exist and only reports/checksums need to
be rebuilt, run `python3 code/finalize_reports.py`.

After a successful run, start with:

- `REPORT.md`
- `FILE_INDEX.md`
- `06_final_model/final_curve_classification.csv`
- `05_cluster_number_decision/SI_METHOD_APPLICATION.md`
- `08_group_presentation_comparison/PRESENTATION_GUIDE_CN.md`
- `09_si_exact_replication/SI_EXACT_REPLICATION_CN.md`
