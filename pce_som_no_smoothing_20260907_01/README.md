# PCE curve discovery — unsupervised SOM without smoothing

This is a new, self-contained result folder based on the method in
`../thesis` and applied to `../lab/data_all`.

Key constraints:

- no class labels are used in preprocessing, training, parameter selection,
  cluster-number selection, or final-model selection;
- smoothing is disabled everywhere;
- the SOM cluster count is selected by the paper's quantization-error elbow
  followed by the paper's distinct/overlapping-centre check;
- IFO Bridge/Hill/Slope/Valley names are considered only after the final
  partition is frozen;
- the 500 h window is primary because the requested morphology distinguishes
  behaviour before and after 200 h; the paper's 150 h window is a sensitivity
  analysis.

Linear interpolation is used only to put irregularly sampled curves on a fixed
time grid required by SOM. It connects observed points and does not apply a
filter. Raw-point plots are exported beside the model-grid plots.

Run:

```bash
MPLCONFIGDIR=/tmp/pce_som_mpl python3 code/run_analysis.py --stage all
MPLCONFIGDIR=/tmp/pce_som_mpl python3 code/run_shape_sensitivity.py
```

The primary report is `reports/final_report_cn.md`; the shape-sensitive
sensitivity report is `08_shape_sensitive/shape_sensitive_report_cn.md`. The project can be
verified with:

For the short answer first, open `FINAL_DECISION_CN.md`.

```bash
python3 code/verify_outputs.py
```
