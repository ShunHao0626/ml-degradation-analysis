# PCE–hour curve taxonomy

Independent, reproducible analysis of PCE-versus-hour trajectories based on the
PvkSOM workflow and its cited literature.

## Non-negotiable constraints

- Treat the existing `thesis` and `lab-v2` trees as read-only inputs.
- Do not force four clusters. Determine the cluster count using the procedure
  supported by the source literature.
- Do not add labels, engineered variables, auxiliary measurements, or supervised
  feedback to the unsupervised learner.
- Change only parameters explicitly supported by the source literature. Every
  changed parameter must be entered in the parameter evidence registry before it
  is used.
- Assess IFO-Bridge, IFO-Hill, IFO-Slope, and IFO-Valley only after clustering;
  those target names must not influence model fitting or model selection.
- Preserve unsuccessful experiments and full provenance.

## Read-only inputs

- Method/code: `/Users/shunhao/Desktop/ML/thesis`
- Initial samples: `/Users/shunhao/Desktop/ML/lab-v2/som_references/accepted/samples_test`

## Workflow gates

1. Audit source method and dataset.
2. Establish a literature-backed parameter whitelist and cluster-count rule.
3. Reproduce an immutable baseline.
4. Run constrained parameter configurations.
5. Evaluate cluster stability and interpret curve morphologies post hoc.
6. If the target morphologies remain unsupported, start a recent-literature
   review and archive lawful full text separately from metadata-only records.

## Directory guide

- `00_project_scope`: scope, decisions, source inventory, and provenance
- `01_source_method`: method extraction and parameter evidence
- `02_dataset_audit`: sample eligibility and data-quality reports
- `03_baseline_reproduction`: frozen baseline outputs
- `04_literature_parameter_sweeps`: permitted configurations and outputs
- `05_cluster_selection`: literature-backed cluster-count assessment
- `06_curve_type_results`: post-clustering morphology interpretation
- `07_robustness_checks`: seeds and stability diagnostics
- `08_literature_review`: metadata, screening, evidence, and lawful PDFs
- `src`: project-owned analysis code
- `config`: immutable machine-readable configurations
- `logs`: execution logs
- `reports`: Chinese and technical reports

## Current outcome

The paper-style cluster-count decision is `k=5` for the independently audited
150 h, 300 h, and 500 h windows. Four IFO-like individual curves are present,
but the literature-constrained SOM does not recover all four as independent
clusters; IFO-Valley-like remains mixed with decay trajectories. See
`reports/final_report.md` for the guarded interpretation and
`08_literature_review/literature_review_report.md` for the 1,068-record recent
literature corpus and evidence synthesis.
