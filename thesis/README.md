> **About this folder (added in ml-degradation-analysis).** This README below the line is an unmodified copy of the upstream [noortitan/PvkSOM](https://github.com/noortitan/PvkSOM) README by Noor Titan Putri Hartono (BSD 2-Clause, see `LICENSE`). Its installation steps refer to the upstream repository: `PCE_df_grouping.csv` and the Zenodo `.pkl` files are **not** included here and must be obtained from the upstream repository and [Zenodo](https://doi.org/10.5281/zenodo.8185882).
>
> Files added for this project:
>
> | File | Purpose |
> |---|---|
> | `run_pipeline.py`, `environment_mac.yml` | Script version of the upstream notebook workflow (revision 11, with revision 10 as reference) for running outside Jupyter |
> | `run_som_preprocessing_comparison.py`, `Three_Run_SOM_Preprocessing_Comparison_Prompt.md`, `som_preprocessing_comparison/` | Three-way preprocessing comparison (raw, resample + MaxAbs, resample + MaxAbs + Savitzky–Golay) on the 2,245 Hartono curves; see `som_preprocessing_comparison/final_analysis.md` (Chinese) |
> | `my_pipeline.py` | Adaptation of the pipeline to the author's own 30 MPPT measurements (hard-coded local paths) |
> | `normalize_data_200h.py` | Converts literature curves to a 0–200 h, 10-min grid with MaxAbs scaling (hard-coded local paths) |
> | `paper/convert_pdf.py`, `paper/convert_supp.py` | Convert the Hartono article and its supplement to Markdown for reading; the PDFs themselves are not included |
> | `result/`, `20230816_run_revision_excN2/` | Summary figures and cluster tables from local runs |
>
> 中文说明：分隔线以下为上游 PvkSOM README 的原样副本；`PCE_df_grouping.csv` 与 Zenodo 数据不在本仓库中。本项目新增的文件见上表。

---

# PvkSOM: Perovskite solar cells degradation data clustering using self-organizing map
This documentation is prepared as the workflow to accompany the following study:

**"Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset"**

*Nat Commun 14, 4869 (2023) https://doi.org/10.1038/s41467-023-40585-3*

Noor Titan Putri Hartono (1), Hans Köbler (1), Paolo Graniero (1,2), Mark Khenkin (1), Rutger Schlatmann (1), Carolin Ulbrich (1), Antonio Abate (1)

Affiliations:

1. Helmholtz-Zentrum-Berlin für Materialien und Energie, 14109 Berlin, Germany
2. Department of Business Informatics, Freie Universität Berlin, 14195 Berlin, Germany

## Installation and Requirements
To install, just clone this repository:

`$ git clone https://github.com/noortitan/PvkSOM.git`

`$ cd PvkSOM`

To install the required packages, create a virtual environment using Anaconda/ Miniconda (https://docs.conda.io/en/latest/miniconda.html). The optional but recommended setup on Anaconda/ Miniconda:

`$ conda env create -f environment.yml`

`$ conda activate PvkSOM`

`$ jupyter notebook`

## Workflow

The processed data in `.pkl` format is available on Zenodo (https://doi.org/10.5281/zenodo.8185882), along with the dataset in this repository (`PCE_df_grouping.csv`).

**Use `20230816_degradation_analysis_revision_11_cleaned.ipynb` Jupyter Notebook with this data.** This notebook loads the `.pkl` and `.csv` files, further pre-processing me (including smoothing (using Savitzky-Golay filter), and normalizing the data). At the end of the pre-processing, 2,245 data points are left, and we can start to train the dataset using self-organizing map (SOM). SOM is a popular unsupervised machine learning method for clustering data by converting the nonlinear statistical relationships of high-dimensional data into geometric relationships in low-dimensional nodes while preserving the topological structure of data, i.e. the relations between one data point and others (Kohonen, T., 1990, https://doi.org/10.1109/5.58325). The *k*-means clustering algorithm is also performed as a comparison to the SOM clustering method. 

**The other Jupyter Notebooks (which can't be run),** `20221122_degradation_preprocessing_2.ipynb` and `20230227_degradation_analysis_revision_10_cleaned.ipynb`, are meant to demonstrate how the initial raw data is generated from the High-Throughput Ageing Test setup is first processed. The raw data (that comes in `.json` format; where each file corresponds to one degradation batch) is first converted into dataframes and saved in a pickle (`.pkl`) file on `20221122_degradation_preprocessing_2.ipynb` Jupyter Notebook. Then, the dataframe `.pkl` files are loaded to `20230227_degradation_analysis_revision_10_cleaned.ipynb` Jupyter Notebook, where data pre-processing is performed, which involves excluding 'bad' data, resampling, cutting down the data to 150 hours, smoothing (using Savitzky-Golay filter), and normalizing the data. After that, the subsequent steps are similar to the `20230816_degradation_analysis_revision_11_cleaned.ipynb` Jupyter Notebook.

## Authors
| |  | 
|---|---|
|**Author(s)** | Noor Titan Putri Hartono |
|**Version** | 1.1/ August 2023  |   
|**E-mail(s)**   | titan dot hartono at helmholtz-berlin dot de  |
| | |

## Attribution
This work is under an BSD 2-Clause License License. Please, acknowledge use of this work with the appropriate citation to the repository and research article.

## Citation

    @software{noortitan_2023_8181602,
      author       = {noortitan},
      title        = {noortitan/PvkSOM: First Release},
      month        = jul,
      year         = 2023,
      publisher    = {Zenodo},
      version      = {v1.0},
      doi          = {10.5281/zenodo.8181602},
      url          = {\url{https://doi.org/10.5281/zenodo.8181602}}
    }
