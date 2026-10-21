# Graph-Based Patient Similarity Network for Alzheimer's Disease

A multi-modal graph neural network for patient similarity learning, missing data imputation, and Alzheimer's disease classification.

## Overview

This model learns patient representations from heterogeneous Electronic Health Records (EHR) data using graph-based contrastive learning. It handles missing data through graph-based imputation and produces embeddings useful for downstream classification.

## Trained Models

Pretrained checkpoints and experiment outputs are hosted on Google Drive:

- **Trained model** — main BRIDGE model checkpoints (`best_model.pth`) ready for
  imputation and classification:
  [Google Drive](https://drive.google.com/drive/folders/17FRTZfJoOYDgTdeHuXHYgXScyKskr2Tx?usp=drive_link)
- **Experiment models** — checkpoints and outputs from the multi-seed, ablation,
  and inference-ablation experiments used to reproduce the paper tables/figures:
  [Google Drive](https://drive.google.com/drive/folders/1js3vT8jz_R_Bf7naalmK9GXdXwZJgI-L?usp=drive_link)

## Install

We use conda enviroment
`conda create --name psn python=3.10`

Activate and install requirements

```
conda activate psn
pip install -r requirements.txt
```

## Project Structure

```
├── config.py              # Hyperparameters and paths
├── data_loader.py         # Data loading and preprocessing
├── graph_builder.py       # Graph construction (ModalityEncoder, ModalityGraphConstructor)
├── gcn.py                 # Graph Convolutional Network
├── fusion.py              # Representation fusion module
├── decoders.py            # Feature reconstruction decoders
├── data_augmentation.py   # Augmentation and loss functions
├── train.py               # Training pipeline (10-fold CV)
├── imputation.py          # Embedding extraction, baseline imputers, imputation evaluation
├── gain_imputer.py        # GAIN baseline
├── grape_imputer.py       # GRAPE baseline
├── classification.py      # Classification evaluation
└── hyperparam_tuning.py   # Hyperparameter search
```

### Notebooks

The pipeline above runs on the preprocessed modality CSVs in `Features/`. The
notebooks below build those CSVs from raw OMOP/EHR data and analyze the results.

```
Preprocessing/
├── extraction_code/
│   ├── feature_extraction.ipynb   # Raw OMOP → comorbidity (CCI/ICD), demographic,
│   │                              #   lab, drug, cognitive features per patient
│   └── find_group.ipynb           # Cohort selection (AD/MCI/control group finding)
├── New.ipynb                      # Current PSN extraction: reads 2026 refresh +
│                                  #   OMOP tables → M1–M7 modality CSVs
└── M5_further_process.ipynb       # Min-max scales M5 cognitive scores (CDRGLOB, GDS)

dataset_description.ipynb          # Table 1 cohort summary + missingness analysis
aggregate_results_v3.ipynb         # Aggregates ablation multi-seed runs → CI tables
                                   #   + baseline-vs-model comparison tables
result_analysis/
└── imputation_result_process.ipynb  # Neighbor / same-label-ratio analysis
```

## Data

7 modalities from EHR data:
- M1: Demographics (11 features)
- M2: Comorbidities (13 features)
- M3: Lab Tests (7 features)
- M4: Drugs (7 features)
- M5: Cognitive Scores (9 features)
- M6: Genetic/APOE (1 feature)
- M7: Protein Biomarkers (8 features)

## Imputation Baselines

`imputation.py` compares the model against the baselines below on the imputed
modalities (M3, M5, M6, M7). Every baseline is fit on the training split and
fills only missing entries. Each one is used twice: for imputation evaluation
(test positions hidden by `R_eval`) and for downstream classification. Outputs are
saved as `imputations/{train,test}_{method}_{modality}.npy`.

| Method (CSV name) | Implementation | Reference |
|---|---|---|
| `Mean`, `Median` | `sklearn.impute.SimpleImputer` | — |
| `KNN1`, `KNN3`, `KNN` (k=5) | `sklearn.impute.KNNImputer` | — |
| `LinearRegression` | One linear regression per feature on mean-filled other features (single pass, not iterative) | — |
| `MICE` | `sklearn.impute.IterativeImputer` with `BayesianRidge` + posterior sampling, 10 iterations, m=5 imputations averaged (Rubin's-rule point estimate) | van Buuren & Groothuis-Oudshoorn 2011; Zhang 2016 |
| `MissForest` | `sklearn.impute.IterativeImputer` with `RandomForestRegressor` (100 trees, `max_features="sqrt"`), mean initial guess, columns in ascending order of missingness | Stekhoven & Bühlmann 2012 |
| `PPM` | Pattern-mixture model: MICE whose imputation models also take every feature's missingness indicator, plus a delta adjustment for the feature's own missingness (MNAR sensitivity) | Little 1993; Ratitch et al. 2013; Yan et al. 2025 |
| `GAIN` | `gain_imputer.py` (continuous modalities; M6 uses the Mean proxy) | Yoon et al. 2018 |
| `GRAPE` | `grape_imputer.py` (continuous modalities; M6 uses the Mean proxy) | You et al. 2020 |

PPM's sensitivity parameter is `ppm_delta` in `config.py`. It shifts imputed
values by `ppm_delta × training SD` of each feature. The default `0.0` is equivalent
to MAR. Because the evaluation masks are MCAR, use non-zero values (e.g. ±0.25, ±0.5)
for a tipping-point / MNAR sensitivity analysis.

> **Adding baselines to finished runs:** the new baselines are listed as done
> markers in `run_multi_seed.py`, so seeds completed before they were added will
> rerun. To skip retraining, rerun only the downstream steps:
> `bash run_multi_seed.sh --steps imputation error_analysis classification eval_demographics`


## Usage

The full reproduction path is: **(0) build features from raw data → (1) train &
evaluate → (2) aggregate & analyze.** Steps 0 and 2 are notebooks; step 1 is the
Python pipeline (run locally or on SLURM).

### Step 0: Build features from raw data (notebooks)

Skip this step if you already have the preprocessed modality CSVs in `Features/`.
Run the notebooks top-to-bottom in this order. Each writes CSVs consumed by the
next step; edit the hard-coded input paths at the top of each notebook to point
at your raw data location.

| Order | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `Preprocessing/extraction_code/find_group.ipynb` | Raw OMOP tables | Cohort / group assignments |
| 2 | `Preprocessing/extraction_code/feature_extraction.ipynb` | OMOP + cohort | Comorbidity / demographic / lab / drug / cognitive features |
| 3 | `Preprocessing/New.ipynb` | 2026 refresh CSVs + OMOP tables | `Features/m{1..7}_features*.csv` |
| 4 | `Preprocessing/M5_further_process.ipynb` | `M5_features_clean.csv` | `M5_features_scaled.csv` |

```bash
jupyter lab        # or: jupyter notebook
# open each notebook above and Run All, in order
```

The training pipeline reads the 7 modality CSVs (M1–M7) produced here. See the
[Data](#data) section for the feature counts per modality.

### Step 1: Train & evaluate

#### Single Run (local / interactive)
Run the full pipeline for one seed in sequence:
```bash
python train.py
python decoder_mod.py
python imputation.py
python error_analysis.py
python classification.py
python eval_demographics.py
```

Override output directory or random seed with CLI flags:
```bash
python train.py --output-dir results/my_run --random-state 42
```

---

## Running Experiments on SLURM

All cluster experiments are submitted via scripts in the project root. Edit the
`CPUS`, `MEM`, `TIME`, and `GPUS` variables at the top of each script to match
your cluster allocation before submitting.

### 1. Main Experiment — 5-Seed CI Results (Table 2 & 3, Figure 3)

Submits **one SLURM job** that runs all 5 seeds (1, 2, 3, 4, 42) sequentially
and then aggregates 95% confidence intervals.

```bash
# Run all 5 seeds, all pipeline steps
bash slurm_multi_seed.sh
```

Outputs written to `results_multi_seed/seed_{N}/` for each seed, and
aggregated CSVs to `results_multi_seed/`:
- `classification_aggregated.csv` — Table 3 (mean ± 95% CI)
- `imputation_aggregated.csv` — Table 2 (mean ± 95% CI)
- `classification_by_demographics_aggregated.csv` — demographic subgroup results

Monitor with: `squeue -u $USER`

---

### 2. Top-k Ablation Study (Table 3 — top-k sensitivity)

Submits **one SLURM job per stage** (6 stages total) so all stages run in
parallel. Each job runs all 5 seeds for its top-k configuration.

| Stage | top-k | Description |
|---|---|---|
| `S_trim_topk1_rr1_ma1` | 1 | 1 neighbor per patient |
| `S_trim_topk3_rr1_ma1` | 3 | 3 neighbors per patient |
| `S_trim_topk5_rr1_ma1` | 5 | 5 neighbors per patient |
| `S_trim_topk10_rr1_ma1` | 10 | 10 neighbors per patient |
| `S_trim_topk20_rr1_ma1` | 20 | 20 neighbors per patient |
| `S_trim_nofilter_rr1_ma1` | — | No top-k filtering |

```bash
# Submit all 6 stages in parallel
bash slurm_ablation.sh
```

Outputs written to `results_ablation/{stage_name}/seed_{N}/`.

---

### 3. Inference Ablation (graph/fusion configuration sweep)

Re-uses trained models from the main experiment (`results_multi_seed/`) and
evaluates imputation + classification under different inference-time
graph/fusion settings — **no retraining required**.

Submits **one SLURM job per stage** (6 stages total).

| Stage | filter-graph | fusion | Description |
|---|---|---|---|
| `S_no_filter_avg` | off | average | No graph filter, average fusion |
| `S_no_filter_gated` | off | gated | No graph filter, gated fusion |
| `S_topk1` | on (k=1) | gated | top-k=1 at inference |
| `S_topk3` | on (k=3) | gated | top-k=3 at inference |
| `S_topk5` | on (k=5) | gated | top-k=5 at inference |
| `S_topk10` | on (k=10) | gated | top-k=10 at inference |

```bash
# Submit all stages in parallel
bash slurm_inference_ablation.sh
```

Outputs written to `results_inference_ablation/{stage_name}/seed_{N}/`,
with per-stage aggregated CSVs alongside.

---

### 4. Single-Seed Quick Run (development / debugging)

Submits **one SLURM job** that runs the pipeline for a single seed without
multi-seed orchestration. Useful for verifying a new configuration before
committing to a full 5-seed run.

```bash
sbatch run.sh
```

> `run.sh` uses hardcoded `#SBATCH` directives — edit the file header directly
> to change CPU, GPU, memory, or time limits.

---

### Output Directory Layout

```
results_multi_seed/
├── seed_1/
│   ├── output/          # best_model.pth, classification_all_methods.csv, ...
│   ├── imputations/     # train_*/test_*_{method}_{modality}.npy
│   ├── embeddings/      # train_embeddings.npy, test_embeddings.npy
│   └── plots/
├── seed_2/ ... seed_42/
├── classification_aggregated.csv       ← main CI table
├── imputation_aggregated.csv           ← imputation CI table
└── classification_by_demographics_aggregated.csv

results_ablation/
└── {stage_name}/seed_{N}/  (same layout as above)

results_inference_ablation/
└── {stage_name}/
    ├── seed_{N}/
    └── classification_aggregated.csv   ← per-stage CI table
```

---

## Step 2: Aggregate & analyze (notebooks)

The main experiment (`slurm_multi_seed.sh`) writes its own CI tables directly to
`results_multi_seed/`. The notebooks below produce the remaining paper tables,
figures, and ablation summaries. Run them after the corresponding experiments
finish; edit the input paths near the top of each notebook to match your output
directories.

| Notebook | Reads | Produces |
|---|---|---|
| `dataset_description.ipynb` | `Features/m{1..7}_features*.csv`, `Features/observation_periods.csv` | Table 1 cohort summary, per-feature/per-modality missingness tables and heatmaps |
| `aggregate_results_v3.ipynb` | `results_ablation/{stage}/seed_{seed}/output/` (or `results_multi_seed/seed_{seed}/` with `STAGES = [""]`) | Per-stage CI tables (`results_ablation_agg/`), ablation summary CSVs, and baseline comparison tables: `baseline_imputation_mae_comparison.csv`, `baseline_imputation_categorical_comparison.csv`, `baseline_classification_auroc_comparison.csv` |
| `result_analysis/imputation_result_process.ipynb` | neighbor dump (`top25_neighbors_*.txt`) + modality CSVs | Same-label-ratio and neighbor-missingness analysis |

```bash
jupyter lab        # open the notebook, Run All
```

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gcn_dim` | Hidden dimension | 256 |
| `graph_heads` | Attention heads | 8 |
| `top_k` | Neighbors per patient | 25 |
| `tau` | Temperature | 0.5 |
| `alpha1` | Forward loss weight | 0.55 |
| `alpha2` | Backward loss weight | 0.4 |
| `lambda_imp` | Imputation loss weight | 2.0 |
| `prob_edge` | Edge dropout probability | 0.4 |
| `ppm_delta` | PPM baseline delta shift (× feature SD); 0 = MAR | 0.0 |

## Output

- `best_model.pth` - Trained model checkpoint
- `embeddings/` - Patient embeddings (1792-dim)
- `imputations/` - Imputed values
- `output/` - Classification reports and metrics
- `plots/` - Visualizations

## Citation

If you use this code or models, please cite our paper (citation is a placeholder
and will be updated):

```bibtex
@article{chang_bridge,
  title   = {Bridging Missing Domains with Graph Embeddings via Patient Similarity Networks for ADRD Analysis},
  author  = {Chang, Tien Yu and Gunturu, Phaneendra Babu and Wang, Jiankun and Zhuang, Yan and Gao, Sujuan and Zhou, Jiayu and Huang, Yu and Bian, Jiang},
  note    = {Placeholder citation; to be updated},
  year    = {2026}
}
```
