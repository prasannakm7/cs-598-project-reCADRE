# reCADRE: Reproducing CADRE in PyHealth

A reproduction and extension of **CADRE** (Contextual Attention-based Drug REsponse) within the [PyHealth](https://pyhealth.readthedocs.io/) framework, originally published as:

> Tao, Y., Ren, S., Ding, M.Q., Schwartz, R., & Lu, X. (2020). *Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention.* Proceedings of Machine Learning Research, 126, 456–477. PMLR (MLHC 2020).

**Original code:** [github.com/yifengtao/CADRE](https://github.com/yifengtao/CADRE)

## Overview

CADRE predicts the sensitivity of cancer cell lines to oncology drugs using gene expression profiles. It combines:

1. **Collaborative Filtering** — jointly predicts all drug responses for a cell line via learned drug embeddings and dot-product decoding
2. **Contextual Attention** — weights gene importance differently per drug based on the drug's target pathway, producing drug-specific cell line representations
3. **Pretrained Gene Embeddings** — transfers biological knowledge from Gene2Vec embeddings trained on large-scale co-expression data

This project re-implements the full CADRE architecture using PyHealth's `SampleBaseDataset` abstraction and a standard PyTorch `DataLoader`, enabling integration with the broader PyHealth ecosystem. It also includes Extension 2: a scaled dot-product attention variant (`CADREDotAttn`) that replaces CADRE's additive contextual conditioning with transformer-style query/key alignment.

## Project Structure

```
reCADRE/
├── README.md               # This file
├── dataset.py              # GDSC dataset wrapper (PyHealth SampleBaseDataset)
├── model.py                # CADRE model (encoder, decoder, attention, loss)
├── model_dot_attn.py       # Extension 2: CADREDotAttn (dot-product attention)
├── train.py                # Training script with OneCycle LR, evaluation, logging
├── run_extension2.py       # Extension 2: trains both models and prints comparison
├── originalData/           # Pre-processed GDSC data files
│   ├── exp_gdsc.csv            # Binary gene expression (1014 × 3000)
│   ├── gdsc.csv                # Binary drug sensitivity (846 × 260)
│   ├── drug_info_gdsc.csv      # Drug metadata with target pathways
│   ├── exp_emb_gdsc.csv        # Gene2Vec embeddings (3001 × 200)
│   ├── mut_gdsc.csv            # Gene mutation data
│   ├── cnv_gdsc.csv            # Copy number variation data
│   ├── met_gdsc.csv            # Gene methylation data
│   └── rng.txt                 # Shuffle indices for reproducibility
└── outputs/                # Training outputs (generated)
    ├── results.txt             # Human-readable results summary
    ├── logs.pkl                # Full training logs (metrics, predictions)
    ├── model.pt                # Best model checkpoint
    └── extension2/             # Extension 2 outputs
        ├── cadre/              # CADRE run (logs, model, results)
        └── dot_product/        # DotAttn run (logs, model, results)
```

## Dataset

The GDSC (Genomics of Drug Sensitivity in Cancer) dataset:

| Statistic | Value |
|-----------|-------|
| Cell lines | 846 |
| Drugs | 260 |
| Genes | 3,000 (top variable) |
| Active genes per cell line | 1,500 (top expressed) |
| Drug target pathways | 25 |
| Tested pairs | 179,928 (81.8%) |
| Missing pairs | 40,032 (18.2%) |
| Sensitive labels | 32.2% |
| Resistant labels | 67.8% |

### Preprocessing (from paper)

- **Gene expression:** Top 3,000 most variable genes selected from ~20k. Per cell line, top 1,500 highly expressed genes flagged as 1, rest as 0.
- **Drug sensitivity:** Activity area discretized into binary sensitive/resistant labels using the waterfall algorithm.
- **Gene embeddings:** 200-dimensional Gene2Vec embeddings pretrained on Gene Expression Omnibus (GEO).

### Dataset Usage

`dataset.py` wraps the data into PyHealth's `SampleBaseDataset`, providing a uniform interface compatible with the PyHealth ecosystem and a standard PyTorch `DataLoader`. Each sample represents one cell line:

```python
from dataset import GDSCDataset, split_dataset

ds = GDSCDataset(data_dir="originalData")
dataset = ds.to_pyhealth()                         # PyHealth SampleBaseDataset (846 samples)
train_ds, val_ds, test_ds = split_dataset(dataset) # 60/20/20 split

gene_embeddings = ds.get_gene_embeddings()  # (3001, 200) for model init
pathway_info = ds.get_pathway_info()        # pathway ID mappings
```

Sample format:
```python
{
    "patient_id": "COSMIC.906826",       # cell line ID
    "visit_id": "COSMIC.906826",         # same (one per cell line)
    "gene_indices": [2, 3, 4, ...],      # ~1500 active gene indices (1-indexed)
    "labels": [1, 0, 1, ...],            # 260 binary drug sensitivity labels
    "mask": [1, 1, 0, ...],              # 1=tested, 0=missing
    "drug_pathway_ids": [3, 7, ...],     # pathway index per drug
}
```

## Model Architecture

### CADRE (`model.py`)

#### ExpEncoder
- Looks up pretrained Gene2Vec embeddings (frozen) for active genes
- Expands gene embeddings across all 260 drugs
- Adds drug pathway embedding (contextual conditioning)
- Computes multi-head attention (8 heads): `softmax(β^T tanh(W·e_gene + e_pathway))`
- Sums across heads, then weighted-averages gene embeddings per drug
- Applies dropout

#### DrugDecoder
- Learned drug embedding per drug (260 × 200)
- Dot product between encoder output and drug embedding
- Per-drug bias term

#### Loss
- `BCEWithLogitsLoss` masked to tested (cell line, drug) pairs only
- L2 regularization via SGD weight decay

#### Parameters

| Group | Count | Status |
|-------|-------|--------|
| Gene embeddings (E_G) | 600,200 | Frozen (pretrained) |
| Drug embeddings (E_D) | 52,000 | Trainable |
| Pathway embeddings (E_P) | 3,200 | Trainable |
| Attention W (200→128) | 25,728 | Trainable |
| Attention β (128→8) | 1,032 | Trainable |
| Drug bias | 260 | Trainable |
| **Total trainable** | **82,220** | |

---

### Extension 2 — CADREDotAttn (`model_dot_attn.py`)

Replaces CADRE's additive contextual attention with scaled dot-product attention. Instead of conditioning gene importance via an additive pathway embedding, drug embeddings from the decoder act as queries that probe gene key vectors for geometric alignment:

```
Keys    = key_proj(e_gene)          # gene embeddings -> key space
Queries = query_proj(e_drug)        # drug embeddings -> query space
Scores  = Q · K^T / sqrt(d_k)      # scaled dot product
Output  = W_O · concat(heads)
```

Drug embeddings receive gradients from two paths: the prediction dot-product (decoder) and the attention alignment scores (encoder), jointly shaping them to predict response and attend to relevant genes.

#### Parameters

| Group | Count | Status |
|-------|-------|--------|
| Gene embeddings (E_G) | 600,200 | Frozen (pretrained) |
| Drug embeddings (E_D) | 52,000 | Trainable |
| key\_proj (200→512) | 102,400 | Trainable |
| query\_proj (200→512) | 102,400 | Trainable |
| W\_O (512→200) | 102,400 | Trainable |
| Drug bias | 260 | Trainable |
| **Total trainable** | **359,460** | |

## Training

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train CADRE (reduced steps for quick validation, ~50 min on CPU)
python train.py --max_iter 12000 --cpu

# Train CADRE (full paper settings, ~3-5 hours on CPU)
python train.py --cpu

# With CADRE GPU (CUDA)
python train.py

# Train dot-product attention variant (Extension 2)
python train.py --dot_product_attn

# Quick smoke test
python run_extension2.py --max_iter 800 --cpu

# Run Extension 2 comparison (trains both, prints side-by-side table)
python run_extension2.py

```

### Training Configuration

Matches paper Table A2 (GDSC hyperparameters):

| Hyperparameter | Default | Paper |
|----------------|---------|-------|
| Batch size | 8 | 8×260 |
| Max training steps | 48,000 | 48k |
| Max learning rate (η) | 0.3 | 0.3 |
| Weight decay (λ₂) | 3e-4 | 3e-4 |
| Dropout rate (ρ) | 0.6 | 0.6 |
| Embedding dimension (s) | 200 | 200 |
| Attention dimension (q) | 128 | 128 |
| Attention heads (h) | 8 | 8 |
| Data split | 60/20/20 | 60/20/20 |

### OneCycle Learning Rate Policy

Following Section 3.5 of the paper:
- **Warm-up (45%):** LR η/10 → η, momentum 0.95 → 0.85
- **Cool-down (45%):** LR η → η/10, momentum 0.85 → 0.95
- **Annihilation (10%):** LR η/10 → η/100, momentum 0.95

### Missing Value Imputation

Following Section 4.2: during training, missing drug sensitivity labels are filled with the per-drug mode (majority class) of available labels. At evaluation, missing entries are masked out.

### Command-Line Arguments

```
python train.py --help

--data_dir            Path to data directory (default: originalData/)
--output_dir          Path to output directory (default: outputs/)
--embedding_dim       Gene embedding dimension (default: 200)
--attention_size      Attention hidden dimension for CADRE (default: 128)
--attention_head      Number of attention heads (default: 8)
--d_k                 Key/query dim per head for dot-product attention (default: 64)
--dropout_rate        Dropout probability (default: 0.6)
--no_attention        Disable attention (vanilla collaborative filtering)
--no_cntx_attn        Disable contextual attention (SADRE variant)
--dot_product_attn    Use scaled dot-product attention (Extension 2)
--batch_size          Training batch size (default: 8)
--max_iter            Total training steps (default: 48000)
--learning_rate       Max learning rate for OneCycle (default: 0.3)
--weight_decay        L2 regularization coefficient (default: 3e-4)
--eval_every          Evaluate every N epochs (default: 10)
--seed                Random seed (default: 2019)
--cpu                 Force CPU training
```

## Results

### Reproduction + Extension 2 (48k steps)

| Metric | reCADRE (CADRE) | CADREDotAttn | Paper (CADRE) |
|--------|-----------------|--------------|---------------|
| F1 Score | 63.46 | **64.24** | 64.3 ± 0.22 |
| Accuracy | 78.15 | **78.17** | 78.6 ± 0.34 |
| AUROC | 83.25 | **83.32** | 83.4 ± 0.19 |
| AUPR | 70.94 | **71.04** | 70.6 ± 1.30 |
| Precision | 69.86 | 69.02 | — |
| Recall | 58.14 | **60.09** | — |
| Training time | 5049s | **197s** | — |

Both models trained on GDSC, seed=2019, evaluated on the held-out test set (170 cell lines).

**Key findings:**
- reCADRE closely reproduces the paper's reported CADRE numbers across all metrics
- CADREDotAttn matches or marginally outperforms CADRE on every metric, most notably F1 (64.24 vs 63.46) and Recall (60.09 vs 58.14)
- CADREDotAttn trains ~26× faster (197s vs 5049s) due to more parallelisable matrix operations in scaled dot-product attention vs. the sequential additive conditioning in CADRE

### Planned Extension — Cross-Dataset Generalization
Train on GDSC, evaluate on CCLE overlapping drugs to test whether contextual attention (and dot-product attention) produce representations that transfer across datasets. See `cadre-extension-plan.md` for full design.

## Dependencies

Requires **Python 3.12** (recommended). PyHealth 1.x does not support Python 3.13+.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate

# Install core dependencies
pip install torch numpy pandas scikit-learn

# Install pyhealth without its conflicting pandas<2 constraint
pip install "pyhealth>=1.1.0,<2.0.0" --no-deps

# Install pyhealth runtime dependencies
pip install tqdm pandarallel mne torchvision "setuptools<81"
```

Or, if you want to try `pip install -r requirements.txt` directly and it fails on pyhealth, fall back to the `--no-deps` approach above.

## References

- Tao, Y. et al. (2020). Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention. *MLHC 2020*.
- Yang, W. et al. (2013). Genomics of Drug Sensitivity in Cancer (GDSC). *Nucleic Acids Research*.
- Du, J. et al. (2019). Gene2vec: distributed representation of genes based on co-expression. *BMC Genomics*.
- PyHealth Contributors. (2026). PyHealth Documentation. https://pyhealth.readthedocs.io/

## Authors

Natalie Erjavec, Prasanna Murali, Austin Offenberger — University of Illinois, Urbana-Champaign

CS 598: Deep Learning for Healthcare (Spring 2026)
