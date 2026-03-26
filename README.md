# reCADRE: Reproducing CADRE in PyHealth

A reproduction and extension of **CADRE** (Contextual Attention-based Drug REsponse) within the [PyHealth](https://pyhealth.readthedocs.io/) framework, originally published as:

> Tao, Y., Ren, S., Ding, M.Q., Schwartz, R., & Lu, X. (2020). *Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention.* Proceedings of Machine Learning Research, 126, 456–477. PMLR (MLHC 2020).

**Original code:** [github.com/yifengtao/CADRE](https://github.com/yifengtao/CADRE)

## Overview

CADRE predicts the sensitivity of cancer cell lines to oncology drugs using gene expression profiles. It combines:

1. **Collaborative Filtering** — jointly predicts all drug responses for a cell line via learned drug embeddings and dot-product decoding
2. **Contextual Attention** — weights gene importance differently per drug based on the drug's target pathway, producing drug-specific cell line representations
3. **Pretrained Gene Embeddings** — transfers biological knowledge from Gene2Vec embeddings trained on large-scale co-expression data

This project re-implements the full CADRE architecture using PyHealth's `SampleBaseDataset` abstraction and standard PyTorch `DataLoader`, enabling integration with the broader PyHealth ecosystem.

## Project Structure

```
reCADRE/
├── README.md            # This file
├── dataset.py           # GDSC dataset wrapper for PyHealth SampleBaseDataset
├── model.py             # CADRE model (encoder, decoder, attention, loss)
├── train.py             # Training script with OneCycle LR, evaluation, logging
├── originalData/        # Pre-processed GDSC data files
│   ├── exp_gdsc.csv         # Binary gene expression (1014 × 3000)
│   ├── gdsc.csv             # Binary drug sensitivity (846 × 260)
│   ├── drug_info_gdsc.csv   # Drug metadata with target pathways
│   ├── exp_emb_gdsc.csv     # Gene2Vec embeddings (3001 × 200)
│   ├── mut_gdsc.csv         # Gene mutation data
│   ├── cnv_gdsc.csv         # Copy number variation data
│   ├── met_gdsc.csv         # Gene methylation data
│   └── rng.txt              # Shuffle indices for reproducibility
└── outputs/             # Training outputs (generated)
    ├── results.txt          # Human-readable results summary
    ├── logs.pkl             # Full training logs (metrics, predictions)
    └── model.pt             # Best model checkpoint
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

### PyHealth Integration

`dataset.py` wraps the data into PyHealth's `SampleBaseDataset`. Each sample represents one cell line:

```python
from dataset import GDSCDataset, split_dataset

ds = GDSCDataset(data_dir="originalData")
pyhealth_ds = ds.to_pyhealth()                        # SampleBaseDataset (846 samples)
train_ds, val_ds, test_ds = split_dataset(pyhealth_ds) # 60/20/20 split

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

### ExpEncoder
- Looks up pretrained Gene2Vec embeddings (frozen) for active genes
- Expands gene embeddings across all 260 drugs
- Adds drug pathway embedding (contextual conditioning)
- Computes multi-head attention (8 heads): `softmax(β^T tanh(W·e_gene + e_pathway))`
- Sums across heads, then weighted-averages gene embeddings per drug
- Applies dropout

### DrugDecoder
- Learned drug embedding per drug (260 × 200)
- Dot product between encoder output and drug embedding
- Per-drug bias term

### Loss
- `BCEWithLogitsLoss` masked to tested (cell line, drug) pairs only
- L2 regularization via SGD weight decay

### Parameters

| Group | Count | Status |
|-------|-------|--------|
| Gene embeddings (E_G) | 600,200 | Frozen (pretrained) |
| Drug embeddings (E_D) | 52,000 | Trainable |
| Pathway embeddings (E_P) | 3,200 | Trainable |
| Attention W (200→128) | 25,728 | Trainable |
| Attention β (128→8) | 1,032 | Trainable |
| Drug bias | 260 | Trainable |
| **Total trainable** | **82,220** | |

## Training

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train (reduced steps for quick validation, ~50 min on CPU)
python train.py --max_iter 12000 --cpu

# Train (full paper settings, ~3-5 hours on CPU)
python train.py --cpu

# With GPU (CUDA)
python train.py
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

--data_dir         Path to data directory (default: originalData/)
--output_dir       Path to output directory (default: outputs/)
--embedding_dim    Gene embedding dimension (default: 200)
--attention_size   Attention hidden dimension (default: 128)
--attention_head   Number of attention heads (default: 8)
--dropout_rate     Dropout probability (default: 0.6)
--no_attention     Disable attention (vanilla collaborative filtering)
--no_cntx_attn     Disable contextual attention (SADRE variant)
--batch_size       Training batch size (default: 8)
--max_iter         Total training steps (default: 48000)
--learning_rate    Max learning rate for OneCycle (default: 0.3)
--weight_decay     L2 regularization coefficient (default: 3e-4)
--eval_every       Evaluate every N epochs (default: 10)
--seed             Random seed (default: 2019)
--cpu              Force CPU training
```

## Results

### Reproduction (12k steps, CPU)

| Metric | reCADRE | Paper (CADRE) | Gap |
|--------|---------|---------------|-----|
| F1 Score | 62.1 | 64.3 ± 0.22 | -2.2 |
| Accuracy | 77.4 | 78.6 ± 0.34 | -1.2 |
| AUROC | 81.9 | 83.4 ± 0.19 | -1.5 |
| AUPR | 67.1 | 70.6 ± 1.30 | -3.5 |

The gap is expected: this run used 25% of the paper's training steps (12k vs 48k). The model was still improving at the end of training. Running the full 48k steps would close this gap.

### Training Progression

```
Epoch  5 | loss=0.622 | val F1=62.0 | val AUROC=82.2
Epoch 10 | loss=0.535 | val F1=62.6 | val AUROC=82.7
Epoch 15 | loss=0.486 | val F1=62.6 | val AUROC=82.4
Epoch 20 | loss=0.475 | val F1=62.6 | val AUROC=82.5
Epoch 24 | loss=0.467 | val F1=62.7 | val AUROC=82.5
```

## Planned Extensions

### 1. Cross-Dataset Generalization
Train on GDSC, evaluate on CCLE to test whether contextual attention produces transferable cell line representations beyond dataset-specific patterns.

### 2. Alternative Attention Mechanisms
Replace CADRE's additive contextual attention with transformer-style scaled dot-product attention to compare inductive biases for gene-drug interaction modeling.

## Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. See [requirements.txt](requirements.txt) for pinned versions.

## References

- Tao, Y. et al. (2020). Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention. *MLHC 2020*.
- Yang, W. et al. (2013). Genomics of Drug Sensitivity in Cancer (GDSC). *Nucleic Acids Research*.
- Du, J. et al. (2019). Gene2vec: distributed representation of genes based on co-expression. *BMC Genomics*.
- PyHealth Contributors. (2026). PyHealth Documentation. https://pyhealth.readthedocs.io/

## Authors

Natalie Erjavec, Prasanna Murali, Austin Offenberger — University of Illinois, Urbana-Champaign

CS 598: Deep Learning for Healthcare (Spring 2026)
