"""GDSC dataset wrapper for PyHealth's SampleBaseDataset.

Loads the pre-processed CADRE data (originalData/) and wraps it into
PyHealth's SampleBaseDataset format for drug response prediction.

Each sample represents one cell line with multi-label drug sensitivity
targets, preserving CADRE's multi-task collaborative filtering design.
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SampleBaseDataset(Dataset):
    """Minimal drop-in replacement for PyHealth's SampleBaseDataset.

    Holds a list of sample dicts and exposes them via the standard
    PyTorch Dataset interface (len + getitem).
    """

    def __init__(self, samples, dataset_name="", task_name=""):
        self.samples = samples
        self.dataset_name = dataset_name
        self.task_name = task_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GDSCDataset:
    """Loads GDSC originalData and produces a PyHealth SampleBaseDataset.

    Data layout expected in `data_dir`:
        exp_gdsc.csv        - Binary gene expression (cell lines x 3000 genes)
        gdsc.csv            - Binary drug sensitivity (cell lines x 260 drugs)
        drug_info_gdsc.csv  - Drug metadata with target pathways
        exp_emb_gdsc.csv    - Pretrained Gene2Vec embeddings (3001 x 200)
        rng.txt             - Shuffle indices for reproducibility

    Each sample dict contains:
        patient_id       : str   - cell line COSMIC ID
        visit_id         : str   - same as patient_id (one visit per cell line)
        gene_indices     : list[int] - indices of highly expressed genes (~1500)
        labels           : list[int] - binary drug sensitivity (260,)
        mask             : list[int] - 1 if drug was tested, 0 if NaN (260,)
        drug_pathway_ids : list[int] - pathway index per drug (260,)
    """

    def __init__(self, data_dir="originalData", seed=2019):
        self.data_dir = data_dir
        self.seed = seed
        self._load_data()

    def _load_data(self):
        """Load and align all data files."""
        # Gene expression: 1014 cell lines x 3000 genes, binary
        self.exp = pd.read_csv(
            os.path.join(self.data_dir, "exp_gdsc.csv"), index_col=0
        )

        # Drug sensitivity: 846 cell lines x 260 drugs, binary with NaN
        self.tgt = pd.read_csv(
            os.path.join(self.data_dir, "gdsc.csv"), index_col=0
        )

        # Drug metadata: 266 drugs with target pathway info
        self.drug_info = pd.read_csv(
            os.path.join(self.data_dir, "drug_info_gdsc.csv"), index_col=0
        )

        # Gene embeddings: (3001, 200), row 0 is padding zeros
        self.gene_embeddings = np.loadtxt(
            os.path.join(self.data_dir, "exp_emb_gdsc.csv"), delimiter=","
        )

        # Find common cell lines across expression and sensitivity data
        self.common_samples = sorted(
            set(self.exp.index) & set(self.tgt.index)
        )

        # Align data to common samples
        self.exp = self.exp.loc[self.common_samples]
        self.tgt = self.tgt.loc[self.common_samples]

        # Build pathway mapping
        self._build_pathway_mapping()

        # Gene names for interpretability
        self.gene_names = list(self.exp.columns)
        self.drug_ids = list(self.tgt.columns)

    def _build_pathway_mapping(self):
        """Map each drug to a pathway integer ID."""
        id2pw = dict(
            zip(self.drug_info.index, self.drug_info["Target pathway"])
        )

        # Get pathway for each drug column in tgt
        self.drug_pathways = [
            id2pw.get(int(c), "Unknown") for c in self.tgt.columns
        ]

        # Build pathway-to-id mapping
        unique_pathways = sorted(set(self.drug_pathways))
        self.pathway2id = {pw: i for i, pw in enumerate(unique_pathways)}

        # Integer pathway IDs aligned with drug columns
        self.drug_pathway_ids = [
            self.pathway2id[pw] for pw in self.drug_pathways
        ]

    def get_gene_embeddings(self):
        """Return pretrained gene embeddings (3001 x 200).

        Row 0 is the padding vector (zeros). Rows 1..3000 correspond
        to gene indices used in gene_indices fields.
        """
        return self.gene_embeddings

    def get_pathway_info(self):
        """Return pathway metadata."""
        return {
            "pathway2id": self.pathway2id,
            "id2pathway": {v: k for k, v in self.pathway2id.items()},
            "num_pathways": len(self.pathway2id),
            "drug_pathway_ids": self.drug_pathway_ids,
        }

    def to_dataset(self):
        """Build a SampleBaseDataset (PyTorch Dataset) from loaded GDSC data.

        Returns:
            SampleBaseDataset with 846 samples (one per cell line).
        """
        samples = []

        for cell_line in self.common_samples:
            gene_vec = self.exp.loc[cell_line].values
            # Gene indices are 1-indexed (0 reserved for padding in embedding)
            gene_indices = (np.where(gene_vec == 1)[0] + 1).tolist()

            sensitivity = self.tgt.loc[cell_line]
            mask = sensitivity.notnull().astype(int).values.tolist()
            labels = sensitivity.fillna(0).astype(int).values.tolist()

            samples.append(
                {
                    "patient_id": cell_line,
                    "visit_id": cell_line,
                    "gene_indices": gene_indices,
                    "labels": labels,
                    "mask": mask,
                    "drug_pathway_ids": self.drug_pathway_ids,
                }
            )

        return SampleBaseDataset(samples, "GDSC", "drug_response_prediction")

    def summary(self):
        """Print dataset summary statistics."""
        total_pairs = len(self.common_samples) * len(self.tgt.columns)
        tested = self.tgt.notnull().sum().sum()
        sensitive = (self.tgt == 1).sum().sum()
        resistant = (self.tgt == 0).sum().sum()

        print(f"GDSC Dataset Summary")
        print(f"  Cell lines:       {len(self.common_samples)}")
        print(f"  Drugs:            {len(self.tgt.columns)}")
        print(f"  Genes:            {len(self.exp.columns)}")
        print(f"  Active genes/cell:{int(self.exp.sum(axis=1).mean())}")
        print(f"  Total pairs:      {total_pairs}")
        print(f"  Tested pairs:     {int(tested)} ({tested/total_pairs:.1%})")
        print(f"  Missing pairs:    {int(total_pairs - tested)} ({(total_pairs - tested)/total_pairs:.1%})")
        print(f"  Sensitive:        {int(sensitive)} ({sensitive/tested:.1%})")
        print(f"  Resistant:        {int(resistant)} ({resistant/tested:.1%})")
        print(f"  Pathways:         {len(self.pathway2id)}")
        print(f"  Embedding shape:  {self.gene_embeddings.shape}")


def split_dataset(dataset, ratios=(0.6, 0.2, 0.2), seed=2019):
    """Split a SampleBaseDataset by patient_id (cell line).

    Args:
        dataset: SampleBaseDataset from GDSCDataset.to_pyhealth()
        ratios: (train, val, test) split ratios
        seed: random seed for reproducibility

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    samples = dataset.samples
    n = len(samples)

    # Shuffle patient indices
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * ratios[0])
    n_val = int(n * (ratios[0] + ratios[1]))

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_val].tolist()
    test_idx = indices[n_val:].tolist()

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_ds = SampleBaseDataset(train_samples, "GDSC", "drug_response_prediction")
    val_ds = SampleBaseDataset(val_samples, "GDSC", "drug_response_prediction")
    test_ds = SampleBaseDataset(test_samples, "GDSC", "drug_response_prediction")

    return train_ds, val_ds, test_ds
