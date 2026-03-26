"""CADRE model re-implemented for PyHealth's SampleBaseDataset.

Architecture (faithful to Tao et al. 2020):
    1. Gene Embedding Layer   - pretrained Gene2Vec (3001 x 200), frozen
    2. Contextual Attention   - drug pathway conditions gene importance
    3. Collaborative Filtering - learned drug embeddings + dot-product decoder
    4. Prediction Head         - logit per (cell line, drug) pair
    5. Masked BCE Loss         - only scored on tested pairs
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpEncoder(nn.Module):
    """Gene expression encoder with contextual attention.

    Takes gene indices (active genes per cell line) and produces
    drug-specific cell line representations via attention weighted
    by drug target pathway context.

    Shapes through the forward pass:
        gene_indices:  (batch, num_genes)        ~1500 active gene indices
        E:             (batch, num_genes, emb)    gene embeddings
        E expanded:    (batch, num_drugs, num_genes, emb)
        pathway embed: (1, num_drugs, 1, attn_size)  broadcast
        attention:     (batch, num_drugs, num_genes, heads) -> summed -> (batch, num_drugs, num_genes, 1)
        output:        (batch, num_drugs, emb)    weighted gene representation per drug
    """

    def __init__(
        self,
        gene_embeddings,
        num_pathways,
        embedding_dim=200,
        attention_size=128,
        attention_head=8,
        dropout_rate=0.6,
        use_attention=True,
        use_cntx_attn=True,
    ):
        super().__init__()

        self.use_attention = use_attention
        self.use_cntx_attn = use_cntx_attn

        # Gene embedding layer: pretrained, frozen, with padding at index 0
        self.layer_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(gene_embeddings), freeze=True, padding_idx=0
        )

        self.layer_dropout = nn.Dropout(p=dropout_rate)

        if self.use_attention:
            # Linear transform before tanh: (emb_dim) -> (attn_size)
            self.layer_w_0 = nn.Linear(embedding_dim, attention_size, bias=True)

            # Attention scoring: (attn_size) -> (heads)
            self.layer_beta = nn.Linear(attention_size, attention_head, bias=True)

            if self.use_cntx_attn:
                # Drug pathway embedding for contextual conditioning
                self.layer_emb_ptw = nn.Embedding(
                    num_embeddings=num_pathways,
                    embedding_dim=attention_size,
                )

        # Stored for interpretability extraction
        self.attention_weights = None

    def forward(self, gene_indices, ptw_ids):
        """
        Args:
            gene_indices: (batch, num_genes) LongTensor of active gene indices
            ptw_ids:      (1, num_drugs) LongTensor of pathway IDs per drug

        Returns:
            (batch, num_drugs, embedding_dim) drug-specific cell representations
        """
        num_drugs = ptw_ids.shape[1]

        # (batch, num_genes, embedding_dim)
        E = self.layer_emb(gene_indices)

        if self.use_attention:
            # Expand gene embeddings across all drugs
            # (batch, 1, num_genes, emb) -> (batch, num_drugs, num_genes, emb)
            E_exp = E.unsqueeze(1).expand(-1, num_drugs, -1, -1)

            if self.use_cntx_attn:
                # Pathway embeddings: (1, num_drugs, attention_size)
                Ep = self.layer_emb_ptw(ptw_ids)
                # (1, num_drugs, 1, attention_size) -> broadcast to batch & genes
                Ep = Ep.unsqueeze(2)

                # Contextual attention: tanh(W * gene_emb + pathway_emb)
                H = torch.tanh(self.layer_w_0(E_exp) + Ep)
            else:
                # Self-attention only (SADRE variant)
                H = torch.tanh(self.layer_w_0(E_exp))

            # Multi-head attention scores: (batch, num_drugs, num_genes, heads)
            A = self.layer_beta(H)
            A = F.softmax(A, dim=2)

            # Sum across heads: (batch, num_drugs, num_genes, 1)
            A = A.sum(dim=3, keepdim=True)

            # Store for interpretability
            self.attention_weights = A.squeeze(3)  # (batch, num_drugs, num_genes)

            # Weighted sum: A^T @ E -> (batch, num_drugs, 1, emb) -> squeeze
            # A permuted: (batch, num_drugs, 1, num_genes)
            # E_exp:      (batch, num_drugs, num_genes, emb)
            out = torch.matmul(A.permute(0, 1, 3, 2), E_exp)
            out = out.squeeze(2)  # (batch, num_drugs, embedding_dim)

        else:
            # No attention: mean pool genes, replicate for each drug
            out = E.mean(dim=1)  # (batch, embedding_dim)
            out = out.unsqueeze(1).expand(-1, num_drugs, -1)

        out = self.layer_dropout(out)

        return out


class DrugDecoder(nn.Module):
    """Collaborative filtering decoder with learned drug embeddings.

    Computes dot product between drug-specific cell representations
    (from encoder) and learned drug embeddings to produce logits.
    """

    def __init__(self, num_drugs, embedding_dim):
        super().__init__()

        self.layer_emb_drg = nn.Embedding(
            num_embeddings=num_drugs, embedding_dim=embedding_dim
        )
        self.drg_bias = nn.Parameter(torch.zeros(num_drugs))

    def forward(self, cell_repr, drg_ids):
        """
        Args:
            cell_repr: (batch, num_drugs, embedding_dim) from encoder
            drg_ids:   (1, num_drugs) LongTensor [0, 1, ..., num_drugs-1]

        Returns:
            (batch, num_drugs) logits
        """
        # (1, num_drugs, embedding_dim) -> (batch, num_drugs, embedding_dim)
        D = self.layer_emb_drg(drg_ids).expand(cell_repr.shape[0], -1, -1)

        # Element-wise multiply and sum: dot product per drug
        # (batch, num_drugs)
        logits = (cell_repr * D).sum(dim=2)

        # Add per-drug bias
        logits = logits + self.drg_bias.unsqueeze(0)

        return logits


class CADRE(nn.Module):
    """CADRE: Contextual Attention-based Drug REsponse prediction.

    Combines ExpEncoder (gene embeddings + contextual attention) with
    DrugDecoder (collaborative filtering) for multi-task drug sensitivity
    prediction.

    Args:
        gene_embeddings: np.ndarray (3001, 200) pretrained Gene2Vec
        num_drugs:       int, number of drugs (260 for GDSC)
        num_pathways:    int, number of unique drug target pathways (25)
        drug_pathway_ids: list[int], pathway index for each drug
        embedding_dim:   int, gene embedding dimension (default: 200)
        attention_size:  int, attention hidden dimension (default: 128)
        attention_head:  int, number of attention heads (default: 8)
        dropout_rate:    float, dropout probability (default: 0.6)
        use_attention:   bool, enable attention (default: True)
        use_cntx_attn:   bool, enable contextual attention (default: True)
    """

    def __init__(
        self,
        gene_embeddings,
        num_drugs,
        num_pathways,
        drug_pathway_ids,
        embedding_dim=200,
        attention_size=128,
        attention_head=8,
        dropout_rate=0.6,
        use_attention=True,
        use_cntx_attn=True,
    ):
        super().__init__()

        self.num_drugs = num_drugs
        self.embedding_dim = embedding_dim

        # Register pathway IDs as a buffer (moves with model to device, not a parameter)
        self.register_buffer(
            "ptw_ids", torch.LongTensor([drug_pathway_ids])
        )  # (1, num_drugs)

        # Register drug index range as buffer
        self.register_buffer(
            "drg_ids", torch.arange(num_drugs).unsqueeze(0)
        )  # (1, num_drugs)

        self.encoder = ExpEncoder(
            gene_embeddings=gene_embeddings,
            num_pathways=num_pathways,
            embedding_dim=embedding_dim,
            attention_size=attention_size,
            attention_head=attention_head,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            use_cntx_attn=use_cntx_attn,
        )

        self.decoder = DrugDecoder(
            num_drugs=num_drugs, embedding_dim=embedding_dim
        )

        # Masked BCE loss
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, gene_indices, labels=None, mask=None):
        """
        Args:
            gene_indices: (batch, num_genes) LongTensor of active gene indices
            labels:       (batch, num_drugs) FloatTensor, optional for inference
            mask:         (batch, num_drugs) FloatTensor, optional for inference

        Returns:
            dict with:
                "logits": (batch, num_drugs) raw logits
                "probs":  (batch, num_drugs) sigmoid probabilities
                "loss":   scalar, only if labels and mask provided
                "attention": (batch, num_drugs, num_genes) if attention enabled
        """
        # Encode: gene indices -> drug-specific cell representations
        cell_repr = self.encoder(gene_indices, self.ptw_ids)

        # Decode: cell representations -> drug response logits
        logits = self.decoder(cell_repr, self.drg_ids)

        probs = torch.sigmoid(logits)

        result = {"logits": logits, "probs": probs}

        # Compute masked loss if labels provided
        if labels is not None and mask is not None:
            per_element_loss = self.loss_fn(logits, labels)
            result["loss"] = (per_element_loss * mask).sum() / (mask.sum() + 1e-5)

        # Attach attention weights for interpretability
        if self.encoder.attention_weights is not None:
            result["attention"] = self.encoder.attention_weights

        return result

    def get_attention_weights(self):
        """Return last computed attention weights for interpretability."""
        return self.encoder.attention_weights


def collate_fn(batch):
    """Custom collate for DataLoader.

    Pads gene_indices to the same length within a batch and converts
    all fields to tensors.

    Args:
        batch: list of sample dicts from SampleBaseDataset

    Returns:
        dict with batched tensors
    """
    # Find max gene count in this batch for padding
    max_genes = max(len(s["gene_indices"]) for s in batch)

    gene_indices = []
    labels = []
    masks = []
    patient_ids = []

    for s in batch:
        gi = s["gene_indices"]
        # Pad with 0 (padding index) to max length
        padded = gi + [0] * (max_genes - len(gi))
        gene_indices.append(padded)
        labels.append(s["labels"])
        masks.append(s["mask"])
        patient_ids.append(s["patient_id"])

    return {
        "gene_indices": torch.LongTensor(gene_indices),
        "labels": torch.FloatTensor(labels),
        "mask": torch.FloatTensor(masks),
        "patient_ids": patient_ids,
    }
