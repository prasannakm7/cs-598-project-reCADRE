"""Extension 2: CADRE variant with scaled dot-product attention.

Replaces CADRE's additive contextual attention:
    score_g = W_β · tanh(W_α · e_gene + e_pathway)   [CADRE]
with transformer-style scaled dot-product attention:
    score_g = (W_Q · e_drug) · (W_K · e_gene)^T / sqrt(d_k)   [this file]

The drug embedding from DrugDecoder acts as the query, probing gene key
vectors for geometric alignment.  Gradients flow back through both the
decoder (prediction path) and the encoder (attention query path), so drug
embeddings are jointly shaped to predict response AND to attend to relevant
genes — the key inductive-bias difference from CADRE's additive approach.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DrugDecoder


class DotProductExpEncoder(nn.Module):
    """Gene expression encoder using scaled dot-product attention.

    Architecture:
        Keys    = key_proj(e_gene)    (gene embeddings projected to key space)
        Queries = query_proj(e_drug)  (drug embeddings projected to query space)
        Scores  = Q · K^T / sqrt(d_k)
        Output  = W_O · concat(head_1, ..., head_H)

    Padding positions (gene_index == 0) are masked to -inf before softmax
    so they receive zero attention weight, unlike CADRE where padded genes
    still receive a small weight due to the additive pathway bias.

    Shapes through the forward pass:
        gene_indices:    (batch, max_genes)
        G:               (batch, max_genes, emb_dim)            gene embeddings
        K:               (batch, H, max_genes, d_k)             gene keys
        Q:               (batch, H, num_drugs, d_k)             drug queries
        scores:          (batch, H, num_drugs, max_genes)
        attn:            (batch, H, num_drugs, max_genes)       after softmax
        context:         (batch, H, num_drugs, d_k)
        output:          (batch, num_drugs, emb_dim)
    """

    def __init__(
        self,
        gene_embeddings,
        num_drugs,
        embedding_dim=200,
        num_heads=8,
        d_k=64,
        dropout_rate=0.6,
    ):
        super().__init__()

        self.num_drugs = num_drugs
        self.num_heads = num_heads
        self.d_k = d_k
        self.embedding_dim = embedding_dim

        # Gene embedding layer: pretrained Gene2Vec, frozen, padding at index 0
        self.layer_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(gene_embeddings), freeze=True, padding_idx=0
        )

        # key_proj:   gene embeddings -> key vectors  (H * d_k)
        self.key_proj = nn.Linear(embedding_dim, num_heads * d_k, bias=False)
        # query_proj: drug embeddings -> query vectors (H * d_k)
        self.query_proj = nn.Linear(embedding_dim, num_heads * d_k, bias=False)

        # Output projection: concat heads -> embedding_dim
        self.W_O = nn.Linear(num_heads * d_k, embedding_dim, bias=False)

        self.layer_dropout = nn.Dropout(p=dropout_rate)

        # Stored for interpretability (mean across heads, after masking)
        self.attention_weights = None

    def forward(self, gene_indices, drug_embeddings):
        """
        Args:
            gene_indices:    (batch, max_genes) LongTensor; 0 = padding
            drug_embeddings: (num_drugs, embedding_dim) from DrugDecoder

        Returns:
            (batch, num_drugs, embedding_dim)
        """
        batch_size, max_genes = gene_indices.shape
        num_drugs = drug_embeddings.shape[0]
        H = self.num_heads
        dk = self.d_k

        G = self.layer_emb(gene_indices)

        pad_mask = (gene_indices == 0).unsqueeze(1).unsqueeze(2)

        K = self.key_proj(G).view(batch_size, max_genes, H, dk).transpose(1, 2)

        Q = self.query_proj(drug_embeddings).view(num_drugs, H, dk).permute(1, 0, 2)
        Q = Q.unsqueeze(0).expand(batch_size, -1, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)

        scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)

        self.attention_weights = attn.mean(dim=1).detach()


        context = torch.matmul(attn, K)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, num_drugs, H * dk)
        out = self.W_O(context)

        out = self.layer_dropout(out)
        return out


class CADREDotAttn(nn.Module):
    """CADRE with scaled dot-product attention.


    Drug embeddings from DrugDecoder serve as attention queries so they
    receive gradient from two sources — the prediction dot-product and the
    attention alignment scores — shaping them jointly.

    Args:
        gene_embeddings: np.ndarray (3001, 200) pretrained Gene2Vec
        num_drugs:       int (260 for GDSC)
        embedding_dim:   int (default: 200)
        num_heads:       int, attention heads (default: 8)
        d_k:             int, key/query dim per head (default: 64)
        dropout_rate:    float (default: 0.6)
    """

    def __init__(
        self,
        gene_embeddings,
        num_drugs,
        embedding_dim=200,
        num_heads=8,
        d_k=64,
        dropout_rate=0.6,
    ):
        super().__init__()

        self.num_drugs = num_drugs
        self.embedding_dim = embedding_dim

        # Drug index range buffer (same pattern as CADRE)
        self.register_buffer(
            "drg_ids", torch.arange(num_drugs).unsqueeze(0)
        )  # (1, num_drugs)

        self.encoder = DotProductExpEncoder(
            gene_embeddings=gene_embeddings,
            num_drugs=num_drugs,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            d_k=d_k,
            dropout_rate=dropout_rate,
        )

        self.decoder = DrugDecoder(
            num_drugs=num_drugs, embedding_dim=embedding_dim
        )

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, gene_indices, labels=None, mask=None):
        """
        Args:
            gene_indices: (batch, max_genes) LongTensor of active gene indices
            labels:       (batch, num_drugs) FloatTensor, optional
            mask:         (batch, num_drugs) FloatTensor, optional

        Returns:
            dict with: logits, probs, loss (if labels given), attention
        """
        drug_emb = self.decoder.layer_emb_drg(self.drg_ids).squeeze(0)

        # Encode: gene indices + drug queries -> (batch, num_drugs, emb_dim)
        cell_repr = self.encoder(gene_indices, drug_emb)

        # Decode: (batch, num_drugs, emb_dim) -> (batch, num_drugs) logits
        logits = self.decoder(cell_repr, self.drg_ids)
        probs = torch.sigmoid(logits)

        result = {"logits": logits, "probs": probs}

        if labels is not None and mask is not None:
            per_element_loss = self.loss_fn(logits, labels)
            result["loss"] = (per_element_loss * mask).sum() / (mask.sum() + 1e-5)

        if self.encoder.attention_weights is not None:
            result["attention"] = self.encoder.attention_weights

        return result

    def get_attention_weights(self):
        """Return last computed attention weights for interpretability."""
        return self.encoder.attention_weights
