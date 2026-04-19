"""Rebuild CCLE data with guaranteed gene2vec embedding alignment.

Fix #5 from root-cause analysis: instead of selecting CCLE's top-3000 variance
genes (ENSG IDs) and then copying GDSC embeddings blindly, we:
  1. Load CCLE RNA-seq (ENSG → HUGO via mygene)
  2. Intersect CCLE genes with GDSC's 3000 symbols (which have gene2vec rows)
  3. Keep that intersection as the CCLE gene set (up to 3000 genes)
  4. For each cell line, binarize (top 1500 expressed genes = 1, rest = 0)
  5. exp_emb_ccle.csv row i = GDSC embedding for CCLE's i-th gene symbol

This guarantees every kept CCLE gene has a real gene2vec embedding, closing
the 2/3-zero-embedding gap that limited fix_ccle_emb.py.

Writes to ccleDataAligned/ — leaves ccleData/ and ccleDataPaper/ untouched so
existing Extensions 1/2 and earlier experiments still work.
"""

import os
import sys
import numpy as np
import pandas as pd


def map_ensg_to_symbol(ensg_list):
    """Query mygene.info for ENSG→HUGO symbol mapping."""
    import mygene
    print(f"Mapping {len(ensg_list)} ENSG IDs to HUGO symbols via mygene.info...")
    mg = mygene.MyGeneInfo()
    result = mg.querymany(
        ensg_list,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        returnall=False,
        as_dataframe=False,
    )
    mapping = {}
    for r in result:
        if not r.get("notfound") and "symbol" in r:
            mapping[r["query"]] = r["symbol"]
    print(f"  Mapped {len(mapping)} / {len(ensg_list)}")
    return mapping


def load_and_clean_ccle_exp(raw_path):
    """Load CCLE RNA-seq TPM, clean column names, drop duplicate cell lines."""
    print(f"Loading {raw_path} (this is ~735 MB, takes ~60 s)...")
    df = pd.read_csv(raw_path, sep="\t", index_col=0)
    df = df.drop(columns=["transcript_ids"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
    # Column names are "CELLLINE_TISSUE" → keep cell line part
    clean_cols = [str(c).split("_")[0] for c in df.columns]
    df.columns = clean_cols
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    print(f"  shape after cleanup: {df.shape} (genes × cell lines)")
    return df


def main():
    src_raw = "originalData/ccle/CCLE_RNAseq_rsem_genes_tpm_20180929.txt"
    src_drug = "ccleData/ccle.csv"          # already binarized (labels will be
                                            # flipped in CCLEDataset loader)
    src_drug_info_paper = "ccleDataPaper/drug_info_ccle.csv"  # paper Table A5
                                                              # pathway mapping
    dst = "ccleDataAligned"
    os.makedirs(dst, exist_ok=True)

    # ---- 1. CCLE RNA-seq ----
    ccle_exp = load_and_clean_ccle_exp(src_raw)  # (genes × cell lines)
    ensg_with_version = list(ccle_exp.index)
    ensg_clean = [g.split(".")[0] for g in ensg_with_version]

    # ---- 2. ENSG → symbol ----
    ensg2sym = map_ensg_to_symbol(ensg_clean)

    # ---- 3. GDSC gene order (rows 1..3000 of exp_emb_gdsc.csv) ----
    gdsc_exp = pd.read_csv("originalData/exp_gdsc.csv", index_col=0, nrows=0)
    gdsc_symbols = list(gdsc_exp.columns)
    sym2gdsc_row = {sym: i + 1 for i, sym in enumerate(gdsc_symbols)}  # +1 for padding row 0

    # ---- 4. For each CCLE ENSG, assign its symbol if that symbol is in GDSC ----
    ccle_rows_kept = []
    ccle_symbols_kept = []
    for ensg_full, ensg_short in zip(ensg_with_version, ensg_clean):
        sym = ensg2sym.get(ensg_short)
        if sym and sym in sym2gdsc_row and sym not in ccle_symbols_kept:
            ccle_rows_kept.append(ensg_full)
            ccle_symbols_kept.append(sym)
    print(f"\nCCLE genes with symbol in GDSC's 3000: {len(ccle_rows_kept)}")
    print(f"  (pool to draw from, before variance selection)")

    # Subset expression to those rows, re-index by HUGO symbol
    exp_subset = ccle_exp.loc[ccle_rows_kept].copy()
    exp_subset.index = ccle_symbols_kept

    # ---- 5. Top-3000 variance within this aligned pool ----
    gene_var = exp_subset.var(axis=1)
    # If fewer than 3000, keep all; otherwise top 3000
    if len(gene_var) > 3000:
        top_genes = gene_var.nlargest(3000).index
        exp_subset = exp_subset.loc[top_genes]
    print(f"\nFinal CCLE gene set: {exp_subset.shape[0]} genes "
          f"(target 3000, all have gene2vec embeddings)")

    # ---- 6. Binarize: per cell line, top 1500 expressed → 1 ----
    n_top = 1500
    exp_bin = np.zeros(exp_subset.shape, dtype=int)  # (genes × cells)
    for j, cell_line in enumerate(exp_subset.columns):
        col = exp_subset.iloc[:, j].values
        top_idx = np.argsort(-col)[:n_top]
        exp_bin[top_idx, j] = 1

    # Transpose: (cells × genes) and save
    exp_bin_df = pd.DataFrame(
        exp_bin.T,
        index=exp_subset.columns,
        columns=list(exp_subset.index),  # HUGO symbols now
    )
    exp_bin_df.to_csv(f"{dst}/exp_ccle.csv")
    print(f"\n✓ {dst}/exp_ccle.csv "
          f"({exp_bin_df.shape[0]} cells × {exp_bin_df.shape[1]} genes)")

    # ---- 7. Copy drug sensitivity (labels flipped later by CCLEDataset) ----
    import shutil
    shutil.copy(src_drug, f"{dst}/ccle.csv")
    print(f"✓ {dst}/ccle.csv copied")

    # ---- 8. Paper Table A5 pathway mapping ----
    shutil.copy(src_drug_info_paper, f"{dst}/drug_info_ccle.csv")
    print(f"✓ {dst}/drug_info_ccle.csv copied from ccleDataPaper/")

    # ---- 9. Aligned embedding matrix ----
    emb_gdsc = np.loadtxt("originalData/exp_emb_gdsc.csv", delimiter=",")
    emb_ccle = np.zeros((exp_bin_df.shape[1] + 1, emb_gdsc.shape[1]))  # +1 for padding
    for i, sym in enumerate(list(exp_subset.index)):
        emb_ccle[i + 1] = emb_gdsc[sym2gdsc_row[sym]]
    np.savetxt(f"{dst}/exp_emb_ccle.csv", emb_ccle, delimiter=",", fmt="%.6f")
    nonzero = (np.linalg.norm(emb_ccle, axis=1) > 0).sum()
    print(f"✓ {dst}/exp_emb_ccle.csv ({nonzero} / {emb_ccle.shape[0]} "
          f"non-zero rows — 100% gene2vec coverage)")

    print(f"\n✓ Built {dst}/ — use with:")
    print(f"    python3 train.py --dataset ccle --data_dir {dst} \\")
    print(f"      --max_iter 768000 --learning_rate 0.05 --dropout_rate 0.5 \\")
    print(f"      --attention_size 100 --output_dir outputs/ccle_aligned --seed 2019")


if __name__ == "__main__":
    main()
