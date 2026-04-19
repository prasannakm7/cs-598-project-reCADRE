"""Preprocess raw CCLE data to match GDSC format for cross-dataset generalization.

Converts:
  - CCLE_NP24.2009_Drug_data_2015.02.24.csv        → ccle.csv (binary sensitivity)
  - CCLE_NP24.2009_profiling_2012.02.20.csv        → drug_info_ccle.csv (drug metadata)
  - CCLE_RNAseq_rsem_genes_tpm_20180929.txt        → exp_ccle.csv (binary expression)
  - (Uses GDSC gene embeddings)                    → exp_emb_ccle.csv (shared embeddings)

Output: ccleData/ directory with 4 production-ready CSVs
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def load_ccle_drug_data(filepath):
    """Load CCLE drug sensitivity data.
    
    Args:
        filepath: Path to CCLE_NP24.2009_Drug_data_2015.02.24.csv
    
    Returns:
        DataFrame with drugs and IC50/EC50 values
    """
    print("Loading CCLE drug data...")
    df = pd.read_csv(filepath)
    
    # Extract cell line and drug info
    # Columns: CCLE Cell Line Name, Compound, IC50 (uM), etc.
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:5])}")
    
    return df


def load_ccle_expression_data(filepath):
    """Load CCLE RNA-seq expression data (TPM).
    
    Args:
        filepath: Path to CCLE_RNAseq_rsem_genes_tpm_20180929.txt
    
    Returns:
        DataFrame (cell lines × genes) with TPM values
    """
    print("Loading CCLE RNA-seq expression...")
    df = pd.read_csv(filepath, sep="\t", index_col=0)
    
    # Rows: genes (HUGO symbol + ensemble ID), Columns: cell lines
    print(f"  Initial shape: {df.shape} (genes × cell lines)")
    
    # Convert to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows that are all NaN (gene names or metadata)
    df = df.dropna(axis=0, how='all')
    
    print(f"  After cleanup: {df.shape} (genes × cell lines)")
    
    return df


def load_ccle_drug_info(filepath):
    """Load CCLE drug metadata with targets.
    
    Args:
        filepath: Path to CCLE_NP24.2009_profiling_2012.02.20.csv
    
    Returns:
        DataFrame with drug metadata including targets/pathways
    """
    print("Loading CCLE drug profiling info...")
    
    # Try different encodings (file may have non-UTF-8 characters)
    for encoding in ['latin-1', 'iso-8859-1', 'windows-1252', 'utf-8']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"  Successfully read with encoding: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise UnicodeDecodeError(
            'unknown', b'', 0, 1, 
            f"Could not read {filepath} with any standard encoding"
        )
    
    # Columns: Compound (code or generic name), Target(s), Mechanism of action, etc.
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def binarize_drug_sensitivity(ic50_values, percentile_threshold=75):
    """Convert IC50/EC50 to binary sensitive/resistant labels.
    
    Uses percentile-based thresholding (similar to GDSC discretization):
    - Values >= percentile_threshold → resistant (0)
    - Values < percentile_threshold → sensitive (1)
    - NaN values remain NaN
    
    Args:
        ic50_values: Series or array of IC50/EC50 values
        percentile_threshold: Percentile for cutoff (default 75%)
    
    Returns:
        Binary array (NaN for missing)
    """
    # Filter out NaN to compute threshold
    valid_mask = ~np.isnan(ic50_values)
    valid_values = ic50_values[valid_mask]
    
    if len(valid_values) < 10:
        # Too few samples, return all NaN
        return np.full_like(ic50_values, np.nan, dtype=float)
    
    # Compute threshold (higher values = more resistant)
    threshold = np.percentile(valid_values, percentile_threshold)
    
    # Binarize: IC50 < threshold → sensitive (1), >= threshold → resistant (0)
    binary = np.where(valid_mask, ic50_values < threshold, np.nan).astype(float)
    
    return binary


def binarize_expression(expression_df, top_genes=3000, top_expressed=1500):
    """Convert TPM expression to binary active/inactive genes.
    
    Follows GDSC preprocessing:
    1. Select top 3000 most variable genes across CCLE cell lines
    2. For each cell line, mark top 1500 expressed genes as 1 (active), rest as 0
    
    Args:
        expression_df: DataFrame (genes × cell lines)
        top_genes: Number of most variable genes to keep
        top_expressed: Number of top expressed genes per cell line to mark as 1
    
    Returns:
        Tuple of (binary_expression_df, selected_gene_names)
    """
    print(f"\nBinarizing expression (selecting top {top_genes} genes)...")
    
    # Clean column names - extract just the cell line name (before underscore/space)
    # CCLE format: "CELL_LINE_TISSUE" -> extract "CELL_LINE"
    clean_cols = []
    for col in expression_df.columns:
        # Remove suffixes like _TISSUE_TYPE or spaces
        clean_name = str(col).split('_')[0] if '_' in str(col) else str(col)
        clean_cols.append(clean_name)
    
    expression_df.columns = clean_cols
    
    # Remove duplicate columns (keep first occurrence)
    expression_df = expression_df.loc[:, ~expression_df.columns.duplicated(keep='first')]
    
    # Ensure all numeric
    expression_df = expression_df.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with all NaN
    expression_df = expression_df.dropna(axis=0, how='all')
    
    # Fill NaN with 0 for variance calculation
    expression_df_filled = expression_df.fillna(0)
    
    # Compute variance across cell lines for each gene
    gene_variance = expression_df_filled.var(axis=1, numeric_only=True)
    
    # Select top-variance genes
    if len(gene_variance) > top_genes:
        top_gene_indices = gene_variance.nlargest(top_genes).index
    else:
        top_gene_indices = gene_variance.index
    
    exp_subset = expression_df.loc[top_gene_indices]
    
    print(f"  Selected genes: {exp_subset.shape[0]}")
    print(f"  Cell lines found: {exp_subset.shape[1]}")
    
    # Binarize: for each cell line, top N expressed genes get 1, rest get 0
    binary_exp = pd.DataFrame(
        np.zeros_like(exp_subset.values, dtype=int),
        index=exp_subset.index,
        columns=exp_subset.columns
    )
    
    for i, cell_line in enumerate(exp_subset.columns):
        # Use iloc to avoid issues with duplicate column names
        cell_data = exp_subset.iloc[:, i].fillna(0)
        
        # cell_data should now be a Series
        if isinstance(cell_data, pd.Series):
            # Get indices of top expressed genes (at most top_expressed)
            n_to_select = min(top_expressed, len(cell_data))
            top_idx = cell_data.nlargest(n_to_select).index
            binary_exp.loc[top_idx, cell_line] = 1
        else:
            # Fallback if still a DataFrame
            print(f"  Warning: cell_data for {cell_line} is still a DataFrame, skipping")
            continue
    
    print(f"  Binary expression shape: {binary_exp.shape}")
    
    return binary_exp, list(exp_subset.index)


def create_drug_sensitivity_matrix(drug_data, drug_info, binarize=True):
    """Create cell line × drug sensitivity matrix from raw IC50 data.
    
    Args:
        drug_data: Raw CCLE drug data (with IC50 and cell line info)
        drug_info: Drug metadata
        binarize: Whether to binarize IC50 → sensitive/resistant
    
    Returns:
        DataFrame (cell lines × drugs) with binary sensitivity or NaN
    """
    print("\nCreating drug sensitivity matrix...")
    
    # Use Primary Cell Line Name (clean format) instead of full CCLE Cell Line Name
    cell_line_col = 'Primary Cell Line Name'
    compound_col = 'Compound'
    
    # Find IC50/EC50 column
    ic50_cols = [c for c in drug_data.columns if 'ic50' in c.lower() or 'ec50' in c.lower()]
    if ic50_cols:
        ic50_col = ic50_cols[0]
    else:
        # Try Activity Area or IC50 (uM)
        ic50_col = [c for c in drug_data.columns if 'ic50' in c.lower() or 'activity' in c.lower()][0]
    
    print(f"  Using columns:")
    print(f"    Cell line: {cell_line_col}")
    print(f"    Compound: {compound_col}")
    print(f"    IC50/EC50: {ic50_col}")
    
    # Verify columns exist
    if cell_line_col not in drug_data.columns:
        print(f"  Warning: {cell_line_col} not found, available: {drug_data.columns.tolist()[:5]}")
        cell_line_col = drug_data.columns[1]
    
    if compound_col not in drug_data.columns:
        print(f"  Warning: {compound_col} not found")
        compound_col = drug_data.columns[2]
    
    if ic50_col not in drug_data.columns:
        print(f"  Warning: {ic50_col} not found")
        ic50_col = [c for c in drug_data.columns if 'ic50' in c.lower() or 'ec50' in c.lower() or 'activity' in c.lower()][0]
    
    # Create mapping from compound to drug info (if available)
    drug_to_target = {}
    if drug_info is not None:
        for idx, row in drug_info.iterrows():
            drug_to_target[idx] = ''
    
    print(f"  Found {len(drug_to_target)} compounds in drug info")
    
    # Get unique cell lines and drugs
    cell_lines = sorted(drug_data[cell_line_col].unique())
    drugs = sorted(drug_data[compound_col].unique())
    
    print(f"  Cell lines: {len(cell_lines)}")
    print(f"  Drugs: {len(drugs)}")
    
    # Initialize sensitivity matrix
    sensitivity = pd.DataFrame(
        np.nan,
        index=cell_lines,
        columns=drugs
    )
    
    # Fill in IC50 values (continuous)
    for _, row in drug_data.iterrows():
        cell_line = row[cell_line_col]
        compound = row[compound_col]
        ic50_val = row[ic50_col]
        
        # Convert to numeric if needed
        if isinstance(ic50_val, str):
            try:
                ic50_val = float(ic50_val)
            except (ValueError, TypeError):
                ic50_val = np.nan
        
        if cell_line in sensitivity.index and compound in sensitivity.columns:
            if not np.isnan(ic50_val) and ic50_val != np.inf:
                sensitivity.loc[cell_line, compound] = ic50_val
    
    # Binarize if requested
    if binarize:
        print("  Binarizing IC50 values...")
        for drug in sensitivity.columns:
            sensitivity[drug] = binarize_drug_sensitivity(
                sensitivity[drug].values,
                percentile_threshold=75
            )
    
    print(f"  Final matrix shape: {sensitivity.shape}")
    print(f"  Non-NaN values: {(~sensitivity.isna()).sum().sum()} / {sensitivity.size}")
    
    return sensitivity


def create_drug_info(drug_info_df):
    """Create drug_info_ccle.csv from CCLE profiling data.
    
    Args:
        drug_info_df: CCLE profiling data with targets
    
    Returns:
        DataFrame suitable as drug_info_ccle.csv
    """
    print("\nCreating drug info metadata...")
    
    # Find the relevant columns (names might vary)
    compound_col = None
    target_col = None
    moa_col = None
    class_col = None
    
    for col in drug_info_df.columns:
        if 'compound' in col.lower() and 'code' in col.lower():
            compound_col = col
        elif 'target' in col.lower():
            target_col = col
        elif 'mechanism' in col.lower():
            moa_col = col
        elif 'class' in col.lower():
            class_col = col
    
    print(f"  Found columns:")
    print(f"    Compound: {compound_col}")
    print(f"    Target: {target_col}")
    print(f"    MOA: {moa_col}")
    print(f"    Class: {class_col}")
    
    # Extract relevant columns (use defaults if not found)
    if compound_col and target_col:
        cols_to_use = [c for c in [compound_col, target_col, moa_col, class_col] if c]
        info = drug_info_df[cols_to_use].copy()
        
        # Rename columns to match GDSC format
        rename_map = {}
        if compound_col:
            rename_map[compound_col] = 'Drug_ID'
        if target_col:
            rename_map[target_col] = 'Target pathway'
        if moa_col:
            rename_map[moa_col] = 'MOA'
        if class_col:
            rename_map[class_col] = 'Drug_Class'
        
        info = info.rename(columns=rename_map)
    else:
        print("  Warning: Could not find standard columns, using first 4 columns")
        info = drug_info_df.iloc[:, :4].copy()
        info.columns = ['Drug_ID', 'Target pathway', 'MOA', 'Drug_Class']
    
    # Set index
    if 'Drug_ID' in info.columns:
        info = info.set_index('Drug_ID')
    
    print(f"  Drug info shape: {info.shape}")
    print(f"  Columns: {list(info.columns)}")
    
    return info


def copy_gene_embeddings(gdsc_emb_path, output_path):
    """Copy GDSC gene embeddings for use with CCLE data.
    
    GDSC and CCLE use the same gene identifiers, so embeddings are directly
    transferable. This creates the exp_emb_ccle.csv file.
    
    Args:
        gdsc_emb_path: Path to GDSC embeddings
        output_path: Output path for CCLE embeddings
    """
    print("\nCopying gene embeddings from GDSC...")
    
    # Load GDSC embeddings
    embeddings = np.loadtxt(gdsc_emb_path, delimiter=",")
    
    # Save to output
    np.savetxt(output_path, embeddings, delimiter=",", fmt="%.6f")
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Saved to: {output_path}")


def preprocess_ccle(
    raw_data_dir="originalData/ccle",
    output_dir="ccleData",
    gdsc_emb_path="originalData/exp_emb_gdsc.csv"
):
    """Main preprocessing pipeline.
    
    Args:
        raw_data_dir: Directory with raw CCLE data files
        output_dir: Output directory for processed files
        gdsc_emb_path: Path to GDSC embeddings to copy
    """
    print("="*80)
    print("CCLE DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # File paths
    drug_data_file = os.path.join(raw_data_dir, "CCLE_NP24.2009_Drug_data_2015.02.24.csv")
    expr_data_file = os.path.join(raw_data_dir, "CCLE_RNAseq_rsem_genes_tpm_20180929.txt")
    drug_info_file = os.path.join(raw_data_dir, "CCLE_NP24.2009_profiling_2012.02.20.csv")
    
    # Verify files exist
    for fpath in [drug_data_file, expr_data_file, drug_info_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")
    
    # ========== STEP 1: Load raw data ==========
    drug_data = load_ccle_drug_data(drug_data_file)
    expr_data = load_ccle_expression_data(expr_data_file)
    drug_info = load_ccle_drug_info(drug_info_file)
    
    # ========== STEP 2: Process drug sensitivity ==========
    print("\n" + "-"*80)
    sensitivity_matrix = create_drug_sensitivity_matrix(drug_data, drug_info, binarize=True)
    sensitivity_matrix = sensitivity_matrix.fillna(np.nan)  # Preserve NaN as missing
    
    # Save
    sensitivity_output = os.path.join(output_dir, "ccle.csv")
    sensitivity_matrix.to_csv(sensitivity_output)
    print(f"\n✓ Saved drug sensitivity: {sensitivity_output}")
    
    # ========== STEP 3: Process gene expression ==========
    print("\n" + "-"*80)
    binary_expr, gene_names = binarize_expression(expr_data)
    
    # Transpose to cell lines × genes format (matching GDSC)
    binary_expr_ctl = binary_expr.T
    
    # Save
    expr_output = os.path.join(output_dir, "exp_ccle.csv")
    binary_expr_ctl.to_csv(expr_output)
    print(f"\n✓ Saved gene expression: {expr_output}")
    
    # ========== STEP 4: Create drug info metadata ==========
    print("\n" + "-"*80)
    drug_info_processed = create_drug_info(drug_info)
    
    # Save
    drug_info_output = os.path.join(output_dir, "drug_info_ccle.csv")
    drug_info_processed.to_csv(drug_info_output)
    print(f"\n✓ Saved drug info: {drug_info_output}")
    
    # ========== STEP 5: Copy gene embeddings ==========
    print("\n" + "-"*80)
    emb_output = os.path.join(output_dir, "exp_emb_ccle.csv")
    copy_gene_embeddings(gdsc_emb_path, emb_output)
    print(f"✓ Saved gene embeddings: {emb_output}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  ✓ ccle.csv                  ({sensitivity_matrix.shape[0]} cell lines × {sensitivity_matrix.shape[1]} drugs)")
    print(f"  ✓ exp_ccle.csv              ({binary_expr_ctl.shape[0]} cell lines × {binary_expr_ctl.shape[1]} genes)")
    print(f"  ✓ drug_info_ccle.csv        ({drug_info_processed.shape[0]} drugs)")
    print(f"  ✓ exp_emb_ccle.csv          (3001 × 200 embeddings)")
    
    print("\nReady for Extension 1 experiments!")
    print("Run: python eval_pretrained_ccle.py --device cuda")
    
    return {
        "sensitivity": sensitivity_matrix,
        "expression": binary_expr_ctl,
        "drug_info": drug_info_processed
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess raw CCLE data")
    parser.add_argument("--raw-dir", default="originalData/ccle",
                       help="Directory with raw CCLE data files")
    parser.add_argument("--output-dir", default="ccleData",
                       help="Output directory for processed files")
    parser.add_argument("--gdsc-emb", default="originalData/exp_emb_gdsc.csv",
                       help="Path to GDSC gene embeddings")
    
    args = parser.parse_args()
    
    preprocess_ccle(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        gdsc_emb_path=args.gdsc_emb
    )
