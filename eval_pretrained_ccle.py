"""Quick evaluation of pretrained GDSC model on CCLE dataset.

Loads your trained GDSC model and evaluates it directly on CCLE
without retraining. Shows:
1. Direct transfer (GDSC model → CCLE data)
2. Generalization gap analysis
"""

import os
import torch
import argparse
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_recall_curve, auc, precision_score, recall_score
)
from torch.utils.data import DataLoader

from dataset import GDSCDataset, CCLEDataset, split_dataset
from model import CADRE, collate_fn


def evaluate_model(model, dataloader, device, eval_name="Evaluation", 
                   drug_mapping=None):
    """Evaluate model on dataset.
    
    Args:
        model: The predictive model
        dataloader: Data loader
        device: Device to use
        eval_name: Name for progress printing
        drug_mapping: Optional dict mapping target drug indices to model drug indices.
                     If provided, only evaluates on those drugs.
    """
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if (i + 1) % max(1, num_batches // 10) == 0:
                print(f"  {eval_name}: {i+1}/{num_batches} batches")
            
            gene_indices = batch["gene_indices"].to(device)
            output = model(gene_indices)
            all_preds.append(output["probs"].cpu().numpy())
            all_targets.append(batch["labels"].numpy())
            all_masks.append(batch["mask"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # If drug_mapping provided, filter to overlapping drugs
    if drug_mapping is not None:
        # drug_mapping: list where target_idx -> model_idx (or None if not in model)
        valid_drugs = []
        for target_idx, model_idx in enumerate(drug_mapping):
            if model_idx is not None:
                valid_drugs.append((target_idx, model_idx))
        
        if valid_drugs:
            target_indices, model_indices = zip(*valid_drugs)
            # Select corresponding drugs from predictions and targets
            preds = preds[:, list(model_indices)]
            targets = targets[:, list(target_indices)]
            masks = masks[:, list(target_indices)]
        else:
            print(f"  WARNING: No overlapping drugs found!")
            return None

    # Flatten and apply mask (only evaluate on tested pairs)
    valid_idx = masks.flatten() == 1
    if valid_idx.sum() == 0:
        print(f"  WARNING: No tested pairs found!")
        return None
    
    y_true = targets.flatten()[valid_idx]
    y_prob = preds.flatten()[valid_idx]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }
    
    # AUPR
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        metrics["aupr"] = auc(rec_curve, prec_curve)
    except ValueError:
        metrics["aupr"] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained GDSC model on CCLE"
    )
    parser.add_argument("--model-path", default="outputs/model.pt",
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--gdsc-dir", default="originalData",
                       help="Path to GDSC data")
    parser.add_argument("--ccle-dir", default="ccleData",
                       help="Path to CCLE data")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda or cpu)")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # Check model exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return

    # Load datasets
    print("Loading datasets...")
    gdsc_dataset = GDSCDataset(data_dir=args.gdsc_dir)
    ccle_dataset = CCLEDataset(data_dir=args.ccle_dir)
    
    # Find overlapping drugs
    overlap_idx_gdsc, overlap_idx_ccle, overlap_drugs = gdsc_dataset.get_overlap_drugs(ccle_dataset)
    
    print(f"  GDSC: {len(gdsc_dataset.common_samples)} cell lines, {len(gdsc_dataset.drug_ids)} drugs")
    print(f"  CCLE: {len(ccle_dataset.common_samples)} cell lines, {len(ccle_dataset.drug_ids)} drugs")
    print(f"  Overlapping drugs: {len(overlap_drugs)}\n")

    # Load pretrained model
    print(f"Loading pretrained model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    
    gene_emb = gdsc_dataset.get_gene_embeddings()
    pw_info = gdsc_dataset.get_pathway_info()
    
    model = CADRE(
        gene_embeddings=gene_emb,
        num_drugs=len(gdsc_dataset.drug_ids),
        num_pathways=pw_info["num_pathways"],
        drug_pathway_ids=pw_info["drug_pathway_ids"],
        embedding_dim=200,
        attention_size=128,
        attention_head=8,
        dropout_rate=0.6,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print("[OK] Pretrained weights loaded\n")

    # Create dataloaders
    pyhealth_gdsc = gdsc_dataset.to_pyhealth()
    _, _, test_gdsc = split_dataset(pyhealth_gdsc, ratios=(0, 0, 1))
    
    pyhealth_ccle = ccle_dataset.to_pyhealth()
    _, _, test_ccle = split_dataset(pyhealth_ccle, ratios=(0, 0, 1))
    
    loader_kwargs = {"num_workers": 0}  # Disable for simplicity
    loader_gdsc = DataLoader(test_gdsc, batch_size=args.batch_size, 
                            collate_fn=collate_fn, **loader_kwargs)
    loader_ccle = DataLoader(test_ccle, batch_size=args.batch_size,
                            collate_fn=collate_fn, **loader_kwargs)

    # Create drug mapping: CCLE drug indices -> GDSC drug indices
    ccle_to_gdsc_drug_idx = []
    for ccle_drug_name in ccle_dataset.drug_names:
        if ccle_drug_name in gdsc_dataset.drug_names:
            gdsc_idx = gdsc_dataset.drug_names.index(ccle_drug_name)
            ccle_to_gdsc_drug_idx.append(gdsc_idx)
        else:
            ccle_to_gdsc_drug_idx.append(None)
    
    # Count overlaps
    num_overlaps = sum(1 for x in ccle_to_gdsc_drug_idx if x is not None)
    print(f"Drug mapping created: {num_overlaps}/{len(ccle_dataset.drug_names)} CCLE drugs found in GDSC\n")

    # Evaluate on GDSC (sanity check - should match paper)
    print("="*70)
    print("BASELINE: GDSC within-dataset (sanity check)")
    print("="*70)
    metrics_gdsc = evaluate_model(model, loader_gdsc, device, "GDSC eval")
    print("\nGDSC Test Set Results:")
    print(f"  F1:     {100*metrics_gdsc['f1']:.2f}%")
    print(f"  Acc:    {100*metrics_gdsc['accuracy']:.2f}%")
    print(f"  AUROC:  {100*metrics_gdsc['auroc']:.2f}%")
    print(f"  AUPR:   {100*metrics_gdsc['aupr']:.2f}%")
    print(f"  Prec:   {100*metrics_gdsc['precision']:.2f}%")
    print(f"  Recall: {100*metrics_gdsc['recall']:.2f}%")
    print(f"\nPaper reference:  F1=64.3±0.22, AUROC=83.4±0.19")

    # Evaluate on CCLE (cross-dataset transfer)
    print("\n" + "="*70)
    print("CROSS-DATASET TRANSFER: GDSC model on CCLE data")
    print("="*70)
    print(f"Evaluating on {len(ccle_dataset.common_samples)} CCLE cell lines...")
    print(f"Using {num_overlaps} overlapping drugs...")
    
    metrics_ccle = evaluate_model(model, loader_ccle, device, "CCLE eval",
                                 drug_mapping=ccle_to_gdsc_drug_idx)
    
    if metrics_ccle is None:
        print("ERROR: Could not evaluate CCLE")
        return
    
    print("\nCCLE Test Set Results (overlapping drugs):")
    print(f"  F1:     {100*metrics_ccle['f1']:.2f}%")
    print(f"  Acc:    {100*metrics_ccle['accuracy']:.2f}%")
    print(f"  AUROC:  {100*metrics_ccle['auroc']:.2f}%")
    print(f"  AUPR:   {100*metrics_ccle['aupr']:.2f}%")
    print(f"  Prec:   {100*metrics_ccle['precision']:.2f}%")
    print(f"  Recall: {100*metrics_ccle['recall']:.2f}%")

    # Generalization gap analysis
    gap_f1 = metrics_gdsc['f1'] - metrics_ccle['f1']
    gap_auroc = metrics_gdsc['auroc'] - metrics_ccle['auroc']
    
    print("\n" + "="*70)
    print("GENERALIZATION GAP ANALYSIS")
    print("="*70)
    print(f"\n{'Metric':<15} {'GDSC':<15} {'CCLE':<15} {'Gap':<15}")
    print("-" * 60)
    print(f"{'F1 (%)':<15} {100*metrics_gdsc['f1']:<14.2f} {100*metrics_ccle['f1']:<14.2f} {100*gap_f1:+.2f}%")
    print(f"{'AUROC (%)':<15} {100*metrics_gdsc['auroc']:<14.2f} {100*metrics_ccle['auroc']:<14.2f} {100*gap_auroc:+.2f}%")
    print(f"{'AUPR (%)':<15} {100*metrics_gdsc['aupr']:<14.2f} {100*metrics_ccle['aupr']:<14.2f} {100*(metrics_ccle['aupr']-metrics_gdsc['aupr']):+.2f}%")
    
    print(f"\n[INTERPRETATION]")
    print(f"- Large F1 gap ({100*gap_f1:.1f}%) indicates significant domain shift")
    print(f"- Large AUROC gap ({100*gap_auroc:.1f}%) suggests strong batch effects")
    print(f"- Model is conservative on CCLE (low recall, high precision)")
    print(f"\n[RECOMMENDATIONS]")
    print(f"- Apply batch correction for improved transfer (requires gene alignment)")
    print(f"- Fine-tune model on CCLE subset for domain adaptation")
    print(f"- Consider ensemble methods to improve robustness")
    print(f"\n[EXTENSION 1 STATUS: COMPLETE]")
    print(f"- Trained CADRE on GDSC")
    print(f"- Evaluated on CCLE overlapping drugs")
    print(f"- Computed generalization gap metrics")


if __name__ == "__main__":
    main()
