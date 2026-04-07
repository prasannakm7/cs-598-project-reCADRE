"""Training script for reCADRE.

Reproduces the CADRE training procedure (Tao et al. 2020) using
PyHealth's SampleBaseDataset and PyTorch DataLoader.

Key components matching the paper:
    - OneCycle LR policy (Section 3.5)
    - Missing value imputation via mode filling (Section 4.2)
    - SGD with momentum and weight decay
    - Masked BCEWithLogitsLoss
    - 60/20/20 train/val/test split
    - Evaluation: F1, Accuracy, AUPR, AUROC
"""

import os
import time
import pickle
import random
import argparse

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import GDSCDataset, split_dataset
from model import CADRE, collate_fn
from model_dot_attn import CADREDotAttn


# ---------------------------------------------------------------------------
# OneCycle LR/Momentum scheduler (Section 3.5)
# ---------------------------------------------------------------------------
class OneCycle:
    """1-Cycle policy for learning rate and momentum scheduling.

    Phase 1 (warm-up, 45%):  LR η/10 → η,  momentum 0.95 → 0.85
    Phase 2 (cool-down, 45%): LR η → η/10, momentum 0.85 → 0.95
    Phase 3 (annihilation, 10%): LR η/10 → η/100, momentum 0.95
    """

    def __init__(self, total_steps, max_lr, div=10, prcnt=10,
                 momentum_vals=(0.95, 0.85)):
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.div = div
        self.step_len = int(total_steps * (1 - prcnt / 100) / 2)
        self.high_mom = momentum_vals[0]
        self.low_mom = momentum_vals[1]
        self.iteration = 0

    def step(self):
        """Returns (lr, momentum) for current step, then advances."""
        self.iteration += 1
        lr = self._calc_lr()
        mom = self._calc_mom()
        return lr, mom

    def _calc_lr(self):
        it = self.iteration
        if it > 2 * self.step_len:  # annihilation phase
            ratio = (it - 2 * self.step_len) / (self.total_steps - 2 * self.step_len)
            return self.max_lr / self.div * (1 - ratio * (1 - 1 / self.div))
        elif it > self.step_len:  # cool-down phase
            ratio = 1 - (it - self.step_len) / self.step_len
            return self.max_lr * (1 + ratio * (self.div - 1)) / self.div
        else:  # warm-up phase
            ratio = it / self.step_len
            return self.max_lr * (1 + ratio * (self.div - 1)) / self.div

    def _calc_mom(self):
        it = self.iteration
        if it > 2 * self.step_len:  # annihilation
            return self.high_mom
        elif it > self.step_len:  # cool-down
            ratio = (it - self.step_len) / self.step_len
            return self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:  # warm-up
            ratio = it / self.step_len
            return self.high_mom - ratio * (self.high_mom - self.low_mom)


# ---------------------------------------------------------------------------
# Missing value imputation (Section 4.2)
# ---------------------------------------------------------------------------
def fill_mask_training(train_ds):
    """Fill missing drug labels with per-drug mode in training set.

    Paper Section 4.2: 'if the sensitivity of a cell line to a drug was
    missing, we filled the missing value with the mode of the available
    sensitivities to this specific drug.'

    Modifies samples in-place: sets mask to all 1s and fills labels.
    """
    samples = train_ds.samples
    num_drugs = len(samples[0]["labels"])
    num_samples = len(samples)

    # Collect labels and masks into arrays for vectorized computation
    labels = np.array([s["labels"] for s in samples], dtype=np.float32)
    masks = np.array([s["mask"] for s in samples], dtype=np.float32)

    for d in range(num_drugs):
        tested = masks[:, d] == 1
        if tested.sum() == 0:
            continue
        # Mode = 1 if more positives than negatives, else 0
        pos_count = labels[tested, d].sum()
        neg_count = tested.sum() - pos_count
        fill_val = 1 if pos_count > neg_count else 0

        # Fill untested entries
        untested = masks[:, d] == 0
        labels[untested, d] = fill_val

    # Write back
    for i in range(num_samples):
        samples[i]["labels"] = labels[i].astype(int).tolist()
        samples[i]["mask"] = [1] * num_drugs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, dataloader, device):
    """Evaluate model on a dataset, returns metrics dict."""
    model.eval()
    all_labels, all_probs, all_masks = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            gene_indices = batch["gene_indices"].to(device)
            result = model(gene_indices)
            all_probs.append(result["probs"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            all_masks.append(batch["mask"].numpy())

    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Flatten and apply mask
    flat_labels = labels.flatten()
    flat_probs = probs.flatten()
    flat_masks = masks.flatten()

    idx = flat_masks == 1
    y_true = flat_labels[idx]
    y_prob = flat_probs[idx]
    y_pred = (y_prob >= 0.5).astype(float)

    eps = 1e-5

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5

    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        aupr = auc(rec_curve, prec_curve)
    except ValueError:
        aupr = 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
        "aupr": aupr,
        "labels": labels,
        "probs": probs,
        "masks": masks,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args):
    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    elif args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS needs batch_size=1 for this model due to large attention tensors
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Data ---
    print("Loading dataset...")
    ds = GDSCDataset(data_dir=args.data_dir)
    ds.summary()

    pyhealth_ds = ds.to_dataset()
    train_ds, val_ds, test_ds = split_dataset(pyhealth_ds, seed=args.seed)
    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Apply missing value imputation on training set (Section 4.2)
    fill_mask_training(train_ds)
    print("Applied fill_mask to training set")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # --- Model ---
    gene_emb = ds.get_gene_embeddings()
    pw_info = ds.get_pathway_info()

    if getattr(args, "dot_product_attn", False):
        model = CADREDotAttn(
            gene_embeddings=gene_emb,
            num_drugs=len(ds.drug_ids),
            embedding_dim=args.embedding_dim,
            num_heads=args.attention_head,
            d_k=getattr(args, "d_k", 64),
            dropout_rate=args.dropout_rate,
        ).to(device)
    else:
        model = CADRE(
            gene_embeddings=gene_emb,
            num_drugs=len(ds.drug_ids),
            num_pathways=pw_info["num_pathways"],
            drug_pathway_ids=pw_info["drug_pathway_ids"],
            embedding_dim=args.embedding_dim,
            attention_size=args.attention_size,
            attention_head=args.attention_head,
            dropout_rate=args.dropout_rate,
            use_attention=args.use_attention,
            use_cntx_attn=args.use_cntx_attn,
        ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # --- Optimizer ---
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.95,
        weight_decay=args.weight_decay,
    )

    # --- OneCycle scheduler ---
    # Paper: batch size = 8 cell lines, training steps = 48k
    # 48k steps at batch_size=8 means 48000/8 = 6000 optimizer steps
    # But the paper counts steps as num_samples processed, so:
    # total_optimizer_steps = max_iter / batch_size
    steps_per_epoch = len(train_loader)
    total_optimizer_steps = args.max_iter // args.batch_size
    total_epochs = (total_optimizer_steps + steps_per_epoch - 1) // steps_per_epoch
    scheduler = OneCycle(total_optimizer_steps, args.learning_rate)

    print(f"Training for {total_optimizer_steps} steps ({total_epochs} epochs)")
    print(f"Steps per epoch: {steps_per_epoch}")
    print()

    # --- Training ---
    logs = {
        "args": vars(args),
        "epoch": [],
        "step": [],
        "train_loss": [],
        "train_f1": [],
        "train_acc": [],
        "train_auroc": [],
        "train_aupr": [],
        "val_f1": [],
        "val_acc": [],
        "val_auroc": [],
        "val_aupr": [],
    }

    global_step = 0
    best_val_f1 = 0.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(total_epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            if global_step >= total_optimizer_steps:
                break

            gene_indices = batch["gene_indices"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            # OneCycle LR/momentum update
            lr, mom = scheduler.step()
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                pg["momentum"] = mom

            optimizer.zero_grad()
            result = model(gene_indices, labels=labels, mask=mask)
            loss = result["loss"]
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

        if global_step >= total_optimizer_steps:
            pass  # will evaluate below and break

        # --- Evaluate periodically ---
        if (epoch + 1) % args.eval_every == 0 or global_step >= total_optimizer_steps:
            train_metrics = evaluate(model, train_loader, device)
            val_metrics = evaluate(model, val_loader, device)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            elapsed = time.time() - start_time

            print(
                f"[Epoch {epoch+1:3d} | Step {global_step:5d} | {elapsed:.0f}s] "
                f"loss={avg_loss:.4f} | "
                f"trn F1={100*train_metrics['f1']:.1f} AUC={100*train_metrics['auroc']:.1f} | "
                f"val F1={100*val_metrics['f1']:.1f} AUC={100*val_metrics['auroc']:.1f} "
                f"AUPR={100*val_metrics['aupr']:.1f} Acc={100*val_metrics['accuracy']:.1f}"
            )

            logs["epoch"].append(epoch + 1)
            logs["step"].append(global_step)
            logs["train_loss"].append(avg_loss)
            logs["train_f1"].append(train_metrics["f1"])
            logs["train_acc"].append(train_metrics["accuracy"])
            logs["train_auroc"].append(train_metrics["auroc"])
            logs["train_aupr"].append(train_metrics["aupr"])
            logs["val_f1"].append(val_metrics["f1"])
            logs["val_acc"].append(val_metrics["accuracy"])
            logs["val_auroc"].append(val_metrics["auroc"])
            logs["val_aupr"].append(val_metrics["aupr"])

            # Save best model by val F1
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

        if global_step >= total_optimizer_steps:
            break

    # --- Final evaluation on test set using best model ---
    print("\n=== Final Evaluation (best val F1 model) ===")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)

    test_metrics = evaluate(model, test_loader, device)
    train_metrics_final = evaluate(model, train_loader, device)
    val_metrics_final = evaluate(model, val_loader, device)

    print(f"Train: F1={100*train_metrics_final['f1']:.1f}  "
          f"Acc={100*train_metrics_final['accuracy']:.1f}  "
          f"AUROC={100*train_metrics_final['auroc']:.1f}  "
          f"AUPR={100*train_metrics_final['aupr']:.1f}")
    print(f"Val:   F1={100*val_metrics_final['f1']:.1f}  "
          f"Acc={100*val_metrics_final['accuracy']:.1f}  "
          f"AUROC={100*val_metrics_final['auroc']:.1f}  "
          f"AUPR={100*val_metrics_final['aupr']:.1f}")
    print(f"Test:  F1={100*test_metrics['f1']:.1f}  "
          f"Acc={100*test_metrics['accuracy']:.1f}  "
          f"AUROC={100*test_metrics['auroc']:.1f}  "
          f"AUPR={100*test_metrics['aupr']:.1f}")

    # --- Save outputs ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Save logs
    logs["test_f1"] = test_metrics["f1"]
    logs["test_acc"] = test_metrics["accuracy"]
    logs["test_auroc"] = test_metrics["auroc"]
    logs["test_aupr"] = test_metrics["aupr"]
    logs["test_precision"] = test_metrics["precision"]
    logs["test_recall"] = test_metrics["recall"]
    logs["test_probs"] = test_metrics["probs"]
    logs["test_labels"] = test_metrics["labels"]
    logs["test_masks"] = test_metrics["masks"]
    logs["train_time_seconds"] = time.time() - start_time

    logs_path = os.path.join(args.output_dir, "logs.pkl")
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f, protocol=2)
    print(f"\nLogs saved to {logs_path}")

    # Save model checkpoint
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "pathway_info": pw_info,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # Save summary as text
    summary_path = os.path.join(args.output_dir, "results.txt")
    with open(summary_path, "w") as f:
        f.write("reCADRE Training Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Hyperparameters:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nDataset:\n")
        f.write(f"  Cell lines: 846\n")
        f.write(f"  Drugs: 260\n")
        f.write(f"  Genes: 3000 (1500 active)\n")
        f.write(f"  Pathways: {pw_info['num_pathways']}\n")
        f.write(f"  Split: 60/20/20\n")
        f.write(f"\nResults (Test Set):\n")
        f.write(f"  F1 Score:  {100*test_metrics['f1']:.2f}\n")
        f.write(f"  Accuracy:  {100*test_metrics['accuracy']:.2f}\n")
        f.write(f"  AUROC:     {100*test_metrics['auroc']:.2f}\n")
        f.write(f"  AUPR:      {100*test_metrics['aupr']:.2f}\n")
        f.write(f"  Precision: {100*test_metrics['precision']:.2f}\n")
        f.write(f"  Recall:    {100*test_metrics['recall']:.2f}\n")
        f.write(f"\nResults (Validation Set):\n")
        f.write(f"  F1 Score:  {100*val_metrics_final['f1']:.2f}\n")
        f.write(f"  Accuracy:  {100*val_metrics_final['accuracy']:.2f}\n")
        f.write(f"  AUROC:     {100*val_metrics_final['auroc']:.2f}\n")
        f.write(f"  AUPR:      {100*val_metrics_final['aupr']:.2f}\n")
        f.write(f"\nPaper Reference (CADRE on GDSC, Table 1):\n")
        f.write(f"  F1 Score:  64.3 ± 0.22\n")
        f.write(f"  Accuracy:  78.6 ± 0.34\n")
        f.write(f"  AUROC:     83.4 ± 0.19\n")
        f.write(f"  AUPR:      70.6 ± 1.30\n")
        f.write(f"\nTraining time: {logs['train_time_seconds']:.1f}s\n")
    print(f"Summary saved to {summary_path}")

    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train reCADRE model")

    # Resolve defaults relative to this script's directory
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    # Data
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(_script_dir, "originalData"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_script_dir, "outputs"))

    # Model architecture (Table A2)
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--attention_size", type=int, default=128)
    parser.add_argument("--attention_head", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.6)
    parser.add_argument("--use_attention", action="store_true", default=True)
    parser.add_argument("--no_attention", action="store_true", default=False)
    parser.add_argument("--use_cntx_attn", action="store_true", default=True)
    parser.add_argument("--no_cntx_attn", action="store_true", default=False)
    parser.add_argument("--dot_product_attn", action="store_true", default=False,
                        help="Extension 2: use scaled dot-product attention instead of CADRE's additive attention")
    parser.add_argument("--d_k", type=int, default=64,
                        help="Key/query dim per head for dot-product attention")

    # Training (Table A2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_iter", type=int, default=48000,
                        help="Total training steps (paper: 48k for GDSC)")
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)

    # Misc
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluate every N epochs")
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--use_cuda", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force CPU even if GPU/MPS available")

    args = parser.parse_args()

    # Handle negation flags
    if args.no_attention:
        args.use_attention = False
    if args.no_cntx_attn:
        args.use_cntx_attn = False

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
