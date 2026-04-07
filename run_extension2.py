"""Extension 2: Train CADRE and DotProduct attention side-by-side and compare.

Runs train.py twice (same hyperparameters, same seed) and prints a comparison table.

Usage:
    python run_extension2.py                        # full 48k-step run
    python run_extension2.py --max_iter 800         # quick smoke test
    python run_extension2.py --cpu                  # force CPU
"""

import argparse
import copy
import os

from train import train, parse_args


def run_extension2(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- CADRE (additive contextual attention) ---
    args_cadre = copy.deepcopy(args)
    args_cadre.dot_product_attn = False
    args_cadre.output_dir = os.path.join(args.output_dir, "cadre")
    print("\n" + "=" * 60)
    print("Training CADRE (additive attention)")
    print("=" * 60)
    cadre_metrics = train(args_cadre)

    # --- DotProduct attention ---
    args_dot = copy.deepcopy(args)
    args_dot.dot_product_attn = True
    args_dot.output_dir = os.path.join(args.output_dir, "dot_product")
    print("\n" + "=" * 60)
    print("Training CADREDotAttn (scaled dot-product attention)")
    print("=" * 60)
    dot_metrics = train(args_dot)

    # --- Comparison table ---
    print("\n" + "=" * 60)
    print("Extension 2 Results — Test Set Comparison")
    print("=" * 60)
    print(f"{'Model':<22} {'F1':>7} {'AUROC':>7} {'AUPR':>7} {'Acc':>7}")
    print("-" * 50)
    for name, m in [("CADRE (additive)", cadre_metrics), ("DotProduct", dot_metrics)]:
        print(f"{name:<22} {100*m['f1']:>7.2f} {100*m['auroc']:>7.2f} "
              f"{100*m['aupr']:>7.2f} {100*m['accuracy']:>7.2f}")
    print("-" * 50)
    print("Paper ref (CADRE):     64.30   83.40   70.60   78.60")


if __name__ == "__main__":
    # Inherit all args from train.py's parser, just override output_dir default
    args = parse_args()
    _dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir == os.path.join(_dir, "outputs"):
        args.output_dir = os.path.join(_dir, "outputs", "extension2")
    run_extension2(args)
