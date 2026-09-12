#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run Stage 1 enhanced-position training repeatedly and summarize test metrics."""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd


METRIC_COLUMNS = [
    "best_valid_metric", "best_epoch",
    "loss", "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "mcc", "roc_auc", "average_precision", "tn", "fp", "fn", "tp",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_script", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_prefix", default="translation_strict_lenmatched")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_name_prefix", default="rnafm_self_attention_enhanced_position_strict")
    parser.add_argument("--start_run", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--branch_hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_mult", type=int, default=2)
    parser.add_argument("--position_mode", choices=["none", "enhanced"], default="enhanced")
    parser.add_argument(
        "--monitor_metric",
        choices=["roc_auc", "average_precision"],
        default="roc_auc",
    )
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for run_index in range(args.start_run, args.start_run + args.n_runs):
        dataset_pkl = os.path.join(
            args.dataset_dir, f"{args.dataset_prefix}_run{run_index:02d}.pkl"
        )
        if not os.path.exists(dataset_pkl):
            raise FileNotFoundError(dataset_pkl)

        run_name = f"{args.run_name_prefix}_run{run_index:02d}"
        run_out_dir = os.path.join(args.out_dir, run_name)
        os.makedirs(run_out_dir, exist_ok=True)
        metrics_json = os.path.join(run_out_dir, f"{run_name}_test_metrics.json")

        if not (args.skip_existing and os.path.exists(metrics_json)):
            command = [
                args.python, args.train_script,
                "--dataset_pkl", dataset_pkl,
                "--out_dir", run_out_dir,
                "--run_name", run_name,
                "--seed", str(args.seed + run_index),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--patience", str(args.patience),
                "--num_workers", str(args.num_workers),
                "--dropout", str(args.dropout),
                "--branch_hidden_dim", str(args.branch_hidden_dim),
                "--fusion_hidden_dim", str(args.fusion_hidden_dim),
                "--num_heads", str(args.num_heads),
                "--num_layers", str(args.num_layers),
                "--ff_mult", str(args.ff_mult),
                "--position_mode", args.position_mode,
                "--monitor_metric", args.monitor_metric,
                "--valid_size", str(args.valid_size),
                "--test_size", str(args.test_size),
                "--grad_clip", str(args.grad_clip),
            ]
            if args.no_preload:
                command.append("--no_preload")
            print("\n" + "=" * 100)
            print("Running:", run_name)
            print(" ".join(command))
            print("=" * 100)
            subprocess.run(command, check=True)
        else:
            print(f"Skip existing: {run_name}")

        with open(metrics_json, "r", encoding="utf-8") as handle:
            run_metrics = json.load(handle)
        config_json = os.path.join(run_out_dir, f"{run_name}_config.json")
        with open(config_json, "r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        row = {
            "run": run_index,
            "seed": args.seed + run_index,
            "position_mode": args.position_mode,
            "monitor_metric": run_config["monitor_metric"],
            "best_valid_metric": run_config["best_valid_metric"],
            "best_epoch": run_config["best_epoch"],
            "dataset_pkl": dataset_pkl,
            "run_out_dir": run_out_dir,
        }
        row.update(run_metrics)
        rows.append(row)

    all_runs = pd.DataFrame(rows)
    all_path = os.path.join(args.out_dir, "all_runs_test_metrics.csv")
    all_runs.to_csv(all_path, index=False)

    summary_rows = []
    for column in METRIC_COLUMNS:
        if column in all_runs.columns:
            values = pd.to_numeric(all_runs[column], errors="coerce")
            summary_rows.append(
                {
                    "metric": column,
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "mean±std": f"{values.mean():.4f} ± {values.std(ddof=1):.4f}",
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "summary_mean_std.csv")
    summary.to_csv(summary_path, index=False)

    print("\nSaved:", all_path)
    print("Saved:", summary_path)
    print("\nMean ± std:")
    print(summary[["metric", "mean±std"]].to_string(index=False))


if __name__ == "__main__":
    main()
