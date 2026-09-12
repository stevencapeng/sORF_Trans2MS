#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate 10 Stage2 enhanced_position checkpoints on FMJCpub external data.

The model class and data-padding helpers are imported from the exact training
script used to produce the checkpoints. Model hyperparameters are reconstructed
from each checkpoint's saved ``args`` field, which prevents training/prediction
configuration drift.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


METRICS = [
    "loss",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "average_precision",
    "tn",
    "fp",
    "fn",
    "tp",
    "n",
    "positive",
    "negative",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict FMJCpub with 10 Stage2 enhanced_position checkpoints"
        )
    )
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--train_script", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config_name", default="enhanced_position")
    parser.add_argument("--checkpoint_glob", default=None)
    parser.add_argument("--expected_models", type=int, default=10)
    parser.add_argument("--label_col", default="binary_label")
    parser.add_argument("--positive_value", default="1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rna_up_col", default=None)
    parser.add_argument("--rna_orf_col", default=None)
    parser.add_argument("--rna_down_col", default=None)
    parser.add_argument("--esm2_col", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def save_json(value, path):
    def convert(item):
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        return item

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            {key: convert(item) for key, item in value.items()},
            handle,
            ensure_ascii=False,
            indent=2,
        )


def load_table(path):
    path = Path(path)
    lower = path.name.lower()
    if lower.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    if lower.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, low_memory=False)
    if lower.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t", low_memory=False)
    raise ValueError(
        "--input_table must be PKL, CSV/CSV.GZ, or TSV/TSV.GZ: " + str(path)
    )


def load_training_module(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "stage2_enhanced_position_training_module", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "Stage2RNACNNProteinCNN",
        "TokenDataset",
        "collate_tokens",
        "choose_col",
        "RNA_UP_CANDIDATES",
        "RNA_ORF_CANDIDATES",
        "RNA_DOWN_CANDIDATES",
        "ESM2_CANDIDATES",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Training script lacks required definitions: {missing}"
        )
    return module


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def run_number(path):
    matches = re.findall(r"run(\d+)", str(path))
    if not matches:
        raise ValueError(f"Cannot extract run number from checkpoint: {path}")
    return int(matches[-1])


def find_checkpoints(args):
    if args.checkpoint_glob:
        pattern = args.checkpoint_glob
    else:
        pattern = str(
            Path(args.model_root)
            / args.config_name
            / f"cnn_{args.config_name}_run*"
            / f"cnn_{args.config_name}_run*_best_model.pt"
        )
    checkpoints = [Path(path) for path in glob.glob(pattern)]
    checkpoints = sorted(checkpoints, key=run_number)
    if len(checkpoints) != args.expected_models:
        raise FileNotFoundError(
            f"Expected {args.expected_models} checkpoints but found "
            f"{len(checkpoints)} with pattern:\n{pattern}\n"
            f"Found: {[str(path) for path in checkpoints]}"
        )
    runs = [run_number(path) for path in checkpoints]
    if len(runs) != len(set(runs)):
        raise ValueError(f"Duplicated run numbers in checkpoints: {runs}")
    return list(zip(runs, checkpoints))


def normalize_labels(series, positive_value):
    nonmissing = series.dropna()
    if nonmissing.empty:
        raise ValueError("Label column contains no usable values")
    numeric = pd.to_numeric(nonmissing, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()) <= {0.0, 1.0}:
        labels = pd.to_numeric(series, errors="raise").astype(int)
    else:
        labels = series.astype(str).eq(str(positive_value)).astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError(
            f"External labels must contain both classes; counts="
            f"{labels.value_counts().to_dict()}"
        )
    return labels


def prepare_frame(frame, args, module):
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    if "sample_id" not in frame.columns:
        raise ValueError("FMJCpub table lacks required column: sample_id")
    if args.label_col not in frame.columns:
        raise ValueError(
            f"FMJCpub table lacks label column {args.label_col!r}; "
            f"columns={frame.columns.tolist()}"
        )
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        examples = frame.loc[
            frame["sample_id"].duplicated(keep=False), "sample_id"
        ].head(10).tolist()
        raise ValueError(f"Duplicated sample_id values: {examples}")
    frame["binary_label"] = normalize_labels(
        frame[args.label_col], args.positive_value
    )
    columns = {
        "rna_up": module.choose_col(
            frame,
            args.rna_up_col,
            module.RNA_UP_CANDIDATES,
            "RNA upstream",
        ),
        "rna_orf": module.choose_col(
            frame,
            args.rna_orf_col,
            module.RNA_ORF_CANDIDATES,
            "RNA ORF",
        ),
        "rna_down": module.choose_col(
            frame,
            args.rna_down_col,
            module.RNA_DOWN_CANDIDATES,
            "RNA downstream",
        ),
        "esm2": module.choose_col(
            frame, args.esm2_col, module.ESM2_CANDIDATES, "ESM2"
        ),
    }
    null_rows = frame[list(columns.values())].isna().any(axis=1)
    if null_rows.any():
        raise ValueError(
            f"{int(null_rows.sum())} samples have missing embedding paths; "
            f"examples={frame.loc[null_rows, 'sample_id'].head(10).tolist()}"
        )
    return frame.reset_index(drop=True), columns


def build_model(module, checkpoint):
    if checkpoint.get("model_class") not in {
        None,
        "Stage2RNACNNProteinCNN",
    }:
        raise ValueError(
            f"Unexpected model_class: {checkpoint.get('model_class')}"
        )
    saved = checkpoint.get("args", {})
    required = ["rna_dim", "protein_dim", "model_state_dict"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint lacks required keys: {missing}")
    model = module.Stage2RNACNNProteinCNN(
        rna_dim=int(checkpoint["rna_dim"]),
        protein_dim=int(checkpoint["protein_dim"]),
        architecture=saved.get("architecture", "baseline"),
        conv_hidden=int(saved.get("conv_hidden", 128)),
        kernels=tuple(saved.get("kernels", [3, 5, 7])),
        conv_depth=int(saved.get("conv_depth", 1)),
        pool_mode=saved.get("pool_mode", "max"),
        position_channels=int(saved.get("position_channels", 0)),
        protein_nterm_window=int(saved.get("protein_nterm_window", 0)),
        branch_hidden=int(saved.get("branch_hidden", 256)),
        fusion_hidden=int(saved.get("fusion_hidden", 256)),
        fusion_mode=saved.get("fusion_mode", "concat"),
        dropout=float(saved.get("dropout", 0.5)),
        upstream_padding=saved.get("upstream_padding", "right"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, saved


def make_loader(frame, columns, module, upstream_padding, args):
    return DataLoader(
        module.TokenDataset(frame, columns),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(
            module.collate_tokens, upstream_padding=upstream_padding
        ),
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities = []
    for batch_index, batch in enumerate(loader, 1):
        batch = {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
        logits = model(batch).reshape(-1)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        if batch_index % 20 == 0:
            print(f"  predicted batches: {batch_index}/{len(loader)}", flush=True)
    return np.concatenate(probabilities).astype(float)


def calc_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return {
        "loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "positive": int((y_true == 1).sum()),
        "negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
    }


def base_prediction_frame(frame):
    output = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "y_true": frame["binary_label"].astype(int).to_numpy(),
        }
    )
    if "label_raw" in frame.columns:
        output.insert(1, "label_raw", frame["label_raw"].astype(str).to_numpy())
    return output


def summarize_metrics(metrics_frame):
    rows = []
    for metric in METRICS:
        values = pd.to_numeric(metrics_frame[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "mean_std": (
                    f"{values.mean():.4f} +/- "
                    f"{(values.std(ddof=1) if len(values) > 1 else 0.0):.4f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_csv = out_dir / "fmjcpub_enhanced_position_ensemble_predictions.csv"
    ensemble_metrics_json = out_dir / "fmjcpub_enhanced_position_ensemble_metrics.json"
    if args.skip_existing and ensemble_csv.is_file() and ensemble_metrics_json.is_file():
        print("Skip existing completed ensemble:", ensemble_csv)
        return

    module = load_training_module(args.train_script)
    checkpoint_items = find_checkpoints(args)
    frame, columns = prepare_frame(load_table(args.input_table), args, module)
    print("FMJCpub samples:", len(frame))
    print("Label counts:", frame["binary_label"].value_counts().sort_index().to_dict())
    print("Embedding columns:", columns)
    print("Checkpoints:", len(checkpoint_items))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    y_true = frame["binary_label"].astype(int).to_numpy()
    probability_by_run = {}
    metric_rows = []
    common_signature = None

    for run, checkpoint_path in checkpoint_items:
        run_name = f"cnn_{args.config_name}_run{run:02d}"
        run_dir = out_dir / f"run{run:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / f"{run_name}_fmjcpub_predictions.csv"
        metrics_path = run_dir / f"{run_name}_fmjcpub_metrics.json"

        print(f"\nRun {run:02d}: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        saved_config = checkpoint.get("config_name")
        if saved_config not in {None, args.config_name}:
            raise ValueError(
                f"Checkpoint config_name={saved_config!r}, expected "
                f"{args.config_name!r}: {checkpoint_path}"
            )
        model, saved_args = build_model(module, checkpoint)
        signature = (
            saved_args.get("architecture", "baseline"),
            saved_args.get("upstream_padding", "right"),
            int(saved_args.get("position_channels", 0)),
            tuple(saved_args.get("kernels", [3, 5, 7])),
        )
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError(
                f"Run {run:02d} architecture differs from earlier runs: "
                f"{signature} versus {common_signature}"
            )

        if args.skip_existing and prediction_path.is_file() and metrics_path.is_file():
            existing = pd.read_csv(prediction_path)
            required = {"sample_id", "y_true", "y_prob", "y_pred"}
            if not required.issubset(existing.columns):
                raise ValueError(
                    f"Incomplete existing prediction file: {prediction_path}"
                )
            if not np.array_equal(
                existing["sample_id"].astype(str).to_numpy(),
                frame["sample_id"].astype(str).to_numpy(),
            ):
                raise ValueError(
                    f"Existing predictions have different sample order: "
                    f"{prediction_path}"
                )
            if not np.array_equal(
                existing["y_true"].astype(int).to_numpy(), y_true
            ):
                raise ValueError(
                    f"Existing predictions have different labels: "
                    f"{prediction_path}"
                )
            y_prob = pd.to_numeric(existing["y_prob"], errors="raise").to_numpy()
            if not np.isfinite(y_prob).all():
                raise ValueError(
                    f"Existing predictions contain non-finite scores: "
                    f"{prediction_path}"
                )
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            probability_by_run[run] = y_prob
            metric_rows.append(metrics)
            print("  reused existing predictions")
            del model, checkpoint
            continue

        loader = make_loader(
            frame,
            columns,
            module,
            saved_args.get("upstream_padding", "right"),
            args,
        )
        model = model.to(device)
        y_prob = predict(model, loader, device)
        if len(y_prob) != len(frame):
            raise RuntimeError(
                f"Prediction count mismatch: {len(y_prob)} versus {len(frame)}"
            )
        probability_by_run[run] = y_prob
        predictions = base_prediction_frame(frame)
        predictions["y_prob"] = y_prob
        predictions["y_pred"] = (y_prob >= args.threshold).astype(int)
        predictions.to_csv(prediction_path, index=False)

        metrics = calc_metrics(y_true, y_prob, args.threshold)
        metrics.update(
            {
                "run": int(run),
                "model": "Stage2 RNA-FM + ESM2 CNN",
                "config_name": args.config_name,
                "checkpoint": str(checkpoint_path),
            }
        )
        save_json(metrics, metrics_path)
        metric_rows.append(metrics)
        print(
            f"  ROC-AUC={metrics['roc_auc']:.4f}, "
            f"AUPR={metrics['average_precision']:.4f}"
        )
        del model, loader, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_frame = pd.DataFrame(metric_rows).sort_values("run")
    metrics_frame.to_csv(out_dir / "fmjcpub_all_runs_metrics.csv", index=False)
    summary = summarize_metrics(metrics_frame)
    summary.to_csv(out_dir / "fmjcpub_10runs_summary_mean_std.csv", index=False)

    ensemble = base_prediction_frame(frame)
    probability_columns = []
    for run in sorted(probability_by_run):
        column = f"y_prob_model_{run:02d}"
        ensemble[column] = probability_by_run[run]
        probability_columns.append(column)
    ensemble["y_prob"] = ensemble[probability_columns].mean(axis=1)
    ensemble["y_prob_std"] = ensemble[probability_columns].std(axis=1, ddof=1)
    ensemble["y_pred"] = (ensemble["y_prob"] >= args.threshold).astype(int)
    leading = [column for column in ["sample_id", "label_raw", "y_true"] if column in ensemble]
    ensemble = ensemble[
        leading
        + ["y_prob", "y_prob_std", "y_pred"]
        + probability_columns
    ]
    ensemble.to_csv(ensemble_csv, index=False)

    ensemble_metrics = calc_metrics(
        ensemble["y_true"].to_numpy(),
        ensemble["y_prob"].to_numpy(),
        args.threshold,
    )
    ensemble_metrics.update(
        {
            "model": "Stage2 RNA-FM + ESM2 CNN probability ensemble",
            "config_name": args.config_name,
            "n_models": len(probability_columns),
        }
    )
    save_json(ensemble_metrics, ensemble_metrics_json)
    pd.DataFrame([ensemble_metrics]).to_csv(
        out_dir / "fmjcpub_enhanced_position_ensemble_metrics.csv",
        index=False,
    )
    save_json(
        {
            "input_table": str(Path(args.input_table)),
            "train_script": str(Path(args.train_script)),
            "model_root": str(Path(args.model_root)),
            "config_name": args.config_name,
            "embedding_columns": columns,
            "checkpoint_runs": {
                f"run{run:02d}": str(path) for run, path in checkpoint_items
            },
            "model_signature": {
                "architecture": common_signature[0],
                "upstream_padding": common_signature[1],
                "position_channels": common_signature[2],
                "kernels": list(common_signature[3]),
            },
        },
        out_dir / "fmjcpub_prediction_manifest.json",
    )

    print("\n10-RUN SUMMARY")
    print(summary.to_string(index=False))
    print("\nENSEMBLE METRICS")
    for key, value in ensemble_metrics.items():
        print(f"{key}: {value}")
    print("\nEnsemble predictions:", ensemble_csv)


if __name__ == "__main__":
    main()
