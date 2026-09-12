#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune or finalize Stage2 RNA-FM + ESM2 CNNs over fixed runs.

Hyperparameter configurations are ranked only by mean best validation ROC-AUC.
Test metrics are retained for transparent reporting, but never determine the
ranking. Source run tables are not re-matched, re-sampled, or filtered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
    "best_epoch",
    "best_valid_roc_auc",
    "valid_average_precision",
    "trainable_parameters",
]


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def read_id_list(path):
    frame = pd.read_csv(path, dtype=str)
    if "sample_id" not in frame.columns:
        if len(frame.columns) != 1:
            raise ValueError(f"No sample_id column in {path}")
        frame = frame.rename(columns={frame.columns[0]: "sample_id"})
    ids = frame["sample_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicated IDs in {path}")
    return ids


def validate_split(frame, split_dir, require_full_coverage=True):
    split_dir = Path(split_dir)
    train_ids = read_id_list(split_dir / "train_ids.csv")
    valid_ids = read_id_list(split_dir / "valid_ids.csv")
    test_ids = read_id_list(split_dir / "test_ids.csv")
    train_set, valid_set, test_set = (
        set(train_ids), set(valid_ids), set(test_ids)
    )
    if train_set & valid_set or train_set & test_set or valid_set & test_set:
        raise ValueError(f"Split ID overlap detected in {split_dir}")

    available = set(frame["sample_id"].astype(str))
    selected = train_set | valid_set | test_set
    unknown = selected - available
    omitted = available - selected
    if unknown:
        raise ValueError(
            f"{len(unknown)} split IDs are absent from the dataset; "
            f"examples={sorted(unknown)[:10]}"
        )
    if require_full_coverage and omitted:
        raise ValueError(
            f"{len(omitted)} samples are omitted by the split; "
            f"examples={sorted(omitted)[:10]}"
        )

    indexed = frame.set_index("sample_id")
    counts = {}
    for name, ids in [
        ("train", train_ids),
        ("valid", valid_ids),
        ("test", test_ids),
    ]:
        labels = indexed.loc[ids, "binary_label"]
        if labels.nunique() != 2:
            raise ValueError(f"{name} split in {split_dir} lacks one class")
        counts[name] = {
            "n": int(len(ids)),
            "label_counts": {
                str(key): int(value)
                for key, value in labels.value_counts().sort_index().items()
            },
        }
    return counts


def create_split(frame, split_dir, seed, valid_size, test_size):
    if valid_size <= 0 or test_size <= 0:
        raise ValueError("valid_size and test_size must both be > 0")
    if valid_size + test_size >= 1:
        raise ValueError("valid_size + test_size must be < 1")

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    holdout_size = valid_size + test_size
    train_frame, holdout_frame = train_test_split(
        frame,
        test_size=holdout_size,
        random_state=seed,
        stratify=frame["binary_label"],
    )
    valid_frame, test_frame = train_test_split(
        holdout_frame,
        test_size=test_size / holdout_size,
        random_state=seed,
        stratify=holdout_frame["binary_label"],
    )
    for name, subset in [
        ("train", train_frame),
        ("valid", valid_frame),
        ("test", test_frame),
    ]:
        subset[["sample_id"]].astype(str).to_csv(
            split_dir / f"{name}_ids.csv", index=False
        )
    counts = validate_split(frame, split_dir, require_full_coverage=True)
    save_json(
        {
            "seed": int(seed),
            "valid_size": float(valid_size),
            "test_size": float(test_size),
            **counts,
        },
        split_dir / "split_summary.json",
    )
    return counts


def resolve_or_create_split(
    frame, split_dir, seed, valid_size, test_size, external_split_root
):
    split_dir = Path(split_dir)
    expected = [
        split_dir / "train_ids.csv",
        split_dir / "valid_ids.csv",
        split_dir / "test_ids.csv",
    ]
    present = [path.is_file() for path in expected]
    if all(present):
        return validate_split(frame, split_dir, require_full_coverage=True)
    if any(present):
        raise FileNotFoundError(
            f"Incomplete split in {split_dir}; expected all three ID files"
        )
    if external_split_root:
        raise FileNotFoundError(
            f"--split_root was supplied, but split files are missing: {split_dir}"
        )
    return create_split(frame, split_dir, seed, valid_size, test_size)


def find_esm2_token(root, sample_id, subdirs, patterns):
    for subdir in subdirs:
        directory = root / subdir if subdir else root
        for pattern in patterns:
            candidate = directory / pattern.format(sample_id=sample_id)
            if candidate.is_file():
                return str(candidate)
    return None


def validate_prepared(frame, path):
    required = ["sample_id", "binary_label", "esm2_path"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["binary_label"] = pd.to_numeric(
        frame["binary_label"], errors="raise"
    ).astype(int)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicated sample IDs in {path}")
    if set(frame["binary_label"].unique()) != {0, 1}:
        raise ValueError(f"binary_label in {path} must contain both 0 and 1")
    missing_files = [
        value for value in frame["esm2_path"].astype(str) if not Path(value).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} ESM2 paths in {path} do not exist; "
            f"examples={missing_files[:5]}"
        )
    return frame


def prepare_dataset_with_esm2(
    source_pkl, target_pkl, esm2_root, subdirs, patterns
):
    source_pkl = Path(source_pkl)
    target_pkl = Path(target_pkl)
    if target_pkl.is_file():
        frame = validate_prepared(pd.read_pickle(target_pkl), target_pkl)
        print("Reuse prepared dataset:", target_pkl)
        return frame

    frame = pd.read_pickle(source_pkl)
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    required = ["sample_id", "binary_label"]
    missing_columns = [column for column in required if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing columns in {source_pkl}: {missing_columns}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["binary_label"] = pd.to_numeric(
        frame["binary_label"], errors="raise"
    ).astype(int)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicated sample IDs in {source_pkl}")
    if set(frame["binary_label"].unique()) != {0, 1}:
        raise ValueError("Every run must contain both binary classes")

    original_ids = frame["sample_id"].tolist()
    original_labels = frame["binary_label"].tolist()
    resolved = [
        find_esm2_token(esm2_root, sample_id, subdirs, patterns)
        for sample_id in original_ids
    ]
    missing_mask = pd.isna(resolved)
    if any(missing_mask):
        report = frame.loc[missing_mask, ["sample_id"]].copy()
        report_path = target_pkl.with_name(
            target_pkl.stem + "_missing_esm2.tsv"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, sep="\t", index=False)
        raise FileNotFoundError(
            f"{int(sum(missing_mask))} ESM2 files are missing; "
            f"examples={report['sample_id'].head(10).tolist()}; "
            f"report={report_path}"
        )
    frame["esm2_path"] = resolved
    if frame["sample_id"].tolist() != original_ids:
        raise RuntimeError("Dataset preparation changed sample order")
    if frame["binary_label"].tolist() != original_labels:
        raise RuntimeError("Dataset preparation changed labels")
    target_pkl.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(target_pkl)
    frame.to_csv(target_pkl.with_suffix(".csv.gz"), index=False)
    print("Saved prepared dataset:", target_pkl)
    return frame


def load_search_space(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        configs = json.load(handle)
    if not isinstance(configs, dict) or not configs:
        raise ValueError("Search-space JSON must contain a non-empty object")
    return configs


def summarize(frame, out_file):
    rows = []
    if not frame.empty:
        for config_name, group in frame.groupby("config_name", sort=False):
            for metric in METRICS:
                if metric not in group.columns:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce")
                mean = values.mean()
                std = values.std(ddof=1)
                rows.append(
                    {
                        "config_name": config_name,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "mean±std": f"{mean:.4f} ± {std:.4f}",
                        "min": values.min(),
                        "max": values.max(),
                    }
                )
    pd.DataFrame(rows).to_csv(out_file, index=False)


def build_ranking(frame, configs):
    rows = []
    for config_name, group in frame.groupby("config_name", sort=False):
        valid_values = pd.to_numeric(
            group["best_valid_roc_auc"], errors="coerce"
        )
        test_values = pd.to_numeric(group["roc_auc"], errors="coerce")
        aupr_values = pd.to_numeric(
            group["average_precision"], errors="coerce"
        )
        rows.append(
            {
                "config_name": config_name,
                "description": configs[config_name].get("description", ""),
                "n_runs": int(len(group)),
                "valid_roc_auc_mean": valid_values.mean(),
                "valid_roc_auc_std": valid_values.std(ddof=1),
                "test_roc_auc_mean_report_only": test_values.mean(),
                "test_roc_auc_std_report_only": test_values.std(ddof=1),
                "test_aupr_mean_report_only": aupr_values.mean(),
                "test_aupr_std_report_only": aupr_values.std(ddof=1),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["valid_roc_auc_mean", "valid_roc_auc_std"],
        ascending=[False, True],
    )
    ranking.insert(0, "validation_rank", np.arange(1, len(ranking) + 1))
    return ranking


def append_optional_columns(command, args):
    for flag, value in [
        ("--rna_up_col", args.rna_up_col),
        ("--rna_orf_col", args.rna_orf_col),
        ("--rna_down_col", args.rna_down_col),
        ("--esm2_col", args.esm2_col),
    ]:
        if value:
            command += [flag, value]


def config_to_command(command, config):
    scalar_flags = {
        "architecture": "--architecture",
        "conv_hidden": "--conv_hidden",
        "conv_depth": "--conv_depth",
        "pool_mode": "--pool_mode",
        "position_channels": "--position_channels",
        "protein_nterm_window": "--protein_nterm_window",
        "branch_hidden": "--branch_hidden",
        "fusion_hidden": "--fusion_hidden",
        "fusion_mode": "--fusion_mode",
        "upstream_padding": "--upstream_padding",
        "dropout": "--dropout",
        "lr": "--lr",
        "weight_decay": "--weight_decay",
        "scheduler": "--scheduler",
    }
    for key, flag in scalar_flags.items():
        if key in config:
            command += [flag, str(config[key])]
    if "kernels" in config:
        command += ["--kernels", *[str(value) for value in config["kernels"]]]


def parse_args():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--train_script",
        default=str(script_dir / "train_stage2_rnaesm_cnn_tuned_v1.py"),
    )
    parser.add_argument(
        "--search_space",
        default=str(script_dir / "cnn_search_space_v1.json"),
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument(
        "--dataset_prefix", default="translation_strict_lenmatched"
    )
    parser.add_argument("--esm2_token_dir", required=True)
    parser.add_argument(
        "--esm2_subdirs", default=",orf,protein,esm2"
    )
    parser.add_argument(
        "--esm2_patterns",
        default=(
            "{sample_id}.pt,{sample_id}_esm2.pt,"
            "{sample_id}_protein.pt,{sample_id}_orf.pt"
        ),
    )
    parser.add_argument(
        "--split_root",
        default=None,
        help="Existing run00... split root; recommended for fair comparison",
    )
    parser.add_argument("--out_root", required=True)
    parser.add_argument(
        "--prepared_dataset_dir",
        default=None,
        help="Can point to the baseline v4 prepared_datasets directory",
    )
    parser.add_argument(
        "--config_names",
        nargs="+",
        default=None,
        help="Subset of JSON config names; default is all configs",
    )
    parser.add_argument("--start_run", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rna_up_col", default=None)
    parser.add_argument("--rna_orf_col", default=None)
    parser.add_argument("--rna_down_col", default=None)
    parser.add_argument("--esm2_col", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train_script = Path(args.train_script)
    dataset_root = Path(args.dataset_dir)
    esm2_root = Path(args.esm2_token_dir)
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not esm2_root.is_dir():
        raise FileNotFoundError(esm2_root)

    configs = load_search_space(args.search_space)
    selected_names = args.config_names or list(configs)
    unknown = [name for name in selected_names if name not in configs]
    if unknown:
        raise ValueError(
            f"Unknown config names: {unknown}; available={list(configs)}"
        )
    selected_configs = {name: configs[name] for name in selected_names}

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prepared_root = (
        Path(args.prepared_dataset_dir)
        if args.prepared_dataset_dir
        else out_root / "prepared_datasets"
    )
    external_split_root = args.split_root is not None
    split_root = (
        Path(args.split_root)
        if external_split_root
        else out_root / "generated_splits"
    )
    prepared_root.mkdir(parents=True, exist_ok=True)

    subdirs = [item.strip() for item in args.esm2_subdirs.split(",")]
    patterns = [
        item.strip() for item in args.esm2_patterns.split(",") if item.strip()
    ]
    if not patterns:
        raise ValueError("--esm2_patterns produced no filename patterns")

    prepared_by_run = {}
    split_by_run = {}
    manifest_rows = []
    for run in range(args.start_run, args.start_run + args.n_runs):
        source = dataset_root / f"{args.dataset_prefix}_run{run:02d}.pkl"
        prepared = (
            prepared_root
            / f"{args.dataset_prefix}_rnafm_esm2_run{run:02d}.pkl"
        )
        split_dir = split_root / f"run{run:02d}"
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = prepare_dataset_with_esm2(
            source, prepared, esm2_root, subdirs, patterns
        )
        counts = resolve_or_create_split(
            frame,
            split_dir,
            seed=args.seed + run,
            valid_size=args.valid_size,
            test_size=args.test_size,
            external_split_root=external_split_root,
        )
        prepared_by_run[run] = prepared
        split_by_run[run] = split_dir
        label_counts = frame["binary_label"].value_counts().sort_index()
        manifest_rows.append(
            {
                "run": run,
                "seed": args.seed + run,
                "source_pkl": str(source),
                "prepared_pkl": str(prepared),
                "split_dir": str(split_dir),
                "n": int(len(frame)),
                "negative": int(label_counts.get(0, 0)),
                "positive": int(label_counts.get(1, 0)),
                "train_n": counts["train"]["n"],
                "valid_n": counts["valid"]["n"],
                "test_n": counts["test"]["n"],
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        out_root / "cnn_dataset_split_manifest.csv", index=False
    )
    save_json(selected_configs, out_root / "selected_cnn_configs.json")

    rows = []
    for config_name, config in selected_configs.items():
        config_root = out_root / config_name
        config_root.mkdir(parents=True, exist_ok=True)
        print("\n" + "#" * 100)
        print("CNN CONFIG:", config_name)
        print("DESCRIPTION:", config.get("description", ""))
        print("#" * 100)

        for run in range(args.start_run, args.start_run + args.n_runs):
            run_name = f"cnn_{config_name}_run{run:02d}"
            run_out = config_root / run_name
            run_out.mkdir(parents=True, exist_ok=True)
            metrics_file = run_out / f"{run_name}_test_metrics.json"

            if args.skip_existing and metrics_file.is_file():
                print("Skip existing:", metrics_file)
            else:
                command = [
                    args.python,
                    str(train_script),
                    "--dataset_pkl",
                    str(prepared_by_run[run]),
                    "--split_dir",
                    str(split_by_run[run]),
                    "--out_dir",
                    str(run_out),
                    "--run_name",
                    run_name,
                    "--config_name",
                    config_name,
                    "--seed",
                    str(args.seed + run),
                    "--threshold",
                    str(args.threshold),
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--patience",
                    str(args.patience),
                    "--grad_clip",
                    str(args.grad_clip),
                    "--num_workers",
                    str(args.num_workers),
                ]
                config_to_command(command, config)
                append_optional_columns(command, args)
                print("\nRunning", config_name, f"run{run:02d}")
                print(" ".join(command))
                if not args.dry_run:
                    subprocess.run(command, check=True)

            if args.dry_run:
                continue
            if not metrics_file.is_file():
                raise FileNotFoundError(
                    f"Training finished without metrics: {metrics_file}"
                )
            with metrics_file.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            row = {
                "config_name": config_name,
                "run": run,
                "seed": args.seed + run,
                "dataset_pkl": str(prepared_by_run[run]),
                "split_dir": str(split_by_run[run]),
                "run_out": str(run_out),
            }
            row.update(metrics)
            rows.append(row)

        if args.dry_run:
            continue
        config_frame = pd.DataFrame(
            [row for row in rows if row["config_name"] == config_name]
        )
        config_frame.to_csv(
            config_root / "all_runs_test_metrics.csv", index=False
        )
        summarize(config_frame, config_root / "summary_mean_std.csv")

    if args.dry_run:
        print("\nDry run completed; no CNN training was started.")
        return

    all_frame = pd.DataFrame(rows)
    if all_frame.empty:
        raise RuntimeError("No CNN metrics were generated")
    all_frame.to_csv(out_root / "all_configs_all_runs_metrics.csv", index=False)
    summarize(all_frame, out_root / "all_configs_summary_mean_std.csv")
    ranking = build_ranking(all_frame, selected_configs)
    ranking.to_csv(
        out_root / "cnn_config_ranking_by_validation_roc.csv", index=False
    )

    print("\nFinished.")
    print("Ranking criterion: mean best validation ROC-AUC")
    print(
        ranking[
            [
                "validation_rank",
                "config_name",
                "valid_roc_auc_mean",
                "valid_roc_auc_std",
                "test_roc_auc_mean_report_only",
            ]
        ].to_string(index=False)
    )
    print(
        "Ranking file:", out_root / "cnn_config_ranking_by_validation_roc.csv"
    )


if __name__ == "__main__":
    main()
