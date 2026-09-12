#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build 10 strictly length-matched RNA-FM datasets.

Default matching strategy:
    match_mode = strict_pair

For each combined length bin:
    n = min(n_pos_bin, n_neg_bin)
    keep n positive samples and n negative samples.

This means both positive and negative samples can be removed.
The benefit is that ORF/upstream/downstream length distributions are tightly controlled.

Positive label:
    label_raw contains Ribo+ or Ribo++

Negative label:
    label_raw == ribotricer
"""

import argparse
import os
import random
import numpy as np
import pandas as pd


POS_REGEX = r"MS\+"
NEG_LABEL = "ribotricer"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def clean_seq(x):
    if pd.isna(x):
        return ""
    return str(x).upper().replace("U", "T")


def first_path(base_dir, sample_id, patterns):
    for pat in patterns:
        p = os.path.join(base_dir, pat.format(sample_id=sample_id))
        if os.path.exists(p):
            return p
    return os.path.join(base_dir, patterns[0].format(sample_id=sample_id))


def add_length_bins(df, orf_bin_size=30, flank_bin_size=50):
    # ORF contains stop codon: 30-150 aa => 93-453 nt.
    # Use 481 as the upper boundary so 451-453 nt can fall into (450,480].
    orf_bins = np.arange(90, 481, orf_bin_size)
    flank_bins = np.arange(0, 501, flank_bin_size)

    df["orf_bin"] = pd.cut(df["orf_len"], bins=orf_bins, include_lowest=True)
    df["up_bin"] = pd.cut(df["up_len"], bins=flank_bins, include_lowest=True)
    df["down_bin"] = pd.cut(df["down_len"], bins=flank_bins, include_lowest=True)

    df = df.dropna(subset=["orf_bin", "up_bin", "down_bin"]).copy()

    df["len_bin"] = (
        df["orf_bin"].astype(str) + "|" +
        df["up_bin"].astype(str) + "|" +
        df["down_bin"].astype(str)
    )
    return df


def print_length_stats(df, title="Dataset"):
    """
    Print length statistics of positive and negative samples to screen.
    This is called once for the filtered full dataset and once for every sampled run.
    """
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print("Shape:", df.shape)
    if "binary_label" in df.columns:
        print("\nBinary label counts:")
        print(df["binary_label"].value_counts(dropna=False).sort_index().to_string())

    if "label_raw" in df.columns:
        print("\nlabel_raw counts:")
        print(df["label_raw"].value_counts(dropna=False).to_string())

    length_cols = [c for c in ["orf_len", "up_len", "down_len"] if c in df.columns]
    if len(length_cols) > 0 and "binary_label" in df.columns:
        print("\nLength summary by binary_label:")
        summary = df.groupby("binary_label")[length_cols].agg(
            ["count", "mean", "std", "median", "min", "max"]
        )
        print(summary.round(3).to_string())

        print("\nLength quantiles by binary_label:")
        for label_value, sub in df.groupby("binary_label"):
            print(f"\n  binary_label = {label_value}")
            q = sub[length_cols].quantile([0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00])
            q.index = ["min", "q05", "q25", "median", "q75", "q95", "max"]
            print(q.round(3).to_string())

    if "len_bin" in df.columns and "binary_label" in df.columns:
        bin_tab = df.groupby(["len_bin", "binary_label"], observed=True).size().unstack(fill_value=0)
        if 0 not in bin_tab.columns:
            bin_tab[0] = 0
        if 1 not in bin_tab.columns:
            bin_tab[1] = 0
        bin_tab = bin_tab[[0, 1]]
        bin_tab.columns = ["neg", "pos"]
        bin_tab["total"] = bin_tab["neg"] + bin_tab["pos"]
        print("\nlen_bin coverage:")
        print("  number of non-empty len_bin:", len(bin_tab))
        print("  bins with both pos and neg:", int(((bin_tab["pos"] > 0) & (bin_tab["neg"] > 0)).sum()))
        print("  bins with pos only:", int(((bin_tab["pos"] > 0) & (bin_tab["neg"] == 0)).sum()))
        print("  bins with neg only:", int(((bin_tab["neg"] > 0) & (bin_tab["pos"] == 0)).sum()))

        print("\nTop 20 len_bin by sample count:")
        print(bin_tab.sort_values("total", ascending=False).head(20).to_string())


def strict_pair_sample(pos, neg, seed):
    """
    Strict pair matching:
        for every len_bin, n = min(n_pos_bin, n_neg_bin)
        sample n positives and n negatives.
    """
    pos_keep = []
    neg_keep = []
    bin_rows = []

    all_bins = sorted(set(pos["len_bin"].astype(str)) | set(neg["len_bin"].astype(str)))

    for b in all_bins:
        pos_b = pos[pos["len_bin"].astype(str) == b]
        neg_b = neg[neg["len_bin"].astype(str) == b]

        n_pos = len(pos_b)
        n_neg = len(neg_b)
        n_keep = min(n_pos, n_neg)

        bin_rows.append({
            "len_bin": b,
            "n_pos_raw": n_pos,
            "n_neg_raw": n_neg,
            "n_keep_each_class": n_keep,
            "n_pos_removed": n_pos - n_keep,
            "n_neg_removed": n_neg - n_keep,
        })

        if n_keep == 0:
            continue

        pos_keep.append(pos_b.sample(n=n_keep, random_state=seed, replace=False))
        neg_keep.append(neg_b.sample(n=n_keep, random_state=seed, replace=False))

    if len(pos_keep) == 0 or len(neg_keep) == 0:
        raise ValueError(
            "No matched samples were selected. Try larger bins, for example "
            "--orf_bin_size 60 --flank_bin_size 100."
        )

    out_pos = pd.concat(pos_keep, axis=0)
    out_neg = pd.concat(neg_keep, axis=0)
    bin_stats = pd.DataFrame(bin_rows)

    return out_pos, out_neg, bin_stats


def neg_only_sample_keep_all_pos(pos, neg, seed):
    """
    Alternative mode:
        keep all positives;
        for each len_bin, sample negatives up to the positive count;
        if some bins do not have enough negatives, total negatives may be fewer than positives.
    """
    neg_keep = []
    bin_rows = []

    for b, pos_b in pos.groupby("len_bin", observed=True):
        neg_b = neg[neg["len_bin"] == b]
        n_keep = min(len(pos_b), len(neg_b))

        bin_rows.append({
            "len_bin": str(b),
            "n_pos_raw": len(pos_b),
            "n_neg_raw": len(neg_b),
            "n_keep_each_class": n_keep,
            "n_pos_removed": 0,
            "n_neg_removed": len(neg_b) - n_keep,
        })

        if n_keep > 0:
            neg_keep.append(neg_b.sample(n=n_keep, random_state=seed, replace=False))

    if len(neg_keep) == 0:
        raise ValueError(
            "No negative samples were selected. Try larger bins, for example "
            "--orf_bin_size 60 --flank_bin_size 100."
        )

    out_pos = pos.copy()
    out_neg = pd.concat(neg_keep, axis=0)
    bin_stats = pd.DataFrame(bin_rows)

    return out_pos, out_neg, bin_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", required=True)
    p.add_argument("--rnafm_token_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="translation_strict_lenmatched")
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--sample_id_col", default="sample_id")
    p.add_argument("--label_col", default="label_raw")
    p.add_argument("--orf_seq_col", default="orf_seq")
    p.add_argument("--upstream_col", default="upstream_seq_50nt")
    p.add_argument("--downstream_col", default="downstream_seq_50nt")

    p.add_argument("--min_orf_nt", type=int, default=93)
    p.add_argument("--max_orf_nt", type=int, default=453)
    p.add_argument("--max_flank_nt", type=int, default=450)
    p.add_argument("--require_stop_codon", action="store_true", default=True)
    p.add_argument("--no_require_stop_codon", dest="require_stop_codon", action="store_false")

    p.add_argument("--orf_bin_size", type=int, default=30)
    p.add_argument("--flank_bin_size", type=int, default=50)
    p.add_argument(
        "--match_mode",
        choices=["strict_pair", "keep_all_pos"],
        default="strict_pair",
        help="strict_pair removes both positives and negatives per bin; keep_all_pos keeps all positives and samples negatives only."
    )

    p.add_argument("--rnafm_up_patterns", default="{sample_id}_upstream.pt")
    p.add_argument("--rnafm_orf_patterns", default="{sample_id}_orf.pt")
    p.add_argument("--rnafm_down_patterns", default="{sample_id}_downstream.pt")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.metadata_csv, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required_cols = [
        args.sample_id_col, args.label_col,
        args.orf_seq_col, args.upstream_col, args.downstream_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[args.sample_id_col] = df[args.sample_id_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)

    df[args.orf_seq_col] = df[args.orf_seq_col].apply(clean_seq)
    df[args.upstream_col] = df[args.upstream_col].apply(clean_seq).str[-args.max_flank_nt:]
    df[args.downstream_col] = df[args.downstream_col].apply(clean_seq).str[:args.max_flank_nt]

    df["orf_len"] = df[args.orf_seq_col].str.len()
    df["up_len"] = df[args.upstream_col].str.len()
    df["down_len"] = df[args.downstream_col].str.len()

    is_pos = df[args.label_col].str.contains(POS_REGEX, regex=True, na=False) & (df[args.label_col] != NEG_LABEL)
    is_neg = df[args.label_col].eq(NEG_LABEL)
    df = df[is_pos | is_neg].copy()
    df["binary_label"] = np.where(df[args.label_col].str.contains(POS_REGEX, regex=True, na=False), 1, 0).astype(int)

    print_length_stats(df, title="Before ORF filter")

    keep = df["orf_len"].between(args.min_orf_nt, args.max_orf_nt)
    if args.require_stop_codon:
        keep = keep & df[args.orf_seq_col].str[-3:].isin(["TAA", "TAG", "TGA"])
    df = df[keep].copy()

    df = add_length_bins(
        df,
        orf_bin_size=args.orf_bin_size,
        flank_bin_size=args.flank_bin_size,
    )

    print_length_stats(df, title="After ORF filter and length binning")

    pos = df[df["binary_label"] == 1].copy()
    neg = df[df["binary_label"] == 0].copy()

    if len(pos) == 0:
        raise ValueError("No positive samples after filtering.")
    if len(neg) == 0:
        raise ValueError("No negative samples after filtering.")

    up_dir = os.path.join(args.rnafm_token_dir, "upstream")
    orf_dir = os.path.join(args.rnafm_token_dir, "orf")
    down_dir = os.path.join(args.rnafm_token_dir, "downstream")
    up_pats = [x.strip() for x in args.rnafm_up_patterns.split(",") if x.strip()]
    orf_pats = [x.strip() for x in args.rnafm_orf_patterns.split(",") if x.strip()]
    down_pats = [x.strip() for x in args.rnafm_down_patterns.split(",") if x.strip()]

    summary_rows = []
    all_bin_stats = []

    for run_i in range(args.n_runs):
        seed = args.seed + run_i

        if args.match_mode == "strict_pair":
            out_pos, out_neg, bin_stats = strict_pair_sample(pos, neg, seed)
        else:
            out_pos, out_neg, bin_stats = neg_only_sample_keep_all_pos(pos, neg, seed)

        out_pos = out_pos.copy()
        out_neg = out_neg.copy()
        out_pos["binary_label"] = 1
        out_neg["binary_label"] = 0

        out = pd.concat([out_pos, out_neg], axis=0)
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

        out["rnafm_up_path"] = out[args.sample_id_col].apply(lambda x: first_path(up_dir, x, up_pats))
        out["rnafm_orf_path"] = out[args.sample_id_col].apply(lambda x: first_path(orf_dir, x, orf_pats))
        out["rnafm_down_path"] = out[args.sample_id_col].apply(lambda x: first_path(down_dir, x, down_pats))

        ok = (
            out["rnafm_up_path"].apply(os.path.exists) &
            out["rnafm_orf_path"].apply(os.path.exists) &
            out["rnafm_down_path"].apply(os.path.exists)
        )
        n_before_path_filter = len(out)
        out = out[ok].copy().reset_index(drop=True)

        print_length_stats(
            out,
            title=f"Run {run_i:02d} sampled dataset after RNA-FM path filter"
        )

        keep_cols = [
            args.sample_id_col, args.label_col, "binary_label",
            "rnafm_up_path", "rnafm_orf_path", "rnafm_down_path",
            "orf_len", "up_len", "down_len",
            "orf_bin", "up_bin", "down_bin", "len_bin",
        ]
        extra_cols = [
            "protein_accession", "transcript_accession", "protein_len_aa", "orf_nt_len",
            args.orf_seq_col, args.upstream_col, args.downstream_col,
        ]
        for c in extra_cols:
            if c in out.columns and c not in keep_cols:
                keep_cols.append(c)

        out_save = out[keep_cols].rename(columns={
            args.sample_id_col: "sample_id",
            args.label_col: "label_raw",
        })

        pkl_path = os.path.join(args.out_dir, f"{args.prefix}_run{run_i:02d}.pkl")
        csv_path = os.path.join(args.out_dir, f"{args.prefix}_run{run_i:02d}.csv.gz")
        out_save.to_pickle(pkl_path)
        out_save.to_csv(csv_path, index=False, compression="gzip")

        bin_stats = bin_stats.copy()
        bin_stats["run"] = run_i
        bin_stats["seed"] = seed
        all_bin_stats.append(bin_stats)

        vc = out["binary_label"].value_counts().to_dict()
        summary_rows.append({
            "run": run_i,
            "seed": seed,
            "match_mode": args.match_mode,
            "pkl": pkl_path,
            "csv": csv_path,
            "n_before_rnafm_path_filter": n_before_path_filter,
            "n_after_rnafm_path_filter": len(out),
            "n_pos_after_rnafm_path_filter": int(vc.get(1, 0)),
            "n_neg_after_rnafm_path_filter": int(vc.get(0, 0)),
            "pos_removed_before_path_filter": int(len(pos) - len(out_pos)),
            "neg_removed_before_path_filter": int(len(neg) - len(out_neg)),
            "orf_len_pos_mean": float(out.loc[out["binary_label"] == 1, "orf_len"].mean()),
            "orf_len_neg_mean": float(out.loc[out["binary_label"] == 0, "orf_len"].mean()),
            "up_len_pos_mean": float(out.loc[out["binary_label"] == 1, "up_len"].mean()),
            "up_len_neg_mean": float(out.loc[out["binary_label"] == 0, "up_len"].mean()),
            "down_len_pos_mean": float(out.loc[out["binary_label"] == 1, "down_len"].mean()),
            "down_len_neg_mean": float(out.loc[out["binary_label"] == 0, "down_len"].mean()),
        })

        print("\nSaved:")
        print(pkl_path)
        print(csv_path)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, f"{args.prefix}_dataset_summary.csv")
    summary.to_csv(summary_path, index=False)

    bin_stats_all = pd.concat(all_bin_stats, axis=0)
    bin_stats_path = os.path.join(args.out_dir, f"{args.prefix}_bin_stats.csv")
    bin_stats_all.to_csv(bin_stats_path, index=False)

    print("\nAll finished.")
    print("Dataset summary saved:", summary_path)
    print("Bin stats saved:", bin_stats_path)


if __name__ == "__main__":
    main()
