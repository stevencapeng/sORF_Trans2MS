#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare a balanced MS++ vs ribotricer dataset for ESM2 + RNA-FM cross-attention.

Input:
  - metadata table with sample_id, label_raw, orf_nt_len, etc.
  - ESM2 ORF token files saved as: <esm2_token_dir>/<sample_id>.pt
  - RNA-FM token files saved as:
      <rnafm_token_dir>/upstream/<sample_id>_upstream.pt
      <rnafm_token_dir>/orf/<sample_id>_orf.pt
      <rnafm_token_dir>/downstream/<sample_id>_downstream.pt

Output:
  - pickle/csv.gz table containing paths for ESM2 ORF and RNA-FM upstream/ORF/downstream tokens.
"""

import argparse
import os
import random
import numpy as np
import pandas as pd


POS_REGEX = r"MS\+\+"      # MS++|Ribo-, MS++|Ribo+, MS++|Ribo++ are positive
NEG_LABEL = "ribotricer"  # ribotricer rtorfs are negative


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def read_table(path: str) -> pd.DataFrame:
    if path.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    return pd.read_csv(path, low_memory=False)


def first_path(base_dir: str, sample_id: str, patterns):
    for pat in patterns:
        p = os.path.join(base_dir, pat.format(sample_id=sample_id))
        if os.path.exists(p):
            return p
    return os.path.join(base_dir, patterns[0].format(sample_id=sample_id))


def split_patterns(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", default="/disk3/hejp/00_thesis/4_predict_translation/data/processed/dataset_nt450_v3/train_pos_neg_merged.csv.gz")
    p.add_argument("--esm2_token_dir", default="/disk4hu/hejp/00_thesis/4_predict_translation/results/features/features_v3_esm2_token_pos_neg")
    p.add_argument("--rnafm_token_dir", default="/disk4hu/hejp/00_thesis/4_predict_translation/results/features/features_rnafm_token_ribotricer_openprot_nt450")
    p.add_argument("--out_pkl", default="/disk3/hejp/00_thesis/4_predict_translation/results/esm2_rnafm_cross_attention/dataset_esm2_rnafm_cross_MSpp.pkl")
    p.add_argument("--out_csv", default="")
    p.add_argument("--sample_id_col", default="sample_id")
    p.add_argument("--label_col", default="label_raw")
    p.add_argument("--length_col", default="orf_nt_len")
    p.add_argument("--min_orf_nt_len", type=int, default=90)
    p.add_argument("--max_orf_nt_len", type=int, default=453)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--esm2_pattern", default="{sample_id}.pt")
    p.add_argument("--rnafm_up_patterns", default="{sample_id}_upstream.pt")
    p.add_argument("--rnafm_orf_patterns", default="{sample_id}_orf.pt")
    p.add_argument("--rnafm_down_patterns", default="{sample_id}_downstream.pt")
    p.add_argument("--no_balance", action="store_true", help="Do not downsample negatives to match positives.")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    if not args.out_csv:
        args.out_csv = args.out_pkl.replace(".pkl", ".csv.gz")

    df = read_table(args.metadata_csv)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df[args.sample_id_col] = df[args.sample_id_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)

    if args.length_col in df.columns:
        n0 = len(df)
        length = pd.to_numeric(df[args.length_col], errors="coerce")
        df = df[length.between(args.min_orf_nt_len, args.max_orf_nt_len)].copy()
        print(f"Length filter: {n0} -> {len(df)}")

    is_pos = df[args.label_col].str.contains(POS_REGEX, regex=True, na=False) & (df[args.label_col] != NEG_LABEL)
    is_neg = df[args.label_col].eq(NEG_LABEL)
    pos = df[is_pos].copy()
    neg = df[is_neg].copy()

    print(f"Before balance: pos={len(pos)}, neg={len(neg)}")
    if not args.no_balance:
        if len(neg) < len(pos):
            raise ValueError(f"Negative samples are insufficient: pos={len(pos)}, neg={len(neg)}")
        neg = neg.sample(n=len(pos), random_state=args.seed, replace=False)

    pos["binary_label"] = 1
    neg["binary_label"] = 0
    out = pd.concat([pos, neg], axis=0).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out["esm2_orf_path"] = out[args.sample_id_col].apply(
        lambda x: os.path.join(args.esm2_token_dir, args.esm2_pattern.format(sample_id=x))
    )

    up_dir = os.path.join(args.rnafm_token_dir, "upstream")
    orf_dir = os.path.join(args.rnafm_token_dir, "orf")
    down_dir = os.path.join(args.rnafm_token_dir, "downstream")
    up_pats = split_patterns(args.rnafm_up_patterns)
    orf_pats = split_patterns(args.rnafm_orf_patterns)
    down_pats = split_patterns(args.rnafm_down_patterns)

    out["rnafm_up_path"] = out[args.sample_id_col].apply(lambda x: first_path(up_dir, x, up_pats))
    out["rnafm_orf_path"] = out[args.sample_id_col].apply(lambda x: first_path(orf_dir, x, orf_pats))
    out["rnafm_down_path"] = out[args.sample_id_col].apply(lambda x: first_path(down_dir, x, down_pats))

    ok_esm = out["esm2_orf_path"].apply(os.path.exists)
    ok_rna = (
        out["rnafm_up_path"].apply(os.path.exists)
        & out["rnafm_orf_path"].apply(os.path.exists)
        & out["rnafm_down_path"].apply(os.path.exists)
    )
    ok = ok_esm & ok_rna
    print("ESM2 ORF token files ok:", int(ok_esm.sum()), "/", len(out))
    print("RNA-FM token files ok:", int(ok_rna.sum()), "/", len(out))
    print("All token files ok:", int(ok.sum()), "/", len(out))
    if int(ok.sum()) < len(out):
        miss = out.loc[~ok, [args.sample_id_col, "esm2_orf_path", "rnafm_up_path", "rnafm_orf_path", "rnafm_down_path"]].head(20)
        print("Missing examples:")
        print(miss.to_string(index=False))

    out = out[ok].copy().reset_index(drop=True)

    keep = [
        args.sample_id_col, args.label_col, "binary_label",
        "esm2_orf_path", "rnafm_up_path", "rnafm_orf_path", "rnafm_down_path",
    ]
    optional_cols = [
        "protein_accession", "transcript_accession", "protein_len_aa", "orf_nt_len",
        "orf_seq", "tx_orf_seq_with_stop", "upstream_450nt", "downstream_450nt_after_stop",
        "upstream_seq_450nt", "downstream_seq_450nt", "protein_seq_for_esm2",
    ]
    for c in optional_cols:
        if c in out.columns and c not in keep:
            keep.append(c)

    out = out[keep].rename(columns={args.sample_id_col: "sample_id", args.label_col: "label_raw"})

    print("label_raw distribution:")
    print(out["label_raw"].value_counts(dropna=False))
    print("binary_label distribution:")
    print(out["binary_label"].value_counts(dropna=False))

    out.to_pickle(args.out_pkl)
    out.to_csv(args.out_csv, index=False, compression="gzip")
    print("saved:", args.out_pkl)
    print("saved:", args.out_csv)


if __name__ == "__main__":
    main()
