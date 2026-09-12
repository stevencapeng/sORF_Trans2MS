#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shutil
import argparse

import torch
import pandas as pd
from tqdm import tqdm


DNA_BASES = {"A", "T", "C", "G", "N"}

DNA_TO_RNA_COMP = str.maketrans({
    "A": "U",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
})


def read_table(path):
    if path.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path, compression="infer")


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x


def clean_dna(seq):
    """
    Clean DNA/RNA sequence.
    Keep only A/T/C/G/N.
    If U appears, convert it back to T first.
    """
    if pd.isna(seq):
        return ""

    seq = str(seq).upper().replace("U", "T")
    return "".join(base if base in DNA_BASES else "N" for base in seq)


def dna_to_rna(seq, dna_source="coding"):
    """
    Convert DNA to RNA.

    dna_source:
      coding        : coding/sense DNA 5'->3' -> mRNA = T replaced by U
      template_3to5 : template DNA 3'->5'     -> mRNA = complementary RNA
      template_5to3 : template DNA 5'->3'     -> mRNA = reverse-complementary RNA
    """
    seq = clean_dna(seq)

    if dna_source == "coding":
        return seq.replace("T", "U")

    if dna_source == "template_3to5":
        return seq.translate(DNA_TO_RNA_COMP)

    if dna_source == "template_5to3":
        return seq[::-1].translate(DNA_TO_RNA_COMP)

    raise ValueError(
        "dna_source must be one of: coding, template_3to5, template_5to3"
    )


def infer_aa_len_from_orf(seq, dna_source="coding"):
    """
    Infer protein length from ORF nucleotide sequence.
    If ORF ends with stop codon, remove stop codon from aa length.
    """
    rna = dna_to_rna(seq, dna_source=dna_source)

    if len(rna) < 3:
        return None

    nt_len = len(rna) - len(rna) % 3
    if nt_len < 3:
        return None

    rna = rna[:nt_len]
    aa_len = nt_len // 3

    if rna[-3:] in {"UAA", "UAG", "UGA"}:
        aa_len -= 1

    return aa_len


def get_aa_len(df, aa_len_col, orf_col, dna_source):
    if aa_len_col and aa_len_col in df.columns:
        print(f"[INFO] Use aa length column: {aa_len_col}")
        return pd.to_numeric(df[aa_len_col], errors="coerce")

    print("[INFO] Infer aa length from ORF sequence")
    return df[orf_col].map(
        lambda x: infer_aa_len_from_orf(x, dna_source=dna_source)
    )


def ensure_local_weight(model_name, local_model_dir):
    """
    Prepare local RNA-FM/mRNA-FM weight for fm.pretrained.*().
    The RNA-FM package internally uses torch.hub.load_state_dict_from_url.
    We copy local weight into TORCH_HOME/hub/checkpoints to avoid online download.
    """
    os.environ["TORCH_HOME"] = os.path.join(local_model_dir, ".cache")

    if model_name == "rna-fm":
        filename = "RNA-FM_pretrained.pth"
    elif model_name == "mrna-fm":
        filename = "mRNA-FM_pretrained.pth"
    else:
        raise ValueError("model_name must be 'rna-fm' or 'mrna-fm'")

    src = os.path.join(local_model_dir, filename)
    cache_dir = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints")
    dst = os.path.join(cache_dir, filename)

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Local weight not found: {src}\n"
            f"Please check --local_model_dir."
        )

    os.makedirs(cache_dir, exist_ok=True)

    if (not os.path.exists(dst)) or (os.path.getsize(src) != os.path.getsize(dst)):
        print(f"[INFO] Copy local weight:\n  {src}\n  -> {dst}")
        shutil.copy2(src, dst)
    else:
        print(f"[INFO] Use local cached weight: {dst}")

    return dst


def load_model(model_name, local_model_dir, device):
    ensure_local_weight(model_name, local_model_dir)

    import fm

    print(f"[INFO] Loading model: {model_name}")

    if model_name == "rna-fm":
        model, alphabet = fm.pretrained.rna_fm_t12()
        hidden_dim = 640
        token_unit = "nt"
    elif model_name == "mrna-fm":
        model, alphabet = fm.pretrained.mrna_fm_t12()
        hidden_dim = 1280
        token_unit = "codon"
    else:
        raise ValueError("model_name must be 'rna-fm' or 'mrna-fm'")

    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    print(f"[INFO] Model loaded on {device}")
    print(f"[INFO] hidden_dim = {hidden_dim}")
    print(f"[INFO] token_unit = {token_unit}")

    return model, batch_converter, hidden_dim, token_unit


def prepare_rna(seq, model_name, dna_source, max_len, too_long):
    rna = dna_to_rna(seq, dna_source=dna_source)

    if len(rna) > max_len:
        if too_long == "skip":
            return None, "too_long"
        elif too_long == "truncate":
            rna = rna[:max_len]
        else:
            raise ValueError("too_long must be 'skip' or 'truncate'")

    if model_name == "mrna-fm":
        rna = rna[:len(rna) // 3 * 3]

    if len(rna) == 0:
        return "", "empty_seq"

    return rna, "ok"


def count_model_tokens(rna_seq, model_name):
    if model_name == "mrna-fm":
        return len(rna_seq) // 3
    return len(rna_seq)


def save_empty_embedding(save_path, hidden_dim):
    torch.save(torch.empty(0, hidden_dim), save_path)


@torch.no_grad()
def save_token_embeddings(
    df,
    region,
    seq_col,
    out_dir,
    model,
    batch_converter,
    hidden_dim,
    model_name,
    dna_source,
    device,
    batch_size,
    max_len,
    too_long,
    repr_layer,
    id_col,
    label_col,
):
    save_dir = os.path.join(out_dir, region)
    os.makedirs(save_dir, exist_ok=True)

    summary = []

    batch_data = []
    batch_meta = []

    def flush_batch():
        nonlocal batch_data, batch_meta, summary

        if len(batch_data) == 0:
            return

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        outputs = model(batch_tokens, repr_layers=[repr_layer])
        hidden = outputs["representations"][repr_layer]

        for j, meta in enumerate(batch_meta):
            token_len = meta["token_len"]
            save_path = meta["save_path"]

            # Remove BOS/CLS token; keep only real RNA/codon token embeddings.
            token_emb = hidden[j, 1:1 + token_len, :].detach().cpu()

            if token_emb.shape[0] != token_len:
                raise RuntimeError(
                    f"Token length mismatch: sample_id={meta['sample_id']}, "
                    f"region={region}, expected={token_len}, got={token_emb.shape[0]}"
                )

            torch.save(token_emb, save_path)

            meta["hidden_dim"] = int(token_emb.shape[1])
            meta["status"] = "ok"
            summary.append(meta)

        batch_data = []
        batch_meta = []

    for row_index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Extracting {region}",
    ):
        raw_seq = row[seq_col]

        rna_seq, status = prepare_rna(
            raw_seq,
            model_name=model_name,
            dna_source=dna_source,
            max_len=max_len,
            too_long=too_long,
        )

        sample_id = str(row[id_col]) if id_col in df.columns else f"rtorf_{row_index:07d}"
        sample_id = safe_name(sample_id)

        label = row[label_col] if label_col in df.columns else ""

        # Mimic DNABERT output:
        # downstream/rtorf_0000001_downstream.pt
        save_path = os.path.join(save_dir, f"{sample_id}_{region}.pt")

        if status != "ok":
            if status == "empty_seq":
                save_empty_embedding(save_path, hidden_dim)

            summary.append({
                "row_index": row_index,
                "sample_id": sample_id,
                "label": label,
                "region": region,
                "seq_col": seq_col,
                "dna_source": dna_source,
                "rna_len": 0 if rna_seq is None else len(rna_seq),
                "token_len": 0,
                "hidden_dim": hidden_dim,
                "save_path": save_path if status == "empty_seq" else "",
                "status": status,
            })
            continue

        token_len = count_model_tokens(rna_seq, model_name)

        batch_data.append((sample_id, rna_seq))
        batch_meta.append({
            "row_index": row_index,
            "sample_id": sample_id,
            "label": label,
            "region": region,
            "seq_col": seq_col,
            "dna_source": dna_source,
            "rna_len": len(rna_seq),
            "token_len": token_len,
            "hidden_dim": hidden_dim,
            "save_path": save_path,
            "status": "pending",
        })

        if len(batch_data) >= batch_size:
            flush_batch()

    flush_batch()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract RNA-FM token-level embeddings for microprotein regions."
    )

    parser.add_argument(
        "--input_table",
        default="/disk3/hejp/00_thesis/4_predict_translation/data/processed/ribotricer_and_MSpp_and_double_minus_nt450/train_table_4class_with_ribotricer_rtorfs.pkl",
    )

    parser.add_argument(
        "--out_dir",
        default="/disk4hu/hejp/00_thesis/4_predict_translation/results/features/features_rnafm_token_ribotricer_openprot_nt450",
    )

    parser.add_argument(
        "--local_model_dir",
        default="/disk3/hejp/00_thesis/4_predict_translation/model/rnafm",
    )

    parser.add_argument(
        "--model_name",
        default="rna-fm",
        choices=["rna-fm", "mrna-fm"],
    )

    parser.add_argument(
        "--run_region",
        default="all",
        choices=["upstream", "orf", "downstream", "orf_downstream", "all"],
    )

    parser.add_argument("--upstream_col", default="upstream_seq_450nt")
    parser.add_argument("--orf_col", default="orf_seq")
    parser.add_argument("--downstream_col", default="downstream_seq_450nt")

    parser.add_argument("--id_col", default="sample_id")
    parser.add_argument("--label_col", default="label_raw")
    parser.add_argument("--aa_len_col", default=None)

    parser.add_argument("--min_aa", type=int, default=30)
    parser.add_argument("--max_aa", type=int, default=150)

    parser.add_argument(
        "--dna_source",
        default="coding",
        choices=["coding", "template_3to5", "template_5to3"],
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--repr_layer", type=int, default=12)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--upstream_max_len", type=int, default=450)
    parser.add_argument("--orf_max_len", type=int, default=453)
    parser.add_argument("--downstream_max_len", type=int, default=450)

    parser.add_argument(
        "--too_long",
        default="skip",
        choices=["skip", "truncate"],
    )

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("[INFO] RNA-FM token-level feature extraction")
    print("=" * 80)
    print("[INFO] input_table:", args.input_table)
    print("[INFO] out_dir:", args.out_dir)
    print("[INFO] local_model_dir:", args.local_model_dir)
    print("[INFO] model_name:", args.model_name)
    print("[INFO] run_region:", args.run_region)
    print("[INFO] dna_source:", args.dna_source)
    print("[INFO] device:", device)

    print("\n[INFO] Loading input table...")
    df = read_table(args.input_table)

    print("[INFO] Input shape:", df.shape)
    print("[INFO] Columns:", list(df.columns))

    if args.orf_col not in df.columns:
        raise ValueError(f"Cannot find ORF column: {args.orf_col}")

    aa_len = get_aa_len(
        df=df,
        aa_len_col=args.aa_len_col,
        orf_col=args.orf_col,
        dna_source=args.dna_source,
    )

    keep = (aa_len >= args.min_aa) & (aa_len <= args.max_aa)
    df = df.loc[keep].copy()
    df["rnafm_used_aa_len"] = aa_len.loc[df.index].values

    print(f"\n[INFO] Keep microproteins: {args.min_aa} <= aa_len <= {args.max_aa}")
    print("[INFO] Remaining samples:", df.shape[0])

    model_prefix = args.model_name.replace("-", "")

    filtered_csv = os.path.join(
        args.out_dir,
        f"{model_prefix}_filtered_{args.min_aa}_{args.max_aa}aa_rows.csv"
    )

    df.reset_index(names="row_index").to_csv(filtered_csv, index=False)
    print("[INFO] Filtered rows saved:", filtered_csv)

    model, batch_converter, hidden_dim, token_unit = load_model(
        model_name=args.model_name,
        local_model_dir=args.local_model_dir,
        device=device,
    )

    region_map = {
        "upstream": (args.upstream_col, args.upstream_max_len),
        "orf": (args.orf_col, args.orf_max_len),
        "downstream": (args.downstream_col, args.downstream_max_len),
    }

    if args.run_region == "all":
        regions = ["upstream", "orf", "downstream"]
    elif args.run_region == "orf_downstream":
        regions = ["orf", "downstream"]
    else:
        regions = [args.run_region]

    all_summary = []

    for region in regions:
        seq_col, max_len = region_map[region]

        if seq_col not in df.columns:
            raise ValueError(f"Cannot find column for {region}: {seq_col}")

        print("\n" + "-" * 80)
        print(f"[INFO] Region: {region}")
        print(f"[INFO] seq_col: {seq_col}")
        print(f"[INFO] max_len: {max_len}")
        print("-" * 80)

        region_summary = save_token_embeddings(
            df=df,
            region=region,
            seq_col=seq_col,
            out_dir=args.out_dir,
            model=model,
            batch_converter=batch_converter,
            hidden_dim=hidden_dim,
            model_name=args.model_name,
            dna_source=args.dna_source,
            device=device,
            batch_size=args.batch_size,
            max_len=max_len,
            too_long=args.too_long,
            repr_layer=args.repr_layer,
            id_col=args.id_col,
            label_col=args.label_col,
        )

        all_summary += region_summary

    summary_df = pd.DataFrame(all_summary)

    summary_csv = os.path.join(
        args.out_dir,
        f"{model_prefix}_token_embedding_summary.csv"
    )

    summary_json = os.path.join(
        args.out_dir,
        f"{model_prefix}_token_embedding_summary.json"
    )

    summary_df.to_csv(summary_csv, index=False)

    summary = {
        "input_table": args.input_table,
        "out_dir": args.out_dir,
        "local_model_dir": args.local_model_dir,
        "model_name": args.model_name,
        "model_prefix": model_prefix,
        "dna_source": args.dna_source,
        "device": str(device),
        "batch_size": args.batch_size,
        "repr_layer": args.repr_layer,
        "hidden_dim": hidden_dim,
        "token_unit": token_unit,
        "min_aa": args.min_aa,
        "max_aa": args.max_aa,
        "too_long": args.too_long,
        "n_samples_after_filter": int(df.shape[0]),
        "regions": regions,
        "summary_csv": summary_csv,
        "filtered_csv": filtered_csv,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("[DONE] RNA-FM token embeddings finished.")
    print("[INFO] Summary CSV:", summary_csv)
    print("[INFO] Summary JSON:", summary_json)
    print("=" * 80)


if __name__ == "__main__":
    main()