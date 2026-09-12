#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from build_model_esm2_rnafm_cross_attention import ESM2RNAFMCrossAttentionModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def setup_logger(out_dir, run_name):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(out_dir, f"{run_name}.log"))
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def safe_load(path):
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict):
        for k in ["last_hidden_state", "embedding", "embeddings", "tokens", "x"]:
            if k in x:
                x = x[k]
                break
        else:
            if "hidden_states" in x:
                hs = x["hidden_states"]
                x = hs[-1] if isinstance(hs, (list, tuple)) else hs
            else:
                raise KeyError(f"Cannot find tensor key in {path}: {list(x.keys())}")
    if isinstance(x, (list, tuple)):
        x = x[-1]
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.float()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 2:
        raise ValueError(f"{path} shape should be [seq_len, dim], got {tuple(x.shape)}")
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def pad(xs):
    lengths = torch.tensor([max(1, x.shape[0]) for x in xs], dtype=torch.long)
    dim = xs[0].shape[1]
    out = torch.zeros(len(xs), int(lengths.max()), dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, :x.shape[0]] = x
    return out, lengths


class ESM2RNAFMDataset(Dataset):
    def __init__(self, df, preload=True):
        self.df = df.reset_index(drop=True).copy()
        self.sample_ids = self.df["sample_id"].astype(str).tolist()
        self.labels = self.df["binary_label"].astype(float).values
        self.raw = self.df["label_raw"].astype(str).tolist()
        self.paths = {
            "esm2_orf": self.df["esm2_orf_path"].tolist(),
            "rnafm_up": self.df["rnafm_up_path"].tolist(),
            "rnafm_orf": self.df["rnafm_orf_path"].tolist(),
            "rnafm_down": self.df["rnafm_down_path"].tolist(),
        }
        self.cache = None
        if preload:
            self.cache = {k: [safe_load(p) for p in v] for k, v in self.paths.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = {
            "sample_id": self.sample_ids[i],
            "label_raw": self.raw[i],
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
        }
        for k in ["esm2_orf", "rnafm_up", "rnafm_orf", "rnafm_down"]:
            item[k] = self.cache[k][i] if self.cache is not None else safe_load(self.paths[k][i])
        return item


def collate_fn(batch):
    out = {
        "sample_id": [x["sample_id"] for x in batch],
        "label_raw": [x["label_raw"] for x in batch],
        "label": torch.stack([x["label"] for x in batch]),
    }
    for k in ["esm2_orf", "rnafm_up", "rnafm_orf", "rnafm_down"]:
        out[k], out[k + "_len"] = pad([x[k] for x in batch])
    return out


def move_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.nan_to_num(np.asarray(y_prob), nan=0.5), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, y_true, y_prob, sids, raws = 0, [], [], [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        total += loss.item() * batch["label"].numel()
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.extend(prob.tolist())
        y_true.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
        sids.extend(batch["sample_id"])
        raws.extend(batch["label_raw"])
    m = metrics(y_true, y_prob)
    m["loss"] = total / max(1, len(loader.dataset))
    pred = pd.DataFrame({
        "sample_id": sids,
        "label_raw": raws,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": (np.asarray(y_prob) >= 0.5).astype(int),
    })
    return m, pred


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_pkl", default="/disk3/hejp/00_thesis/4_predict_translation/results/esm2_rnafm_cross_attention/dataset_esm2_rnafm_cross_MSpp.pkl")
    p.add_argument("--out_dir", default="/disk3/hejp/00_thesis/4_predict_translation/results/esm2_rnafm_cross_attention")
    p.add_argument("--run_name", default="esm2_rnafm_cross_attention")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--branch_hidden_dim", type=int, default=256)
    p.add_argument("--fusion_hidden_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2, help="Number of bidirectional cross-attention layers.")
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--valid_size", type=float, default=0.15)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_preload", action="store_true", help="Do not cache all token files in RAM before training.")
    p.add_argument("--no_self_encoder", action="store_true", help="Disable one-layer within-modality self-attention before cross-attention.")
    p.add_argument("--monitor", default="average_precision", choices=["average_precision", "roc_auc", "f1", "mcc"])
    args = p.parse_args()

    set_seed(args.seed)
    logger = setup_logger(args.out_dir, args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("args: %s", vars(args))
    logger.info("device: %s", device)

    df = pd.read_pickle(args.dataset_pkl)
    logger.info("loaded: %s", df.shape)
    logger.info("label_raw:\n%s", df["label_raw"].value_counts().to_string())
    logger.info("binary_label:\n%s", df["binary_label"].value_counts().to_string())

    train_df, tmp_df = train_test_split(
        df,
        test_size=args.valid_size + args.test_size,
        random_state=args.seed,
        stratify=df["binary_label"],
    )
    rel_test = args.test_size / (args.valid_size + args.test_size)
    valid_df, test_df = train_test_split(
        tmp_df,
        test_size=rel_test,
        random_state=args.seed,
        stratify=tmp_df["binary_label"],
    )
    logger.info("split: train=%d valid=%d test=%d", len(train_df), len(valid_df), len(test_df))

    train_ds = ESM2RNAFMDataset(train_df, preload=not args.no_preload)
    valid_ds = ESM2RNAFMDataset(valid_df, preload=not args.no_preload)
    test_ds = ESM2RNAFMDataset(test_df, preload=not args.no_preload)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    first = train_ds[0]
    esm_dim = first["esm2_orf"].shape[1]
    rna_dim = first["rnafm_orf"].shape[1]
    logger.info("esm_dim=%d rna_dim=%d", esm_dim, rna_dim)
    logger.info(
        "example token lengths: esm=%d rna_up=%d rna_orf=%d rna_down=%d",
        first["esm2_orf"].shape[0], first["rnafm_up"].shape[0], first["rnafm_orf"].shape[0], first["rnafm_down"].shape[0]
    )

    model = ESM2RNAFMCrossAttentionModel(
        esm_dim=esm_dim,
        rna_dim=rna_dim,
        branch_hidden_dim=args.branch_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
        use_self_encoder=not args.no_self_encoder,
    ).to(device)

    n_pos = int((train_df["binary_label"] == 1).sum())
    n_neg = int((train_df["binary_label"] == 0).sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(1, n_pos)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_metric, best_state, bad = -np.inf, None, 0
    history = []
    best_path = os.path.join(args.out_dir, f"{args.run_name}_best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, y_true, y_prob = 0, [], []
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total += loss.item() * batch["label"].numel()
            y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            y_true.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())

        train_m = metrics(y_true, y_prob)
        train_m["loss"] = total / max(1, len(train_loader.dataset))
        valid_m, _ = evaluate(model, valid_loader, criterion, device)
        monitor_value = valid_m[args.monitor]
        scheduler.step(monitor_value)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"valid_{k}": v for k, v in valid_m.items()})
        history.append(row)

        logger.info(
            "Epoch %03d | train loss %.4f auc %.4f ap %.4f f1 %.4f | valid loss %.4f auc %.4f ap %.4f f1 %.4f mcc %.4f",
            epoch,
            train_m["loss"], train_m["roc_auc"], train_m["average_precision"], train_m["f1"],
            valid_m["loss"], valid_m["roc_auc"], valid_m["average_precision"], valid_m["f1"], valid_m["mcc"],
        )

        if monitor_value > best_metric:
            best_metric = monitor_value
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                logger.info("early stop at epoch %d", epoch)
                break

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, f"{args.run_name}_history.csv"), index=False)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_m, pred = evaluate(model, test_loader, criterion, device)
    pred.to_csv(os.path.join(args.out_dir, f"{args.run_name}_test_predictions.csv"), index=False)
    save_json(test_m, os.path.join(args.out_dir, f"{args.run_name}_test_metrics.json"))
    logger.info("TEST: %s", test_m)


if __name__ == "__main__":
    main()
