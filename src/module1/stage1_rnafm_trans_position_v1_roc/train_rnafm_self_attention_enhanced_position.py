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
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from build_model_rnafm_self_attention_enhanced_position import (
    RNAFMSelfAttentionEnhancedPositionModel,
)


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
    stream = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(out_dir, f"{run_name}.log"))
    stream.setFormatter(fmt)
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def safe_load(path):
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict):
        for key in ["last_hidden_state", "embedding", "embeddings", "tokens", "x"]:
            if key in x:
                x = x[key]
                break
        else:
            if "hidden_states" in x:
                hidden = x["hidden_states"]
                x = hidden[-1] if isinstance(hidden, (list, tuple)) else hidden
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
        raise ValueError(f"{path} shape should be [seq, dim], got {tuple(x.shape)}")
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def pad_sequences(xs, pad_side="right"):
    lengths = torch.tensor([max(1, x.shape[0]) for x in xs], dtype=torch.long)
    max_len = int(lengths.max())
    out = torch.zeros(len(xs), max_len, xs[0].shape[1], dtype=torch.float32)
    for i, x in enumerate(xs):
        if x.shape[0] == 0:
            continue
        if pad_side == "left":
            out[i, max_len - x.shape[0] :] = x
        elif pad_side == "right":
            out[i, : x.shape[0]] = x
        else:
            raise ValueError(f"Unsupported pad_side: {pad_side}")
    return out, lengths


class RNAFMDataset(Dataset):
    def __init__(self, df, preload=True):
        self.df = df.reset_index(drop=True).copy()
        self.sample_ids = self.df["sample_id"].astype(str).tolist()
        self.labels = self.df["binary_label"].astype(float).values
        self.raw = self.df["label_raw"].astype(str).tolist()
        self.paths = {
            "rna_up": self.df["rnafm_up_path"].tolist(),
            "rna_orf": self.df["rnafm_orf_path"].tolist(),
            "rna_down": self.df["rnafm_down_path"].tolist(),
        }
        self.cache = None
        if preload:
            self.cache = {key: [safe_load(p) for p in paths] for key, paths in self.paths.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = {
            "sample_id": self.sample_ids[index],
            "label_raw": self.raw[index],
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }
        for key in ["rna_up", "rna_orf", "rna_down"]:
            item[key] = self.cache[key][index] if self.cache is not None else safe_load(self.paths[key][index])
        return item


def collate_fn(batch):
    out = {
        "sample_id": [item["sample_id"] for item in batch],
        "label_raw": [item["label_raw"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
    }
    out["rna_up"], out["rna_up_len"] = pad_sequences(
        [item["rna_up"] for item in batch], "left"
    )
    for key in ["rna_orf", "rna_down"]:
        out[key], out[key + "_len"] = pad_sequences(
            [item[key] for item in batch], "right"
        )
    return out


def move_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def calculate_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.nan_to_num(np.asarray(y_prob), nan=0.5), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    result = {
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
    result.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, y_true, y_prob, sample_ids, raw_labels = 0.0, [], [], [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        total_loss += loss.item() * batch["label"].numel()
        probabilities = torch.sigmoid(logits).cpu().numpy()
        y_prob.extend(probabilities.tolist())
        y_true.extend(batch["label"].cpu().numpy().astype(int).tolist())
        sample_ids.extend(batch["sample_id"])
        raw_labels.extend(batch["label_raw"])
    result = calculate_metrics(y_true, y_prob)
    result["loss"] = total_loss / max(1, len(loader.dataset))
    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label_raw": raw_labels,
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": (np.asarray(y_prob) >= 0.5).astype(int),
        }
    )
    return result, predictions


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_pkl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_name", default="rnafm_self_attention_enhanced_position")
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
        help="Validation metric used by LR scheduling, early stopping and best-checkpoint selection.",
    )
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_preload", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger(args.out_dir, args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("args: %s", vars(args))
    logger.info("device: %s", device)

    df = pd.read_pickle(args.dataset_pkl)
    required = {
        "sample_id", "binary_label", "label_raw", "rnafm_up_path",
        "rnafm_orf_path", "rnafm_down_path",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing dataset columns: {missing}")
    logger.info("loaded: %s", df.shape)
    logger.info("label_raw:\n%s", df["label_raw"].value_counts().to_string())
    logger.info("binary_label:\n%s", df["binary_label"].value_counts().to_string())

    train_df, temporary_df = train_test_split(
        df,
        test_size=args.valid_size + args.test_size,
        random_state=args.seed,
        stratify=df["binary_label"],
    )
    relative_test_size = args.test_size / (args.valid_size + args.test_size)
    valid_df, test_df = train_test_split(
        temporary_df,
        test_size=relative_test_size,
        random_state=args.seed,
        stratify=temporary_df["binary_label"],
    )
    logger.info("split: train=%d valid=%d test=%d", len(train_df), len(valid_df), len(test_df))

    train_ds = RNAFMDataset(train_df, preload=not args.no_preload)
    valid_ds = RNAFMDataset(valid_df, preload=not args.no_preload)
    test_ds = RNAFMDataset(test_df, preload=not args.no_preload)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    first = train_ds[0]
    rna_dim = first["rna_up"].shape[1]
    model = RNAFMSelfAttentionEnhancedPositionModel(
        rna_dim=rna_dim,
        branch_hidden_dim=args.branch_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
        position_mode=args.position_mode,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("rna_dim=%d | parameters=%d", rna_dim, parameter_count)
    logger.info("model:\n%s", model)

    n_pos = int((train_df["binary_label"] == 1).sum())
    n_neg = int((train_df["binary_label"] == 0).sum())
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / max(1, n_pos)], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_metric, best_epoch, best_state, bad_epochs = -np.inf, None, None, 0
    history = []
    best_path = os.path.join(args.out_dir, f"{args.run_name}_best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, y_true, y_prob = 0.0, [], []
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * batch["label"].numel()
            y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            y_true.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())

        train_metrics = calculate_metrics(y_true, y_prob)
        train_metrics["loss"] = total_loss / max(1, len(train_loader.dataset))
        valid_metrics, _ = evaluate(model, valid_loader, criterion, device)
        monitor = valid_metrics[args.monitor_metric]
        scheduler.step(monitor)

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
        history.append(row)
        logger.info(
            "Epoch %03d | train loss %.4f auc %.4f ap %.4f f1 %.4f | "
            "valid loss %.4f auc %.4f ap %.4f f1 %.4f mcc %.4f",
            epoch,
            train_metrics["loss"], train_metrics["roc_auc"],
            train_metrics["average_precision"], train_metrics["f1"],
            valid_metrics["loss"], valid_metrics["roc_auc"],
            valid_metrics["average_precision"], valid_metrics["f1"],
            valid_metrics["mcc"],
        )

        if monitor > best_metric:
            best_metric = monitor
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)
            logger.info(
                "new best checkpoint | epoch=%d | valid_%s=%.6f",
                best_epoch,
                args.monitor_metric,
                best_metric,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("early stop at epoch %d", epoch)
                break

    pd.DataFrame(history).to_csv(
        os.path.join(args.out_dir, f"{args.run_name}_history.csv"), index=False
    )
    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint")
    model.load_state_dict(best_state)

    test_metrics, predictions = evaluate(model, test_loader, criterion, device)
    predictions.to_csv(
        os.path.join(args.out_dir, f"{args.run_name}_test_predictions.csv"), index=False
    )
    save_json(
        test_metrics,
        os.path.join(args.out_dir, f"{args.run_name}_test_metrics.json"),
    )
    config = vars(args).copy()
    config.update(
        {
            "rna_dim": rna_dim,
            "parameter_count": parameter_count,
            "monitor_metric": args.monitor_metric,
            "best_valid_metric": float(best_metric),
            f"best_valid_{args.monitor_metric}": float(best_metric),
            "best_epoch": int(best_epoch),
            "upstream_padding": "left",
            "orf_padding": "right",
            "downstream_padding": "right",
        }
    )
    save_json(config, os.path.join(args.out_dir, f"{args.run_name}_config.json"))
    logger.info(
        "BEST CHECKPOINT: epoch=%d | valid_%s=%.6f",
        best_epoch,
        args.monitor_metric,
        best_metric,
    )
    logger.info("TEST: %s", test_metrics)


if __name__ == "__main__":
    main()
