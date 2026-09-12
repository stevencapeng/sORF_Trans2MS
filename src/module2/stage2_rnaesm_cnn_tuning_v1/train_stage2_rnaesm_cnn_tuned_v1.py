#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one Stage2 RNA-FM + ESM2 CNN on an existing fixed split.

Two architectures are supported:

baseline
    Exact structural reproduction of the v4 CNN baseline: one parallel
    Conv1d layer (k=3/5/7 by default), BatchNorm, GELU, dropout and masked
    global-max pooling.

enhanced
    Position-aware residual multi-scale CNN. It supports LayerNorm, masked
    mean+max pooling, a protein N-terminal local pool, and explicit
    RNA-protein interaction fusion. No Stage1 checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, Dataset


RNA_UP_CANDIDATES = [
    "rnafm_up_path", "rnafm_upstream_path", "upstream_rnafm_path"
]
RNA_ORF_CANDIDATES = ["rnafm_orf_path", "orf_rnafm_path"]
RNA_DOWN_CANDIDATES = [
    "rnafm_down_path", "rnafm_downstream_path", "downstream_rnafm_path"
]
ESM2_CANDIDATES = [
    "esm2_path", "esm2_token_path", "esm2_orf_path", "protein_esm2_path",
    "esm2_embedding_path", "esm2_tokens_path",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path) -> None:
    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            {key: convert(value) for key, value in obj.items()},
            handle,
            ensure_ascii=False,
            indent=2,
        )


def safe_load(path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")

    if isinstance(value, dict):
        for key in [
            "last_hidden_state", "embedding", "embeddings", "tokens", "x"
        ]:
            if key in value:
                value = value[key]
                break
        else:
            if "hidden_states" not in value:
                raise KeyError(
                    f"Cannot find an embedding tensor in {path}; "
                    f"keys={list(value.keys())}"
                )
            hidden = value["hidden_states"]
            value = hidden[-1] if isinstance(hidden, (list, tuple)) else hidden

    if isinstance(value, (list, tuple)):
        value = value[-1]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    value = value.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(
            f"Expected [length, dimension], got {tuple(value.shape)}: {path}"
        )
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def choose_col(frame, requested, candidates, label):
    if requested:
        if requested not in frame.columns:
            raise ValueError(
                f"{label} column not found: {requested}; "
                f"columns={frame.columns.tolist()}"
            )
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Cannot detect {label} column. Tried {candidates}; "
        f"columns={frame.columns.tolist()}"
    )


def read_ids(path):
    frame = pd.read_csv(path, dtype=str)
    if "sample_id" not in frame.columns:
        if len(frame.columns) != 1:
            raise ValueError(f"No sample_id column in {path}")
        frame = frame.rename(columns={frame.columns[0]: "sample_id"})
    ids = frame["sample_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicated sample IDs in {path}")
    return ids


def apply_fixed_split(frame, split_dir):
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicated sample_id values in dataset")

    split_dir = Path(split_dir)
    train_ids = read_ids(split_dir / "train_ids.csv")
    valid_ids = read_ids(split_dir / "valid_ids.csv")
    test_ids = read_ids(split_dir / "test_ids.csv")
    id_sets = [set(train_ids), set(valid_ids), set(test_ids)]
    if (
        id_sets[0] & id_sets[1]
        or id_sets[0] & id_sets[2]
        or id_sets[1] & id_sets[2]
    ):
        raise ValueError("Fixed split files overlap")

    selected = set().union(*id_sets)
    available = set(frame["sample_id"])
    missing = selected - available
    omitted = available - selected
    if missing:
        raise ValueError(
            f"{len(missing)} split IDs are absent from the dataset; "
            f"examples={sorted(missing)[:10]}"
        )
    if omitted:
        raise ValueError(
            f"{len(omitted)} dataset IDs are omitted by the fixed split; "
            f"examples={sorted(omitted)[:10]}"
        )

    indexed = frame.set_index("sample_id", drop=False)
    subsets = tuple(
        indexed.loc[ids].reset_index(drop=True)
        for ids in [train_ids, valid_ids, test_ids]
    )
    for name, subset in zip(["train", "valid", "test"], subsets):
        if subset["binary_label"].nunique() != 2:
            raise ValueError(f"{name} split lacks one class")
    return subsets


def calc_metrics(y_true, y_prob, threshold=0.5, loss=None):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return {
        "loss": float(
            loss
            if loss is not None
            else log_loss(y_true, y_prob, labels=[0, 1])
        ),
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


class TokenDataset(Dataset):
    def __init__(self, frame, columns):
        self.frame = frame.reset_index(drop=True)
        self.columns = columns

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return {
            "label": torch.tensor(
                float(row["binary_label"]), dtype=torch.float32
            ),
            "rna_up": safe_load(row[self.columns["rna_up"]]),
            "rna_orf": safe_load(row[self.columns["rna_orf"]]),
            "rna_down": safe_load(row[self.columns["rna_down"]]),
            "protein": safe_load(row[self.columns["esm2"]]),
        }


def pad_tokens(tensors, side="right"):
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown padding side: {side}")
    lengths = torch.tensor(
        [tensor.shape[0] for tensor in tensors], dtype=torch.long
    )
    max_length = max(1, max([tensor.shape[0] for tensor in tensors] + [0]))
    output = torch.zeros(
        len(tensors),
        max_length,
        tensors[0].shape[1],
        dtype=torch.float32,
    )
    for index, tensor in enumerate(tensors):
        length = tensor.shape[0]
        if length == 0:
            continue
        if side == "left":
            output[index, max_length - length :] = tensor
        else:
            output[index, :length] = tensor
    return output, lengths


def collate_tokens(batch, upstream_padding="right"):
    output = {"label": torch.stack([item["label"] for item in batch])}
    for key in ["rna_up", "rna_orf", "rna_down", "protein"]:
        side = upstream_padding if key == "rna_up" else "right"
        output[key], output[key + "_len"] = pad_tokens(
            [item[key] for item in batch], side=side
        )
    return output


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def valid_mask(lengths, max_length, padding_side):
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    if padding_side == "left":
        return positions >= (max_length - lengths).unsqueeze(1)
    return positions < lengths.unsqueeze(1)


def add_position_features(
    x, lengths, padding_side, region, position_channels
):
    if position_channels == 0:
        return x
    if position_channels not in {2, 4}:
        raise ValueError("position_channels must be 0, 2, or 4")

    batch_size, max_length, _ = x.shape
    features = x.new_zeros(batch_size, max_length, position_channels)
    for index, length_value in enumerate(lengths.tolist()):
        if length_value <= 0:
            continue
        start = max_length - length_value if padding_side == "left" else 0
        stop = start + length_value
        if length_value == 1:
            coordinate = x.new_zeros(1)
        else:
            coordinate = torch.linspace(
                0.0, 1.0, length_value, device=x.device, dtype=x.dtype
            )
        if region == "upstream":
            coordinate = coordinate - 1.0
        columns = [coordinate, coordinate.square()]
        if position_channels == 4:
            columns.extend(
                [
                    torch.sin(math.pi * coordinate),
                    torch.cos(math.pi * coordinate),
                ]
            )
        features[index, start:stop] = torch.stack(columns, dim=1)
    return torch.cat([x, features], dim=2)


def masked_mean_max(x, mask, mode="meanmax"):
    """Pool x=[B,L,C] using a Boolean [B,L] valid-token mask."""
    mask3 = mask.unsqueeze(-1)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
    max_values = x.masked_fill(
        ~mask3, torch.finfo(x.dtype).min
    ).max(dim=1).values
    empty = mask.sum(dim=1).eq(0)
    if empty.any():
        max_values = max_values.clone()
        max_values[empty] = 0.0
    if mode == "max":
        return max_values
    if mode != "meanmax":
        raise ValueError(f"Unknown pool mode: {mode}")
    mean_values = (x * mask3.to(x.dtype)).sum(dim=1) / counts
    if empty.any():
        mean_values = mean_values.clone()
        mean_values[empty] = 0.0
    return torch.cat([mean_values, max_values], dim=1)


class BaselineConvEncoder(nn.Module):
    """Exact v4 baseline convolution and pooling structure."""

    def __init__(self, input_dim, hidden, kernels, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden, kernel, padding=kernel // 2),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel in kernels
            ]
        )
        self.out_dim = hidden * len(kernels)

    def forward(self, x, lengths, padding_side="right"):
        x = torch.nan_to_num(x.float())
        mask = valid_mask(lengths, x.shape[1], padding_side)
        source = x.transpose(1, 2)
        outputs = []
        for block in self.blocks:
            value = block(source).transpose(1, 2)
            outputs.append(masked_mean_max(value, mask, mode="max"))
        return torch.cat(outputs, dim=1)


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden, kernel, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel // 2)
        self.conv1 = nn.Conv1d(
            hidden, hidden, kernel, padding=padding, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv1d(
            hidden, hidden, kernel, padding=padding, dilation=dilation
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        value = self.conv1(x).transpose(1, 2)
        value = self.dropout(self.activation(self.norm1(value)))
        value = self.conv2(value.transpose(1, 2)).transpose(1, 2)
        value = self.dropout(self.norm2(value)).transpose(1, 2)
        value = self.activation(value + residual)
        return value * mask.unsqueeze(1).to(value.dtype)


class EnhancedConvEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden,
        kernels,
        depth,
        dropout,
        pool_mode,
        position_channels,
        region,
        nterm_window=0,
    ):
        super().__init__()
        self.region = region
        self.pool_mode = pool_mode
        self.position_channels = position_channels
        self.nterm_window = int(nterm_window)
        augmented_dim = input_dim + position_channels
        self.input_norm = nn.LayerNorm(augmented_dim)
        self.stem = nn.Sequential(
            nn.Linear(augmented_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.branches = nn.ModuleList()
        for branch_index, kernel in enumerate(kernels):
            dilation = 1 if branch_index == 0 else min(2, branch_index + 1)
            self.branches.append(
                nn.ModuleList(
                    [
                        ResidualConvBlock(
                            hidden, kernel, dilation=dilation, dropout=dropout
                        )
                        for _ in range(depth)
                    ]
                )
            )
        pool_multiplier = 1 if pool_mode == "max" else 2
        if self.nterm_window > 0:
            pool_multiplier += 2
        self.out_dim = hidden * len(kernels) * pool_multiplier

    def forward(self, x, lengths, padding_side="right"):
        x = torch.nan_to_num(x.float())
        mask = valid_mask(lengths, x.shape[1], padding_side)
        x = add_position_features(
            x,
            lengths,
            padding_side=padding_side,
            region=self.region,
            position_channels=self.position_channels,
        )
        stem = self.stem(self.input_norm(x)).transpose(1, 2)
        stem = stem * mask.unsqueeze(1).to(stem.dtype)

        outputs = []
        for blocks in self.branches:
            value = stem
            for block in blocks:
                value = block(value, mask)
            token_value = value.transpose(1, 2)
            outputs.append(masked_mean_max(token_value, mask, self.pool_mode))
            if self.nterm_window > 0:
                prefix_mask = torch.zeros_like(mask)
                for index, length_value in enumerate(lengths.tolist()):
                    local_length = min(length_value, self.nterm_window)
                    if local_length > 0:
                        prefix_mask[index, :local_length] = True
                outputs.append(
                    masked_mean_max(token_value, prefix_mask, "meanmax")
                )
        return torch.cat(outputs, dim=1)


class Stage2RNACNNProteinCNN(nn.Module):
    def __init__(
        self,
        rna_dim,
        protein_dim,
        architecture="baseline",
        conv_hidden=128,
        kernels=(3, 5, 7),
        conv_depth=1,
        pool_mode="max",
        position_channels=0,
        protein_nterm_window=0,
        branch_hidden=256,
        fusion_hidden=256,
        fusion_mode="concat",
        dropout=0.5,
        upstream_padding="right",
    ):
        super().__init__()
        self.architecture = architecture
        self.upstream_padding = upstream_padding
        self.fusion_mode = fusion_mode

        if architecture == "baseline":
            if pool_mode != "max" or position_channels != 0:
                raise ValueError(
                    "baseline requires pool_mode=max and position_channels=0"
                )
            if conv_depth != 1 or protein_nterm_window != 0:
                raise ValueError(
                    "baseline requires conv_depth=1 and protein_nterm_window=0"
                )
            encoder = lambda dim, region, nterm=0: BaselineConvEncoder(
                dim, conv_hidden, kernels, dropout
            )
        elif architecture == "enhanced":
            encoder = lambda dim, region, nterm=0: EnhancedConvEncoder(
                input_dim=dim,
                hidden=conv_hidden,
                kernels=kernels,
                depth=conv_depth,
                dropout=dropout,
                pool_mode=pool_mode,
                position_channels=position_channels,
                region=region,
                nterm_window=nterm,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.up = encoder(rna_dim, "upstream")
        self.orf = encoder(rna_dim, "orf")
        self.down = encoder(rna_dim, "downstream")
        self.protein = encoder(
            protein_dim, "protein", nterm=protein_nterm_window
        )

        self.rna_head = nn.Sequential(
            nn.Linear(
                self.up.out_dim + self.orf.out_dim + self.down.out_dim,
                branch_hidden,
            ),
            nn.LayerNorm(branch_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.protein_head = nn.Sequential(
            nn.Linear(self.protein.out_dim, branch_hidden),
            nn.LayerNorm(branch_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fusion_multiplier = 2 if fusion_mode == "concat" else 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_multiplier * branch_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, max(32, fusion_hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, fusion_hidden // 2), 1),
        )

    def forward(self, batch):
        rna = self.rna_head(
            torch.cat(
                [
                    self.up(
                        batch["rna_up"],
                        batch["rna_up_len"],
                        padding_side=self.upstream_padding,
                    ),
                    self.orf(
                        batch["rna_orf"], batch["rna_orf_len"]
                    ),
                    self.down(
                        batch["rna_down"], batch["rna_down_len"]
                    ),
                ],
                dim=1,
            )
        )
        protein = self.protein_head(
            self.protein(batch["protein"], batch["protein_len"])
        )
        if self.fusion_mode == "concat":
            fused = torch.cat([rna, protein], dim=1)
        elif self.fusion_mode == "interaction":
            fused = torch.cat(
                [rna, protein, rna * protein, torch.abs(rna - protein)], dim=1
            )
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        return self.fusion(fused).squeeze(1)


def evaluate(model, loader, device, criterion):
    model.eval()
    labels, probabilities = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * len(logits)
            labels.extend(batch["label"].cpu().numpy())
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
    return (
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
        total_loss / len(labels),
    )


def train_model(model, train_loader, valid_loader, device, args):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=1e-6
        )
    else:
        scheduler = None

    best_auc = -math.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(logits)
            total_n += len(logits)

        valid_y, valid_probability, valid_loss = evaluate(
            model, valid_loader, device, criterion
        )
        valid_auc = roc_auc_score(valid_y, valid_probability)
        valid_aupr = average_precision_score(valid_y, valid_probability)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_n,
                "valid_loss": valid_loss,
                "valid_roc_auc": valid_auc,
                "valid_average_precision": valid_aupr,
                "lr": current_lr,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={total_loss / total_n:.5f} "
            f"valid_loss={valid_loss:.5f} valid_auc={valid_auc:.5f} "
            f"valid_aupr={valid_aupr:.5f} lr={current_lr:.2e}",
            flush=True,
        )

        if valid_auc > best_auc + 1e-6:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.scheduler == "plateau":
            scheduler.step(valid_auc)
        elif args.scheduler == "cosine":
            scheduler.step()

        if bad_epochs >= args.patience:
            print(
                f"Early stop; best epoch={best_epoch}, "
                f"valid AUC={best_auc:.5f}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, best_auc


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_pkl", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--config_name", default="custom")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rna_up_col", default=None)
    parser.add_argument("--rna_orf_col", default=None)
    parser.add_argument("--rna_down_col", default=None)
    parser.add_argument("--esm2_col", default=None)

    parser.add_argument(
        "--architecture", choices=["baseline", "enhanced"], default="baseline"
    )
    parser.add_argument("--conv_hidden", type=int, default=128)
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--conv_depth", type=int, default=1)
    parser.add_argument(
        "--pool_mode", choices=["max", "meanmax"], default="max"
    )
    parser.add_argument(
        "--position_channels", type=int, choices=[0, 2, 4], default=0
    )
    parser.add_argument("--protein_nterm_window", type=int, default=0)
    parser.add_argument("--branch_hidden", type=int, default=256)
    parser.add_argument("--fusion_hidden", type=int, default=256)
    parser.add_argument(
        "--fusion_mode", choices=["concat", "interaction"], default="concat"
    )
    parser.add_argument(
        "--upstream_padding", choices=["left", "right"], default="right"
    )
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--scheduler", choices=["none", "plateau", "cosine"], default="none"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if any(kernel <= 0 or kernel % 2 == 0 for kernel in args.kernels):
        raise ValueError("All kernels must be positive odd integers")
    if args.conv_depth < 1:
        raise ValueError("conv_depth must be >= 1")
    if args.protein_nterm_window < 0:
        raise ValueError("protein_nterm_window must be >= 0")

    set_seed(args.seed)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_pickle(args.dataset_pkl).copy()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["binary_label"] = pd.to_numeric(
        frame["binary_label"], errors="raise"
    ).astype(int)
    columns = {
        "rna_up": choose_col(
            frame, args.rna_up_col, RNA_UP_CANDIDATES, "RNA upstream"
        ),
        "rna_orf": choose_col(
            frame, args.rna_orf_col, RNA_ORF_CANDIDATES, "RNA ORF"
        ),
        "rna_down": choose_col(
            frame, args.rna_down_col, RNA_DOWN_CANDIDATES, "RNA downstream"
        ),
        "esm2": choose_col(frame, args.esm2_col, ESM2_CANDIDATES, "ESM2"),
    }
    train_frame, valid_frame, test_frame = apply_fixed_split(
        frame, args.split_dir
    )
    print("Embedding columns:", columns)
    print(
        f"Fixed split: train={len(train_frame)}, valid={len(valid_frame)}, "
        f"test={len(test_frame)}"
    )

    rna_dim = safe_load(frame.iloc[0][columns["rna_orf"]]).shape[1]
    protein_dim = safe_load(frame.iloc[0][columns["esm2"]]).shape[1]
    collate = partial(
        collate_tokens, upstream_padding=args.upstream_padding
    )
    loaders = []
    for subset, shuffle in [
        (train_frame, True),
        (valid_frame, False),
        (test_frame, False),
    ]:
        loaders.append(
            DataLoader(
                TokenDataset(subset, columns),
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
                collate_fn=collate,
            )
        )

    model = Stage2RNACNNProteinCNN(
        rna_dim=rna_dim,
        protein_dim=protein_dim,
        architecture=args.architecture,
        conv_hidden=args.conv_hidden,
        kernels=tuple(args.kernels),
        conv_depth=args.conv_depth,
        pool_mode=args.pool_mode,
        position_channels=args.position_channels,
        protein_nterm_window=args.protein_nterm_window,
        branch_hidden=args.branch_hidden,
        fusion_hidden=args.fusion_hidden,
        fusion_mode=args.fusion_mode,
        dropout=args.dropout,
        upstream_padding=args.upstream_padding,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Trainable parameters:", f"{count_parameters(model):,}")
    print(model)

    model, history, best_epoch, best_valid_auc = train_model(
        model, loaders[0], loaders[1], device, args
    )
    criterion = nn.BCEWithLogitsLoss()
    y_true, y_probability, test_loss = evaluate(
        model, loaders[2], device, criterion
    )
    valid_y, valid_probability, valid_loss = evaluate(
        model, loaders[1], device, criterion
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "Stage2RNACNNProteinCNN",
        "config_name": args.config_name,
        "args": vars(args),
        "columns": columns,
        "rna_dim": rna_dim,
        "protein_dim": protein_dim,
        "best_epoch": best_epoch,
        "best_valid_roc_auc": best_valid_auc,
    }
    torch.save(
        checkpoint, output_dir / f"{args.run_name}_best_model.pt"
    )
    history.to_csv(
        output_dir / f"{args.run_name}_training_history.csv", index=False
    )

    raw_labels = (
        test_frame["label_raw"].astype(str).to_numpy()
        if "label_raw" in test_frame.columns
        else np.array(["NA"] * len(test_frame))
    )
    pd.DataFrame(
        {
            "sample_id": test_frame["sample_id"].astype(str).to_numpy(),
            "label_raw": raw_labels,
            "y_true": test_frame["binary_label"].astype(int).to_numpy(),
            "y_prob": y_probability,
            "y_pred": (y_probability >= args.threshold).astype(int),
        }
    ).to_csv(
        output_dir / f"{args.run_name}_test_predictions.csv", index=False
    )

    metrics = calc_metrics(
        y_true, y_probability, threshold=args.threshold, loss=test_loss
    )
    metrics.update(
        {
            "model": "cnn",
            "config_name": args.config_name,
            "run_name": args.run_name,
            "best_epoch": int(best_epoch),
            "best_valid_roc_auc": float(best_valid_auc),
            "valid_roc_auc_reloaded": float(
                roc_auc_score(valid_y, valid_probability)
            ),
            "valid_average_precision": float(
                average_precision_score(valid_y, valid_probability)
            ),
            "valid_loss": float(valid_loss),
            "trainable_parameters": int(count_parameters(model)),
        }
    )
    save_json(
        metrics, output_dir / f"{args.run_name}_test_metrics.json"
    )
    save_json(
        {
            "dataset_pkl": args.dataset_pkl,
            "split_dir": args.split_dir,
            "train_n": len(train_frame),
            "valid_n": len(valid_frame),
            "test_n": len(test_frame),
            "columns": columns,
            "config": vars(args),
        },
        output_dir / f"{args.run_name}_split_info.json",
    )

    print("\nTEST METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
