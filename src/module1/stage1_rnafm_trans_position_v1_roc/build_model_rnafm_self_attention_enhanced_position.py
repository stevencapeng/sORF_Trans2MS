#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 1 global Transformer with explicit anchor-aware position channels."""

import math

import torch
import torch.nn as nn


def make_padding_mask(lengths, max_len, pad_side="right"):
    """Return True at padding positions."""
    lengths = lengths.to(dtype=torch.long).clamp(min=1, max=max_len)
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    if pad_side == "right":
        return idx >= lengths.unsqueeze(1)
    if pad_side == "left":
        return idx < (max_len - lengths).unsqueeze(1)
    raise ValueError(f"Unsupported pad_side: {pad_side}")


def make_position_channels(lengths, max_len, region, pad_side):
    """Build p, p^2, sin(pi*p), cos(pi*p) for valid tokens only.

    Coordinates are defined within each biological region:
      upstream:   -1 -> 0 (distal -> AUG)
      ORF:         0 -> 1 (AUG -> stop)
      downstream:  0 -> 1 (stop -> distal)
    """
    device = lengths.device
    dtype = torch.float32
    lengths = lengths.to(device=device, dtype=torch.long).clamp(min=1, max=max_len)
    mask = make_padding_mask(lengths, max_len, pad_side)
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    start = max_len - lengths.unsqueeze(1) if pad_side == "left" else 0
    rank = (idx - start).clamp(min=0).to(dtype)
    fraction = rank / (lengths.unsqueeze(1) - 1).clamp(min=1).to(dtype)

    if region == "upstream":
        positions = -1.0 + fraction
    elif region in {"orf", "downstream"}:
        positions = fraction
    else:
        raise ValueError(f"Unsupported region: {region}")
    positions = positions.masked_fill(mask, 0.0)

    channels = torch.stack(
        [
            positions,
            positions.square(),
            torch.sin(math.pi * positions),
            torch.cos(math.pi * positions),
        ],
        dim=-1,
    )
    return channels.masked_fill(mask.unsqueeze(-1), 0.0), mask


def masked_mean_pool(x, mask):
    valid = (~mask).unsqueeze(-1).to(x.dtype)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def masked_max_pool(x, mask):
    return x.masked_fill(mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values


class SegmentProjector(nn.Module):
    def __init__(
        self,
        input_dim=640,
        hidden_dim=256,
        dropout=0.2,
        position_mode="enhanced",
        region="upstream",
        pad_side="right",
    ):
        super().__init__()
        if position_mode not in {"none", "enhanced"}:
            raise ValueError("position_mode must be 'none' or 'enhanced'")
        self.position_mode = position_mode
        self.region = region
        self.pad_side = pad_side
        position_dim = 4 if position_mode == "enhanced" else 0
        self.proj = nn.Sequential(
            nn.Linear(input_dim + position_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, lengths):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        lengths = lengths.to(x.device)
        mask = make_padding_mask(lengths, x.shape[1], self.pad_side)
        if self.position_mode == "enhanced":
            pos, pos_mask = make_position_channels(
                lengths, x.shape[1], self.region, self.pad_side
            )
            if not torch.equal(mask, pos_mask):
                raise RuntimeError("Position mask and padding mask are inconsistent")
            x = torch.cat([x, pos.to(x.dtype)], dim=-1)
        z = self.proj(x).masked_fill(mask.unsqueeze(-1), 0.0)
        return z, mask


class RNAFMSelfAttentionEnhancedPositionModel(nn.Module):
    """Three RNA regions -> global Transformer -> mean+max pooling -> classifier."""

    def __init__(
        self,
        rna_dim=640,
        branch_hidden_dim=256,
        fusion_hidden_dim=256,
        dropout=0.2,
        num_heads=8,
        num_layers=2,
        ff_mult=2,
        position_mode="enhanced",
    ):
        super().__init__()
        self.position_mode = position_mode
        self.up_proj = SegmentProjector(
            rna_dim, branch_hidden_dim, dropout, position_mode, "upstream", "left"
        )
        self.orf_proj = SegmentProjector(
            rna_dim, branch_hidden_dim, dropout, position_mode, "orf", "right"
        )
        self.down_proj = SegmentProjector(
            rna_dim, branch_hidden_dim, dropout, position_mode, "downstream", "right"
        )
        self.segment_embed = nn.Embedding(3, branch_hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=branch_hidden_dim,
            nhead=num_heads,
            dim_feedforward=branch_hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(branch_hidden_dim)
        middle_dim = max(1, fusion_hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(branch_hidden_dim * 2, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, middle_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(middle_dim, 1),
        )

    def _add_segment(self, z, seg_id, mask):
        seg = torch.full(
            (z.shape[0], z.shape[1]), seg_id, dtype=torch.long, device=z.device
        )
        return (z + self.segment_embed(seg)).masked_fill(mask.unsqueeze(-1), 0.0)

    def forward(self, batch, return_features=False):
        up, up_mask = self.up_proj(batch["rna_up"], batch["rna_up_len"])
        orf, orf_mask = self.orf_proj(batch["rna_orf"], batch["rna_orf_len"])
        down, down_mask = self.down_proj(batch["rna_down"], batch["rna_down_len"])

        x = torch.cat(
            [
                self._add_segment(up, 0, up_mask),
                self._add_segment(orf, 1, orf_mask),
                self._add_segment(down, 2, down_mask),
            ],
            dim=1,
        )
        mask = torch.cat([up_mask, orf_mask, down_mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.out_norm(x).masked_fill(mask.unsqueeze(-1), 0.0)
        pooled = torch.cat(
            [masked_mean_pool(x, mask), masked_max_pool(x, mask)], dim=1
        )
        logits = self.classifier(pooled).squeeze(1)
        if return_features:
            return {"logits": logits, "tokens": x, "mask": mask, "pooled": pooled}
        return logits


# Convenient alias for downstream scripts that expect the old class name.
RNAFMSelfAttentionModel = RNAFMSelfAttentionEnhancedPositionModel
