#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def make_mask(lengths, max_len, device):
    lengths = lengths.to(device).long().clamp(min=1, max=max_len)
    return torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)


def masked_mean_pool(x, mask):
    valid = (~mask).unsqueeze(-1).float()
    return (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)


def masked_max_pool(x, mask):
    return x.masked_fill(mask.unsqueeze(-1), -1e9).max(1).values


class TokenProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, lengths):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = make_mask(lengths, x.shape[1], x.device)
        z = self.proj(x).masked_fill(mask.unsqueeze(-1), 0.0)
        return z, mask


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention block.

    ESM2 tokens attend to RNA-FM tokens, and RNA-FM tokens attend to ESM2 tokens.
    """
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.3, ff_mult=4):
        super().__init__()
        self.esm_to_rna = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.rna_to_esm = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.esm_norm1 = nn.LayerNorm(hidden_dim)
        self.rna_norm1 = nn.LayerNorm(hidden_dim)
        self.esm_norm2 = nn.LayerNorm(hidden_dim)
        self.rna_norm2 = nn.LayerNorm(hidden_dim)

        self.esm_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )
        self.rna_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, esm, esm_mask, rna, rna_mask):
        # Pre-norm attention.
        esm_q = self.esm_norm1(esm)
        rna_kv = self.rna_norm1(rna)
        esm_delta, _ = self.esm_to_rna(
            query=esm_q,
            key=rna_kv,
            value=rna_kv,
            key_padding_mask=rna_mask,
            need_weights=False,
        )
        esm = esm + esm_delta.masked_fill(esm_mask.unsqueeze(-1), 0.0)

        rna_q = self.rna_norm1(rna)
        esm_kv = self.esm_norm1(esm)
        rna_delta, _ = self.rna_to_esm(
            query=rna_q,
            key=esm_kv,
            value=esm_kv,
            key_padding_mask=esm_mask,
            need_weights=False,
        )
        rna = rna + rna_delta.masked_fill(rna_mask.unsqueeze(-1), 0.0)

        esm = esm + self.esm_ffn(self.esm_norm2(esm)).masked_fill(esm_mask.unsqueeze(-1), 0.0)
        rna = rna + self.rna_ffn(self.rna_norm2(rna)).masked_fill(rna_mask.unsqueeze(-1), 0.0)
        return esm, rna


class ESM2RNAFMCrossAttentionModel(nn.Module):
    """ESM2 protein ORF token + RNA-FM transcript-context token cross-attention classifier.

    Batch keys:
      esm2_orf, esm2_orf_len
      rnafm_up, rnafm_up_len
      rnafm_orf, rnafm_orf_len
      rnafm_down, rnafm_down_len
    """

    def __init__(
        self,
        esm_dim=1280,
        rna_dim=640,
        branch_hidden_dim=256,
        fusion_hidden_dim=256,
        dropout=0.3,
        num_heads=8,
        num_layers=2,
        ff_mult=4,
        use_self_encoder=True,
    ):
        super().__init__()
        if branch_hidden_dim % num_heads != 0:
            raise ValueError(f"branch_hidden_dim ({branch_hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.esm_proj = TokenProjector(esm_dim, branch_hidden_dim, dropout)
        self.rna_up_proj = TokenProjector(rna_dim, branch_hidden_dim, dropout)
        self.rna_orf_proj = TokenProjector(rna_dim, branch_hidden_dim, dropout)
        self.rna_down_proj = TokenProjector(rna_dim, branch_hidden_dim, dropout)
        self.rna_segment_embed = nn.Embedding(3, branch_hidden_dim)

        self.use_self_encoder = use_self_encoder
        if use_self_encoder:
            esm_layer = nn.TransformerEncoderLayer(
                d_model=branch_hidden_dim,
                nhead=num_heads,
                dim_feedforward=branch_hidden_dim * ff_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            rna_layer = nn.TransformerEncoderLayer(
                d_model=branch_hidden_dim,
                nhead=num_heads,
                dim_feedforward=branch_hidden_dim * ff_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.esm_self_encoder = nn.TransformerEncoder(esm_layer, num_layers=1)
            self.rna_self_encoder = nn.TransformerEncoder(rna_layer, num_layers=1)

        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(branch_hidden_dim, num_heads, dropout, ff_mult)
            for _ in range(num_layers)
        ])

        self.esm_out_norm = nn.LayerNorm(branch_hidden_dim)
        self.rna_out_norm = nn.LayerNorm(branch_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(branch_hidden_dim * 4, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, max(1, fusion_hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, fusion_hidden_dim // 2), 1),
        )

    def _add_rna_segment(self, z, seg_id):
        seg = torch.full((z.shape[0], z.shape[1]), seg_id, dtype=torch.long, device=z.device)
        return z + self.rna_segment_embed(seg)

    def forward(self, batch):
        esm, esm_mask = self.esm_proj(batch["esm2_orf"], batch["esm2_orf_len"])

        up, up_mask = self.rna_up_proj(batch["rnafm_up"], batch["rnafm_up_len"])
        orf, orf_mask = self.rna_orf_proj(batch["rnafm_orf"], batch["rnafm_orf_len"])
        down, down_mask = self.rna_down_proj(batch["rnafm_down"], batch["rnafm_down_len"])
        rna = torch.cat([
            self._add_rna_segment(up, 0),
            self._add_rna_segment(orf, 1),
            self._add_rna_segment(down, 2),
        ], dim=1)
        rna_mask = torch.cat([up_mask, orf_mask, down_mask], dim=1)

        if self.use_self_encoder:
            esm = self.esm_self_encoder(esm, src_key_padding_mask=esm_mask)
            rna = self.rna_self_encoder(rna, src_key_padding_mask=rna_mask)

        for layer in self.cross_layers:
            esm, rna = layer(esm, esm_mask, rna, rna_mask)

        esm = self.esm_out_norm(esm).masked_fill(esm_mask.unsqueeze(-1), 0.0)
        rna = self.rna_out_norm(rna).masked_fill(rna_mask.unsqueeze(-1), 0.0)
        esm_pool = torch.cat([masked_mean_pool(esm, esm_mask), masked_max_pool(esm, esm_mask)], dim=1)
        rna_pool = torch.cat([masked_mean_pool(rna, rna_mask), masked_max_pool(rna, rna_mask)], dim=1)
        pooled = torch.cat([esm_pool, rna_pool], dim=1)
        return self.classifier(pooled).squeeze(1)
