#!/usr/bin/env python3
"""Find a merged sensor-streaming d_model that best matches canonical parameter count."""

from __future__ import annotations

import argparse
import json

import torch.nn as nn

from src.models import SensoryPerceiver
from src.runtime import resolve_modality_setup


def total_params(model: nn.Module, n_classes: int, d_model: int) -> int:
    classifier = nn.Linear(d_model, n_classes)
    return sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in classifier.parameters()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-d-model", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-d-model", type=int, default=256)
    parser.add_argument("--max-d-model", type=int, default=1536)
    parser.add_argument("--step", type=int, default=32)
    args = parser.parse_args()

    canonical_mods, n_classes = resolve_modality_setup("sensor_streaming", "canonical")
    merged_mods, _ = resolve_modality_setup("sensor_streaming", "merged")

    canonical_model = SensoryPerceiver(
        modality_configs=canonical_mods,
        dim=args.canonical_d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        depth=args.depth,
        dropout=args.dropout,
    )
    canonical_params = total_params(canonical_model, n_classes, args.canonical_d_model)

    best = None
    for d_model in range(args.min_d_model, args.max_d_model + 1, args.step):
        if d_model % args.num_heads != 0:
            continue
        merged_model = SensoryPerceiver(
            modality_configs=merged_mods,
            dim=d_model,
            num_heads=args.num_heads,
            mlp_dim=max(d_model * 2, args.num_heads * 32),
            depth=args.depth,
            dropout=args.dropout,
        )
        merged_params = total_params(merged_model, n_classes, d_model)
        gap = abs(merged_params - canonical_params)
        candidate = {
            "merged_d_model": d_model,
            "merged_params": merged_params,
            "canonical_params": canonical_params,
            "param_gap": gap,
        }
        if best is None or candidate["param_gap"] < best["param_gap"]:
            best = candidate

    print(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
