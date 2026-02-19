"""
Perceiver IO Baseline Implementation for Comparison.

Minimal implementation based on Jaegle et al. "Perceiver IO: A General
Architecture for Structured Inputs & Outputs" (2022).

This is NOT the full DeepMind implementation but a simplified version
for fair comparison with SensoryPerceiver on the same datasets and
experimental protocol.

Reference:
  Jaegle et al. (2022) Perceiver IO: A General Architecture for
  Structured Inputs & Outputs. ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PerceiverIOBaseline(nn.Module):
    """
    Simplified Perceiver IO for fair comparison.

    Key differences from full DeepMind version:
    - Smaller latent array (256 vs thousands)
    - Simpler cross-attention (single layer vs multi-layer)
    - No latent processor (to keep parameters comparable)
    - Fixed positional encodings instead of learned
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 100,
        num_latents: int = 256,
        dim_latent: int = 512,
        num_heads: int = 8,
        depth: int = 6,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimensionality of input features
            output_dim: Number of output classes
            num_latents: Size of latent array (the bottleneck)
            dim_latent: Dimension of latent embeddings
            num_heads: Number of attention heads
            depth: Number of self-attention transformer layers
            mlp_dim: Hidden dimension of feedforward networks
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.dim_latent = dim_latent

        # Input projection
        self.input_projection = nn.Linear(input_dim, dim_latent)

        # Learned latent array (the bottleneck!)
        # This is what makes Perceiver IO unique: everything compresses here
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim_latent))
        nn.init.normal_(self.latents, std=0.02)

        # Cross-attention layer: input -> latents
        # Compresses variable-length input into fixed latent array
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim_latent,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention transformer in latent space
        # This is where "thinking" happens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_latent,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        # Output projection
        self.output_projection = nn.Linear(dim_latent, output_dim)

        # LayerNorm for stability
        self.norm = nn.LayerNorm(dim_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass following Perceiver IO protocol.

        Args:
            x: Input tensor of shape:
               - (batch, channels, height, width) for images
               - (batch, seq_len, features) for sequences
               - (batch, features) for flat features

        Returns:
            output: (batch, output_dim)
        """
        batch_size = x.shape[0]

        # Step 1: Handle different input shapes
        if x.dim() == 4:
            # Images: (batch, channels, height, width) -> (batch, seq_len, features)
            batch, channels, height, width = x.shape
            x = x.flatten(2).permute(0, 2, 1)  # (batch, h*w, channels)
        elif x.dim() == 2:
            # Flat features: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)

        # Step 2: Project input to latent dimension
        x = self.input_projection(x)  # (batch, seq_len, dim_latent)

        # Step 3: Cross-attention - compress input into latent array
        # Query: latents, Key/Value: input
        latents = self.latents.expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attention(
            query=latents,
            key=x,
            value=x,
            need_weights=False,
        )
        latents = latents + attn_out  # Residual connection

        # Step 4: Self-attention in latent space (the "thinking")
        # Process through transformer
        latents = self.norm(latents)
        latents = self.transformer(latents)

        # Step 5: Pool latents to single vector
        # Options: mean, max, or use CLS token
        latents_pooled = latents.mean(dim=1)  # (batch, dim_latent)

        # Step 6: Project to output dimension
        output = self.output_projection(latents_pooled)

        return output

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PerceiverIOLargant(nn.Module):
    """
    Larger variant of Perceiver IO for capacity comparison.

    Used in ablation studies to understand if performance gains
    come from architecture or just model size.
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 100,
        num_latents: int = 512,  # 2x larger
        dim_latent: int = 768,  # 1.5x larger
        num_heads: int = 12,  # More heads
        depth: int = 8,  # More layers
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, dim_latent)

        self.latents = nn.Parameter(torch.randn(1, num_latents, dim_latent))
        nn.init.normal_(self.latents, std=0.02)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim_latent,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_latent,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.output_projection = nn.Linear(dim_latent, output_dim)
        self.norm = nn.LayerNorm(dim_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if x.dim() == 4:
            batch, channels, height, width = x.shape
            x = x.flatten(2).permute(0, 2, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_projection(x)

        latents = self.latents.expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attention(
            query=latents,
            key=x,
            value=x,
            need_weights=False,
        )
        latents = latents + attn_out

        latents = self.norm(latents)
        latents = self.transformer(latents)

        latents_pooled = latents.mean(dim=1)
        output = self.output_projection(latents_pooled)

        return output

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    model = PerceiverIOBaseline(
        input_dim=3,
        output_dim=100,
        num_latents=256,
        dim_latent=512,
        num_heads=8,
        depth=6,
    )

    x = torch.randn(32, 3, 32, 32)  # CIFAR-100 batch
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.get_num_params():,}")

    # Compare with larger variant
    model_large = PerceiverIOLargant(
        input_dim=3,
        output_dim=100,
        num_latents=512,
        dim_latent=768,
        num_heads=12,
        depth=8,
    )
    print(f"\nLarge variant parameters: {model_large.get_num_params():,}")
