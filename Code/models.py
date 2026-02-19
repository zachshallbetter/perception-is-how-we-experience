"""
Sensory Perceiver: Unified multimodal neural architecture with attention-based integration
and contrastive replay mechanism for continual learning.

Core components:
- PerceiverTower: Modality-specific feature extraction (Conv2d/Linear → MLP → MHA)
- SensoryPerceiver: Cross-modal attention integration + replay buffer
- ContrastiveLoss: NT-Xent loss for replay mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from typing import Dict, Tuple, Optional


class PerceiverTower(nn.Module):
    """Single modality-specific processing tower.

    Flow: [input] → Conv2d (spatial) → Flatten → Linear (mlp_in)
          → ReLU → Dropout → Linear (mlp_out) → MHA → FC → LayerNorm
    """

    def __init__(
        self,
        input_dim: int,
        dim: int = 768,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        depth: int = 6,
        dropout: float = 0.1,
        use_conv: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.input_dim = input_dim
        self.use_conv = use_conv

        # B: Convolutional feature extraction (for image-like inputs)
        if use_conv:
            self.conv = nn.Conv2d(input_dim, dim, kernel_size=3, stride=2, padding=1)
        else:
            # For non-image modalities: simple linear projection
            self.conv = nn.Linear(input_dim, dim)

        # C: Multi-Layer Perceptron (after conv)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            DropPath(dropout),
        )

        # D: Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # E: Fully Connected output layer
        self.fc = nn.Sequential(
            nn.Linear(mlp_dim, dim),
            DropPath(dropout),
        )

        # F: Layer Normalization
        self.norm = nn.LayerNorm(dim)

        # Residual connections
        self.residual_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
               - If use_conv: (batch, channels, height, width)
               - Otherwise: (batch, seq_len, features)

        Returns:
            (seq_len, batch, dim) for attention integration
        """
        # B: Convolutional or linear projection
        if self.use_conv:
            x = self.conv(x)  # (batch, dim, H', W')
            batch, dim, h, w = x.shape
            # Flatten spatial dims for attention: (batch, dim, H'*W')
            x = x.flatten(2)  # (batch, dim, H'*W')
            x = x.permute(2, 0, 1)  # (H'*W', batch, dim)
        else:
            # (batch, seq_len, features) → (seq_len, batch, dim)
            x = self.conv(x)
            x = x.permute(1, 0, 2)

        seq_len, batch, _ = x.shape

        # D: Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out

        # C: MLP with residual
        mlp_out = self.mlp(x)
        x = x + self.fc(mlp_out)

        # F: Layer normalization
        x = self.norm(x)

        return x  # (seq_len, batch, dim)


class SensoryPerceiver(nn.Module):
    """Unified multimodal Perceiver with MIME-Type routing and contrastive replay.

    Architecture:
    1. MIME-Type router directs inputs to modality-specific towers
    2. Towers extract embeddings (seq_len, batch, dim)
    3. Cross-modal attention integrates all modalities
    4. Contrastive replay buffer stores/updates tokens
    5. Output: Unified representation for downstream tasks
    """

    def __init__(
        self,
        modality_configs: Dict[str, Dict],
        dim: int = 768,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        depth: int = 6,
        dropout: float = 0.1,
        replay_buffer_size: int = 2000,
        contrastive_temp: float = 0.07,
    ):
        """
        Args:
            modality_configs: Dict mapping modality_name → {input_dim, use_conv}
                Example: {
                    'vision': {'input_dim': 3, 'use_conv': True},
                    'audio': {'input_dim': 128, 'use_conv': False},
                }
            dim: Embedding dimension
            num_heads: Attention heads
            mlp_dim: MLP hidden dimension
            depth: Number of layers per tower
            dropout: Dropout rate
            replay_buffer_size: Token buffer for contrastive learning
            contrastive_temp: Temperature for NT-Xent loss
        """
        super().__init__()
        self.dim = dim
        self.contrastive_temp = contrastive_temp
        self.num_modalities = len(modality_configs)

        # MIME-Type routed towers
        self.towers = nn.ModuleDict()
        for modality, config in modality_configs.items():
            self.towers[modality] = PerceiverTower(
                input_dim=config['input_dim'],
                dim=dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                depth=depth,
                dropout=dropout,
                use_conv=config.get('use_conv', False),
            )

        # Cross-modal integration (global attention)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.final_norm = nn.LayerNorm(dim)

        # Replay buffer (contrastive learning)
        self.register_buffer(
            'replay_buffer',
            torch.randn(replay_buffer_size, dim)
        )
        self.register_buffer(
            'replay_buffer_ptr',
            torch.tensor(0, dtype=torch.long)
        )
        self.replay_buffer_size = replay_buffer_size

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            inputs: Dict[modality_name] → tensor
            return_embeddings: If True, return per-modality embeddings

        Returns:
            (integrated_output, embeddings_dict or None)
            integrated_output: (batch, seq_len, dim) unified representation
        """
        embeddings = {}
        all_embeddings = []

        # 1. Route through towers
        for modality, tensor in inputs.items():
            if modality in self.towers:
                emb = self.towers[modality](tensor)  # (seq_len, batch, dim)
                embeddings[modality] = emb
                all_embeddings.append(emb)

        # 2. Concatenate all modality embeddings
        combined = torch.cat(all_embeddings, dim=0)  # (total_seq, batch, dim)

        # 3. Cross-modal attention
        attn_out, _ = self.global_attn(combined, combined, combined)
        integrated = combined + attn_out

        # 4. Output projection + normalization
        integrated = self.output_proj(integrated)
        integrated = self.final_norm(integrated)

        # 5. Update replay buffer with new embeddings
        self._update_replay_buffer(integrated.detach())

        # Convert to (batch, seq_len, dim) for downstream
        integrated = integrated.permute(1, 0, 2)

        if return_embeddings:
            return integrated, embeddings
        return integrated, None

    def _update_replay_buffer(self, embeddings: torch.Tensor) -> None:
        """Update replay buffer with new embeddings (circular buffer)."""
        batch_size = embeddings.size(1)
        ptr = self.replay_buffer_ptr.item()

        if ptr + batch_size <= self.replay_buffer_size:
            self.replay_buffer[ptr:ptr + batch_size] = embeddings.mean(dim=0)
            self.replay_buffer_ptr.fill_((ptr + batch_size) % self.replay_buffer_size)
        else:
            # Wrap around
            remaining = self.replay_buffer_size - ptr
            self.replay_buffer[ptr:] = embeddings.mean(dim=0)[:remaining]
            self.replay_buffer[:batch_size - remaining] = embeddings.mean(dim=0)[remaining:]
            self.replay_buffer_ptr.fill_((batch_size - remaining) % self.replay_buffer_size)

    def get_replay_samples(self, n_samples: int) -> torch.Tensor:
        """Sample from replay buffer for contrastive loss."""
        indices = torch.randint(0, self.replay_buffer_size, (n_samples,))
        return self.replay_buffer[indices]


class ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Maximizes similarity between current and replay embeddings,
    minimizes similarity with negative samples.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        current: torch.Tensor,  # (batch, dim)
        replay: torch.Tensor,   # (batch, dim)
    ) -> torch.Tensor:
        """
        Args:
            current: Current batch embeddings
            replay: Replayed batch embeddings

        Returns:
            Contrastive loss scalar
        """
        batch_size = current.size(0)

        # Normalize embeddings
        current = F.normalize(current, dim=1)
        replay = F.normalize(replay, dim=1)

        # Compute similarity matrix
        logits = torch.mm(current, replay.t()) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=current.device)

        # NT-Xent loss
        loss = F.cross_entropy(logits, labels)
        return loss
