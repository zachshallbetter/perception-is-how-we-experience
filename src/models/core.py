"""
Sensory Perceiver: Unified multimodal neural architecture with attention-based integration
and contrastive replay mechanism for continual learning.

Core components:
- PerceiverTower: Modality-specific feature extraction (Conv2d/Linear → MLP → MHA)
- SensoryPerceiver: Cross-modal attention integration + replay buffer
- ContrastiveLoss: NT-Xent loss for replay mechanism
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from src.runtime.errors import AppError, ErrorCode


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

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
               - If use_conv: (batch, channels, height, width)
               - Otherwise: (batch, seq_len, features)

        Returns:
            (seq_len, batch, dim) for attention integration
        """
        # B: Convolutional or linear projection
        key_padding_mask = None

        if self.use_conv:
            x = self.conv(x)  # (batch, dim, H', W')
            batch, dim, h, w = x.shape
            # Flatten spatial dims for attention: (batch, dim, H'*W')
            x = x.flatten(2)  # (batch, dim, H'*W')
            x = x.permute(2, 0, 1)  # (H'*W', batch, dim)
        else:
            if x.dim() == 3:
                # (batch, seq_len, features) -> (seq_len, batch, dim)
                batch, seq_len, _ = x.shape
                if valid_mask is None:
                    valid_mask = torch.ones(
                        (batch, seq_len), dtype=torch.bool, device=x.device
                    )
                elif valid_mask.shape != (batch, seq_len):
                    raise AppError(
                        ErrorCode.INVALID_ARGUMENT,
                        "PerceiverTower valid_mask must match non-conv batch/seq shape",
                        context={
                            "expected_mask_shape": (batch, seq_len),
                            "actual_mask_shape": tuple(valid_mask.shape),
                        },
                    )
                x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
                x = x.reshape(batch * seq_len, self.input_dim)
                x = self.conv(x)
                x = x.view(batch, seq_len, self.dim).permute(1, 0, 2)
                key_padding_mask = ~valid_mask
            elif x.dim() == 2:
                # (batch, features) -> (1, batch, dim)
                x = self.conv(x)
                x = x.unsqueeze(0)
                batch = x.shape[1]
                key_padding_mask = torch.zeros(
                    (batch, 1), dtype=torch.bool, device=x.device
                )
            else:
                raise AppError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"Expected non-conv input with 2 or 3 dims, got {x.dim()}",
                )

        seq_len, batch, _ = x.shape

        # D: Self-attention with residual
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # C: MLP with residual
        mlp_out = self.mlp(x)
        x = x + self.fc(mlp_out)

        # F: Layer normalization
        x = self.norm(x)

        if key_padding_mask is not None:
            valid_positions = (
                (~key_padding_mask).transpose(0, 1).unsqueeze(-1).to(dtype=x.dtype)
            )
            x = x * valid_positions

        return x  # (seq_len, batch, dim)


class TaskAdapter(nn.Module):
    """Lightweight task-specific adapter module for parameter isolation.

    Sits between cross-modal attention and output projection.
    Enables parameter isolation: freeze after each task, allocate new adapter for next task.
    """

    def __init__(self, dim: int = 768, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter transformation with residual connection."""
        return x + self.adapter(x)

    def freeze(self) -> None:
        """Freeze all parameters in this adapter (called after task completes)."""
        for param in self.parameters():
            param.requires_grad = False


class SensoryPerceiver(nn.Module):
    """Unified multimodal Perceiver with MIME-Type routing, contrastive replay, and parameter isolation.

    Architecture:
    1. MIME-Type router directs inputs to modality-specific towers (task-agnostic)
    2. Towers extract embeddings (seq_len, batch, dim)
    3. Cross-modal attention integrates all modalities
    4. Task-specific adapter applies parameter isolation (frozen per task)
    5. Contrastive replay buffer stores/updates tokens
    6. Output: Unified representation for downstream tasks
    """

    supports_live_valid_mask = True

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
        enable_parameter_isolation: bool = True,
        adapter_hidden_dim: int = 512,
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
            enable_parameter_isolation: If True, use task-specific adapters for parameter isolation
            adapter_hidden_dim: Hidden dimension for task adapter layers
        """
        super().__init__()
        self.dim = dim
        self.contrastive_temp = contrastive_temp
        self.num_modalities = len(modality_configs)
        self.enable_parameter_isolation = enable_parameter_isolation
        self.current_task_id = 0

        # MIME-Type routed towers (task-agnostic)
        self.towers = nn.ModuleDict()
        for modality, config in modality_configs.items():
            self.towers[modality] = PerceiverTower(
                input_dim=config["input_dim"],
                dim=dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                depth=depth,
                dropout=dropout,
                use_conv=config.get("use_conv", False),
            )

        # Cross-modal integration (global attention)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # Task-specific adapters (parameter isolation)
        if enable_parameter_isolation:
            self.current_adapter = TaskAdapter(
                dim=dim, hidden_dim=adapter_hidden_dim, dropout=dropout
            )
            self.task_adapters = nn.ModuleDict({"task_0": self.current_adapter})
        else:
            self.current_adapter = None
            self.task_adapters = None

        # Output projection (shared)
        self.output_proj = nn.Linear(dim, dim)
        self.final_norm = nn.LayerNorm(dim)

        # Replay buffer (stores RAW inputs and labels)
        self.replay_buffer = {}  # mod_name -> tensor (buffer_size, ...)
        self.register_buffer(
            "replay_labels", torch.zeros(replay_buffer_size, dtype=torch.long)
        )
        self.register_buffer("replay_count", torch.tensor(0, dtype=torch.long))
        self.replay_buffer_size = replay_buffer_size
        self.modality_configs = modality_configs

        # Initialize modality-specific raw input buffers
        # Buffers will be created dynamically in _update_replay_buffer on first call.

    def forward(
        self,
        inputs: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
        update_buffer: bool = True,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            inputs: Dict[modality_name] → tensor
            labels: Ground truth labels (needed for populating replay buffer)
            update_buffer: If True, update the replay buffer with current batch
            return_embeddings: If True, return per-modality embeddings

        Returns:
            (integrated_output, embeddings_dict or None)
            integrated_output: (batch, seq_len, dim) unified representation
        """
        embeddings = {}
        all_embeddings = []
        all_key_padding_masks = []
        replay_inputs: Dict[str, torch.Tensor] = {}

        # 1. Route through towers
        for modality, payload in inputs.items():
            if modality in self.towers:
                tensor, valid_mask = self._normalize_model_input(modality, payload)
                emb = self.towers[modality](
                    tensor, valid_mask=valid_mask
                )  # (seq_len, batch, dim)
                embeddings[modality] = emb
                all_embeddings.append(emb)
                replay_inputs[modality] = tensor

                if valid_mask is None:
                    seq_len = emb.shape[0]
                    batch = emb.shape[1]
                    key_padding = torch.zeros(
                        (batch, seq_len), dtype=torch.bool, device=emb.device
                    )
                else:
                    key_padding = ~valid_mask
                all_key_padding_masks.append(key_padding)

        # 2. Concatenate all modality embeddings
        combined = torch.cat(all_embeddings, dim=0)  # (total_seq, batch, dim)
        combined_key_padding_mask = (
            torch.cat(all_key_padding_masks, dim=1) if all_key_padding_masks else None
        )

        # 3. Cross-modal attention
        attn_out, _ = self.global_attn(
            combined,
            combined,
            combined,
            key_padding_mask=combined_key_padding_mask,
        )
        integrated = combined + attn_out

        # 3b. Task-specific adapter (parameter isolation)
        if self.enable_parameter_isolation and self.current_adapter is not None:
            integrated = self.current_adapter(integrated)

        # 4. Output projection + normalization (shared)
        integrated = self.output_proj(integrated)
        integrated = self.final_norm(integrated)

        if combined_key_padding_mask is not None:
            valid_positions = (
                (~combined_key_padding_mask)
                .transpose(0, 1)
                .unsqueeze(-1)
                .to(dtype=integrated.dtype)
            )
            integrated = integrated * valid_positions

        # 5. Update replay buffer with raw inputs and labels
        if update_buffer and labels is not None:
            self._update_replay_buffer(replay_inputs, labels.detach())

        # Convert to (batch, seq_len, dim) for downstream
        integrated = integrated.permute(1, 0, 2)

        if return_embeddings:
            return integrated, embeddings
        return integrated, None

    def _normalize_model_input(
        self,
        modality: str,
        payload: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if torch.is_tensor(payload):
            tensor = payload
            if tensor.dim() == 3 and not self.towers[modality].use_conv:
                batch, seq_len, _ = tensor.shape
                valid_mask = torch.ones(
                    (batch, seq_len), dtype=torch.bool, device=tensor.device
                )
            else:
                valid_mask = None
            return tensor, valid_mask

        if isinstance(payload, dict):
            if "tensor" not in payload:
                raise AppError(
                    ErrorCode.INVALID_ARGUMENT,
                    "Structured modality payload is missing tensor",
                    context={"modality": modality},
                )
            tensor = payload["tensor"]
            valid_mask = payload.get("valid_mask")
            if valid_mask is not None and not torch.is_tensor(valid_mask):
                raise AppError(
                    ErrorCode.INVALID_ARGUMENT,
                    "Structured modality valid_mask must be a tensor",
                    context={"modality": modality},
                )
            return tensor, valid_mask

        raise AppError(
            ErrorCode.INVALID_ARGUMENT,
            "Unsupported modality payload type",
            context={"modality": modality, "payload_type": type(payload).__name__},
        )

    def _update_replay_buffer(
        self, inputs: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> None:
        """Update replay buffer using Reservoir Sampling to ensure long-term retention."""
        batch_size = labels.size(0)

        for i in range(batch_size):
            count = self.replay_count.item()
            if count < self.replay_buffer_size:
                # Buffer not full, just append
                idx = count
            else:
                # Buffer full, replace with probability buffer_size / total_seen
                idx = torch.randint(0, count + 1, (1,)).item()
                if idx >= self.replay_buffer_size:
                    idx = -1  # Do not keep

            if idx != -1:
                # Update all modality buffers
                for name, tensor in inputs.items():
                    # Ensure buffer exists
                    if name not in self.replay_buffer:
                        shape = [self.replay_buffer_size] + list(tensor.shape[1:])
                        self.register_buffer(
                            f"replay_buf_{name}",
                            torch.zeros(*shape, device=tensor.device),
                        )
                        self.replay_buffer[name] = getattr(self, f"replay_buf_{name}")

                    self.replay_buffer[name][idx] = tensor[i].detach()

                # Update labels
                self.replay_labels[idx] = labels[i].detach()

            self.replay_count += 1

    def get_replay_samples(
        self, n_samples: int
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """Sample from replay buffer for continual learning replay."""
        count = min(self.replay_count.item(), self.replay_buffer_size)
        if count == 0:
            return None, None
        n_samples = min(n_samples, count)
        indices = torch.randperm(count, device=self.replay_labels.device)[:n_samples]

        samples = {}
        for name, buf in self.replay_buffer.items():
            samples[name] = buf[indices]

        return samples, self.replay_labels[indices]

    def freeze_task(self) -> None:
        """Freeze current task's adapter and prepare for next task (parameter isolation).

        Called at the end of task T training to prevent further updates to task T's parameters.
        Allocates a new adapter for task T+1 with trainable parameters.
        """
        if not self.enable_parameter_isolation or self.current_adapter is None:
            return

        # Freeze current adapter
        old_adapter = self.current_adapter
        old_adapter.freeze()

        # Allocate new adapter for next task
        self.current_task_id += 1
        task_name = f"task_{self.current_task_id}"
        new_adapter = TaskAdapter(
            dim=self.dim,
            hidden_dim=old_adapter.adapter[0].out_features,  # Match hidden dim
            dropout=(
                old_adapter.adapter[2].p
                if isinstance(old_adapter.adapter[2], nn.Dropout)
                else 0.1
            ),
        ).to(next(self.parameters()).device)
        self.current_adapter = new_adapter
        self.task_adapters[task_name] = new_adapter

    def get_trainable_parameters(self) -> int:
        """Count trainable parameters (excluding frozen task adapters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_task_coverage(self) -> float:
        """Compute task parameter coverage: fraction of original capacity used.

        After K tasks with parameter isolation, coverage indicates how much of the
        original network capacity remains trainable.
        """
        if self.current_task_id == 0:
            return 1.0

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_parameters()
        return trainable_params / total_params if total_params > 0 else 0.0


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
        replay: torch.Tensor,  # (batch, dim)
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
