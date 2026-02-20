"""
Perceiver Baseline Wrapper

This provides a wrapper around the original DeepMind Perceiver (Jaegle et al., 2021)
using the Hugging Face Transformers library for benchmarking against the Sensory Perceiver.

References:
- Paper: "Perceiver: General Perception with Iterative Attention" (Jaegle et al., 2021)
- Hugging Face: https://huggingface.co/docs/transformers/model_doc/perceiver
"""

import torch
import torch.nn as nn

try:
    from transformers import PerceiverModel, PerceiverConfig
except ImportError:
    PerceiverModel, PerceiverConfig = None, None


class PerceiverBaseline(nn.Module):
    def __init__(self, num_classes=100, use_pretrained=False):
        super().__init__()
        self.name = "Perceiver (Jaegle et al., 2021)"

        if PerceiverModel is None:
            raise ImportError("Please install transformers: `pip install transformers`")

        if use_pretrained:
            # Using deepmind's perceiver-io as original perceiver isn't separately hosted usually,
            # but we configure it for standard classification to mimic the original.
            self.perceiver = PerceiverModel.from_pretrained(
                "deepmind/vision-perceiver-learned"
            )
            config = self.perceiver.config
        else:
            config = PerceiverConfig()
            self.perceiver = PerceiverModel(config)

        # Add projection for raw 3-channel pixels to d_model
        self.input_proj = nn.Linear(3, config.d_model)
        # Assuming maximum sequence length of 224*224 = 50176
        self.pos_emb = nn.Parameter(torch.randn(1, 50176, config.d_model) * 0.02)

        # Add a classification head
        hidden_size = self.perceiver.config.d_latents
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        # inputs: (batch, channels, height, width) for vision
        # The transformers perceiver expects (batch, num_pixels, d_model)
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            inputs = inputs.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)

        # Project 3 channels to d_model and add 1D positional tokens
        seq_len = inputs.size(1)
        inputs = self.input_proj(inputs) + self.pos_emb[:, :seq_len, :]

        outputs = self.perceiver(inputs=inputs)
        # Sequence classification uses the mean of the latent array
        latents = outputs.last_hidden_state  # (batch, num_latents, d_latents)
        pooled = latents.mean(dim=1)

        return self.classifier(pooled)
