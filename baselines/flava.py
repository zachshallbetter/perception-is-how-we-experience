"""
FLAVA (Foundational Language And Vision Alignment) Baseline Wrapper

Provides a wrapper around Meta's FLAVA (Singh et al., 2022) using the
Hugging Face Transformers library, representing a foundational unimodal
and multimodal alignment baseline.

References:
- Paper: "FLAVA: A Foundational Language And Vision Alignment Model" (Singh et al., 2022)
- Hugging Face: https://huggingface.co/docs/transformers/model_doc/flava
"""

import torch
import torch.nn as nn

try:
    from transformers import FlavaModel, FlavaConfig
except ImportError:
    FlavaModel, FlavaConfig = None, None


class FLAVABaseline(nn.Module):
    def __init__(self, num_classes=100, use_pretrained=False):
        super().__init__()
        self.name = "FLAVA (Singh et al., 2022)"

        if FlavaModel is None:
            raise ImportError("Please install transformers: `pip install transformers`")

        if use_pretrained:
            self.flava = FlavaModel.from_pretrained("facebook/flava-full")
        else:
            config = FlavaConfig()
            self.flava = FlavaModel(config)

        hidden_size = self.flava.config.multimodal_config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids=None, pixel_values=None):
        outputs = self.flava(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )

        if input_ids is None and pixel_values is not None:
            # Vision-only mode
            pooled_output = outputs.image_embeddings[:, 0, :]  # CLS token
        else:
            # Multimodal mode
            pooled_output = outputs.multimodal_embeddings[:, 0, :]

        return self.classifier(pooled_output)
