"""
ViLT (Vision-and-Language Transformer) Baseline Wrapper

This provides a wrapper around ViLT (Kim et al., 2021) using the
Hugging Face Transformers library, representing a state-of-the-art vision+language baseline.

References:
- Paper: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision" (Kim et al., 2021)
- Hugging Face: https://huggingface.co/docs/transformers/model_doc/vilt
"""

import torch
import torch.nn as nn

try:
    from transformers import ViltModel, ViltConfig
except ImportError:
    ViltModel, ViltConfig = None, None


class ViLTBaseline(nn.Module):
    def __init__(self, num_classes=100, use_pretrained=False):
        super().__init__()
        self.name = "ViLT (Kim et al., 2021)"

        if ViltModel is None:
            raise ImportError("Please install transformers: `pip install transformers`")

        if use_pretrained:
            self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        else:
            config = ViltConfig()
            self.vilt = ViltModel(config)

        hidden_size = self.vilt.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids=None, pixel_values=None):
        # ViLT strictly expects paired text (input_ids) and image (pixel_values)
        if input_ids is None and pixel_values is not None:
            # Generate dummy text token (e.g., [CLS])
            device = pixel_values.device
            input_ids = torch.tensor([[101, 102]], device=device).repeat(
                pixel_values.size(0), 1
            )

        outputs = self.vilt(input_ids=input_ids, pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # (batch, hidden_size)
        return self.classifier(pooled_output)
