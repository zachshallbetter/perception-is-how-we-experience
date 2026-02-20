"""
CLIP (Contrastive Language-Image Pretraining) Baseline Wrapper

Provides a wrapper around OpenAI's CLIP (Radford et al., 2021) using the
Hugging Face Transformers library, representing a foundational contrastive
vision-language baseline.

References:
- Paper: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- Hugging Face: https://huggingface.co/docs/transformers/model_doc/clip
"""

import torch
import torch.nn as nn

try:
    from transformers import CLIPModel, CLIPConfig
except ImportError:
    CLIPModel, CLIPConfig = None, None


class CLIPBaseline(nn.Module):
    def __init__(self, use_pretrained=False):
        super().__init__()
        self.name = "CLIP (Radford et al., 2021)"

        if CLIPModel is None:
            raise ImportError("Please install transformers: `pip install transformers`")

        if use_pretrained:
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            config = CLIPConfig()
            self.clip = CLIPModel(config)

        self.classifier = None
        if num_classes is not None:
            hidden_size = self.clip.config.projection_dim
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids=None, pixel_values=None):
        outputs = self.clip(input_ids=input_ids, pixel_values=pixel_values)

        # Determine if we are running in vision-only classification mode
        if (
            self.classifier is not None
            and pixel_values is not None
            and input_ids is None
        ):
            return self.classifier(outputs.image_embeds)

        return {
            "logits_per_image": getattr(outputs, "logits_per_image", None),
            "logits_per_text": getattr(outputs, "logits_per_text", None),
            "text_embeds": getattr(outputs, "text_embeds", None),
            "image_embeds": getattr(outputs, "image_embeds", None),
        }
