"""UniXcoder baseline model.

Notes
-----
Defines text-only downstream classification with UniXcoder features.
"""

import torch
import torch.nn as nn
from layers import TextEncoderCLS as TextEncoder


class Pretrain(nn.Module):
    """Placeholder for supervised-only setting."""

    def __init__(self, *args, **kwargs):
        """Initialize placeholder pretraining module."""
        super().__init__()

    def forward(self, *args, **kwargs):
        """Raise because this variant does not support pretraining."""
        raise NotImplementedError("Text-only ablation uses Downstream directly for supervised tasks.")


class Downstream(nn.Module):
    """Run downstream classification using frozen UniXcoder features."""

    def __init__(self, pretrained_model,args):
        """Initialize UniXcoder encoder, expert head, and classifier."""
        super(Downstream, self).__init__()
        embed_dim=args.hidden_dim # 768
        num_classes=args.num_classes #2, 6
        hidden_dim=args.output_dim # 384
        text_model_name="microsoft/unixcoder-base"
        text_max_length=1024

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            max_length=text_max_length,
            freeze=True,
        )

        self.expert_text = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        """Encode input code and return downstream logits."""
        text_input = batch.code
        feat_t = self.text_encoder(text_input)

        # Experts refine each modality
        exp_out_t = self.expert_text(feat_t)

        # Fuse by router weights
        logits = self.classifier(exp_out_t)

        return logits