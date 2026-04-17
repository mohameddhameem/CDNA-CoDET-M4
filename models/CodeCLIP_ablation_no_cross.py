"""No-cross-loss ablation model.

Notes
-----
Disables cross-modal penalty terms during pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HeteroGraphEncoder, TextEncoder
from models.CodeCLIP_ablation_no_router import Downstream

# Pretraining: contrastive + covariance decorrelation
class Pretrain(nn.Module):
    """Pretrain without cross-covariance penalty between modalities."""

    def __init__(self,args):
        """Initialize encoders and contrastive loss settings."""
        super(Pretrain, self).__init__()
        hetero_metadata=args.metadata
        graph_in_dim=args.input_dim # 768
        embed_dim=args.hidden_dim # 768
        num_layers=args.num_layers # 3
        penalty_weight=1
        text_model_name="microsoft/codebert-base"
        text_max_length=512
        cross_lambda_scale=0.5

        self.graph_encoder = HeteroGraphEncoder(
            metadata=hetero_metadata,
            in_dim=graph_in_dim,
            hidden_dim=embed_dim,
            out_dim=embed_dim,
            num_layers=num_layers,
        )

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            max_length=text_max_length,
            freeze=True,
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

        self.penalty_weight = penalty_weight
        self.cross_lambda_scale = cross_lambda_scale
        self.eps = 1e-6

    def embed_forward(self, hetero_batch, text_input):
        """Encode graph and text inputs for contrastive learning."""
        # text_input: [str * batch_size]
        graph_features = self.graph_encoder(
            hetero_batch.x_dict,
            hetero_batch.edge_index_dict,
            hetero_batch.batch_dict,
        )
        text_features = self.text_encoder(text_input)
        return graph_features, text_features

    def spectral_loss(self, features):
        """Compute isotropy regularization from feature covariance spectrum."""
        z = F.normalize(features, dim=1)
        C = z.T @ z / z.shape[0]
        C = C + self.eps * torch.eye(C.size(0), device=C.device)
        eigvals = torch.linalg.eigvalsh(C)
        target = eigvals.mean()
        loss = ((eigvals - target) ** 2).mean()
        return loss

    def loss(self, graph_features, text_features):
        """Compute contrastive loss plus graph spectral regularization."""
        batch_size = graph_features.shape[0]
        device = graph_features.device

        labels = torch.arange(batch_size, device=device)

        # InfoNCE alignment
        g_norm = F.normalize(graph_features, dim=1)
        t_norm = F.normalize(text_features, dim=1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits_per_graph = logit_scale * g_norm @ t_norm.t()
        logits_per_text = logits_per_graph.t()

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        contrastive_loss = (loss_graph + loss_text) / 2

        # Spectral / isotropic regularization (graph only)
        spectral_reg = self.spectral_loss(graph_features)

        total_loss = contrastive_loss + self.penalty_weight * spectral_reg

        return total_loss, contrastive_loss, spectral_reg, None

    def forward(self, batch):
        """Run a pretraining step and return the total objective."""
        hetero_batch = batch
        text_input = batch.code
        graph_features, text_features = self.embed_forward(hetero_batch, text_input)
        total_loss, _, _, _ = self.loss(graph_features, text_features)
        return total_loss
