import torch
import torch.nn as nn
from .GraphEncoder import HeteroGraphEncoder


class Pretrain(nn.Module):
    """Placeholder for supervised-only setting."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Graph-only ablation uses Downstream directly for supervised tasks.")


class Downstream(nn.Module):
    def __init__(self, pretrained_model, args):
        super(Downstream, self).__init__()
        graph_in_dim=args.input_dim # 768
        embed_dim=args.hidden_dim # 768
        num_classes=args.num_classes #2, 6
        hidden_dim=args.output_dim # 384
        hetero_metadata=args.metadata
        num_layers=args.num_layers # 3
        self.graph_encoder = HeteroGraphEncoder(
            metadata=hetero_metadata,
            in_dim=graph_in_dim,
            hidden_dim=embed_dim,
            out_dim=embed_dim,
            num_layers=num_layers, # 3
        )
        self.expert_graph = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        hetero_batch = batch
        feat_g = self.graph_encoder(
            hetero_batch.x_dict, hetero_batch.edge_index_dict, hetero_batch.batch_dict
        )

        # Experts refine each modality
        exp_out_g = self.expert_graph(feat_g)

        # Fuse by router weights
        logits = self.classifier(exp_out_g)

        return logits
