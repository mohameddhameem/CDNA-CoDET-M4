import torch
import torch.nn as nn
from models.CodeCLIP import Pretrain


class Downstream(nn.Module):
    def __init__(self, pretrained_model,args):
        super(Downstream, self).__init__()
        embed_dim=args.hidden_dim # 768
        num_classes=args.num_classes #2, 6
        hidden_dim=args.output_dim # 384

        self.graph_encoder = pretrained_model.graph_encoder
        self.text_encoder = pretrained_model.text_encoder

        # Freeze pretrained encoders
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.expert_graph = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.expert_text = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        hetero_batch = batch
        text_input = batch.code
        with torch.no_grad():
            feat_g = self.graph_encoder(
                hetero_batch.x_dict, hetero_batch.edge_index_dict, hetero_batch.batch_dict
            )
            feat_t = self.text_encoder(text_input)

        # Experts refine each modality
        exp_out_g = self.expert_graph(feat_g)
        exp_out_t = self.expert_text(feat_t)

        # Fuse by router weights
        fused_feat = (0.5 * exp_out_g) + (0.5 * exp_out_t)
        logits = self.classifier(fused_feat)

        return logits
