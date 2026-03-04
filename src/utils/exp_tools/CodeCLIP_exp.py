import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HeteroGraphEncoder, TextEncoder

# Pretraining: contrastive + covariance decorrelation
class Pretrain(nn.Module):
    def __init__(self,args):
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
        # text_input: [str * batch_size]
        graph_features = self.graph_encoder(
            hetero_batch.x_dict,
            hetero_batch.edge_index_dict,
            hetero_batch.batch_dict,
        )
        text_features = self.text_encoder(text_input)
        return graph_features, text_features

    def spectral_loss(self, features):
        z = F.normalize(features, dim=1)
        C = z.T @ z / z.shape[0]
        C = C + self.eps * torch.eye(C.size(0), device=C.device)
        eigvals = torch.linalg.eigvalsh(C)
        target = eigvals.mean()
        loss = ((eigvals - target) ** 2).mean()
        return loss

    def loss(self, graph_features, text_features):
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

        # Controlled cross-covariance (preserve some difference)
        g_centered = graph_features - graph_features.mean(dim=0)
        t_centered = text_features - text_features.mean(dim=0)
        cross_cov = (g_centered.T @ t_centered) / (batch_size - 1 + self.eps)
        cross_loss = (cross_cov.pow(2).mean())

        # Adaptive weight
        lambda_cross = self.cross_lambda_scale / (contrastive_loss.detach() + self.eps)

        total_loss = contrastive_loss + self.penalty_weight * spectral_reg + lambda_cross * cross_loss

        return total_loss, contrastive_loss, spectral_reg, cross_loss

    def forward(self, batch):
        hetero_batch = batch
        text_input = batch.code
        graph_features, text_features = self.embed_forward(hetero_batch, text_input)
        total_loss, _, _, _ = self.loss(graph_features, text_features)
        return total_loss


# Downstream: MoE fusion on frozen encoders
class Downstream(nn.Module):
    def __init__(self, pretrained_model,args):
        super(Downstream, self).__init__()
        embed_dim=args.hidden_dim # 768
        num_classes=args.num_classes #2, 6
        hidden_dim=args.output_dim # 384
        self.min_weight = 0.01

        self.graph_encoder = pretrained_model.graph_encoder
        self.text_encoder = pretrained_model.text_encoder

        # Freeze pretrained encoders
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Router on concat 768g+768t
        self.router = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

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

        # Concatenate for router
        combined_feat = torch.cat([feat_g, feat_t], dim=-1)
        raw_weights = F.softmax(self.router(combined_feat), dim=-1)

        routing_weights = self.min_weight + \
            (1 - 2 * self.min_weight) * raw_weights
        print(f"\n Routing weights: graph={routing_weights.mean(dim=0)[0]}, text={routing_weights.mean(dim=0)[1]}")  # Debug: print average routing weights

        # Experts refine each modality
        exp_out_g = self.expert_graph(feat_g)
        exp_out_t = self.expert_text(feat_t)

        # Fuse by router weights
        fused_feat = (routing_weights[:, 0:1] * exp_out_g) + (routing_weights[:, 1:2] * exp_out_t)
        logits = self.classifier(fused_feat)

        return logits