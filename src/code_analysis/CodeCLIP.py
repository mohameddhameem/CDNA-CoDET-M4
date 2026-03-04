import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=256, num_layers=2):
        super(GraphEncoder, self).__init__()
        
        # Projection for inputs to match hidden_dim
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Linear(in_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # MLP for GINEConv
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        # Learnable parameter for root node importance
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1. Project inputs
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)
        
        # 2. Apply Graph Convolutions
        for conv in self.convs:
            x = conv(h, edge_index, e) + h
            h = torch.relu(x)
            
        # 3. Aggregation
        # Mean pooling of all nodes
        h_mean = global_mean_pool(h, batch)
        
        # Root node embedding (node 0 of each graph)
        # Using to_dense_batch to extract the first node (index 0) of each graph in batch
        # x_dense: [batch_size, max_nodes, feature_dim]
        h_dense, _ = to_dense_batch(h, batch)
        h_root = h_dense[:, 0, :] 
        
        # 4. Final Combination: mean + alpha * root
        out = h_mean + self.alpha * h_root
        
        return self.out_proj(out)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=256, num_heads=4, num_layers=2, max_len=512):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Simple learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # Embed
        x_emb = self.embedding(x)
        
        # Add Positional Encoding
        seq_len = x.shape[1]
        x_emb = x_emb + self.pos_encoder[:, :seq_len, :]
        
        # Transform
        x_out = self.transformer(x_emb)
        
        # Mean Pooling
        x_pool = x_out.mean(dim=1)
        
        return x_pool 

class Pretrain(nn.Module):
    def __init__(self, graph_in_dim=384, text_vocab_size=30522, embed_dim=256):
        super(Pretrain, self).__init__()
        
        self.graph_encoder = GraphEncoder(in_dim=graph_in_dim, hidden_dim=embed_dim, out_dim=embed_dim)
        self.text_encoder = TextEncoder(vocab_size=text_vocab_size, embed_dim=embed_dim)
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(self, graph_batch, text_input):
        # graph_batch: PyG Batch object
        # text_input: [batch_size, seq_len] tensor
        
        graph_features = self.graph_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch)
        text_features = self.text_encoder(text_input)
        
        # Normalize features
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_graph = logit_scale * graph_features @ text_features.t()
        logits_per_text = logits_per_graph.t()
        
        return logits_per_graph, logits_per_text
        
    def loss(self, logits_per_graph, logits_per_text):
        # Contrastive loss
        batch_size = logits_per_graph.shape[0]
        labels = torch.arange(batch_size, device=logits_per_graph.device)
        
        loss_graph = nn.functional.cross_entropy(logits_per_graph, labels)
        loss_text = nn.functional.cross_entropy(logits_per_text, labels)
        
        return (loss_graph + loss_text) / 2


class Downstream(nn.Module):
    def __init__(self, pretrained_model, embed_dim=256, num_classes=2, hidden_dim=128):
        super(Downstream, self).__init__()
        self.graph_encoder = pretrained_model.graph_encoder
        
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, graph_batch):
        # Get graph embeddings from the encoder
        # Note: GraphEncoder.forward signature is forward(self, x, edge_index, edge_attr, batch=None)
        features = self.graph_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch)
        
        # Predict
        logits = self.classifier(features)
        
        return logits