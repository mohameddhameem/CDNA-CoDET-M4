"""Graph encoder layers.

Notes
-----
Defines heterogeneous GNN encoders used by model variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool


class HeteroGraphEncoder(nn.Module):
    def __init__(self, metadata, in_dim=384, hidden_dim=256, out_dim=256, num_layers=2):
        """
        Args:
            metadata: Tuple of (list_of_node_types, list_of_edge_types) from PyG HeteroData.
            in_dim: Input feature dimension.
            hidden_dim: Hidden dimension for GNN layers.
            out_dim: Final output dimension of the graph embedding.
            num_layers: Number of HeteroConv message passing layers.
        """
        super(HeteroGraphEncoder, self).__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        
        # Linear projection layer for each specific node type
        # Maps varying or uniform input dimensions to the hidden dimension
        self.node_proj = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_dim) 
            for node_type in self.node_types
        })
        
        # Heterogeneous Graph Convolutional layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # SAGEConv acts as the base convolution for each edge type
                edge_type: SAGEConv(hidden_dim, hidden_dim)
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)
            
        # Final projection layer for the graph-level embedding
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        # 1. Project input features for all present node types
        h_dict = {
            node_type: self.node_proj[node_type](x) 
            for node_type, x in x_dict.items()
        }
        
        # 2. Message Passing with residual connections
        for conv in self.convs:
            h_dict_next = conv(h_dict, edge_index_dict)
            h_dict = {
                key: F.relu(h_dict_next[key]) + h_dict.get(key, 0) 
                for key in h_dict_next.keys()
            }
            
        # 3. Robust Heterogeneous Global Pooling Readout
        # Dynamically determine the batch size to handle missing node types in specific graphs
        batch_size = 0
        for b_tensor in batch_dict.values():
            if b_tensor.numel() > 0:
                batch_size = max(batch_size, int(b_tensor.max().item()) + 1)
                
        pooled_features = []
        for node_type, h in h_dict.items():
            if node_type in batch_dict:
                # size=batch_size ensures the output tensor is strictly [batch_size, hidden_dim]
                # even if some graphs in the batch lack this specific node type.
                pooled = global_mean_pool(h, batch_dict[node_type], size=batch_size)
                pooled_features.append(pooled)
        
        # Aggregate pooled representations across all node types (Mean aggregation)
        if len(pooled_features) > 0:
            # Stack shape: [num_node_types, batch_size, hidden_dim] -> [batch_size, hidden_dim]
            h_graph = torch.stack(pooled_features, dim=0).mean(dim=0)
        else:
            # Fallback for completely empty graphs (edge case safety)
            h_graph = torch.zeros((batch_size, self.out_proj.in_features), device=list(h_dict.values())[0].device)
                
        # 4. Final projection
        return self.out_proj(h_graph)