import torch
from torch_geometric.data import Data
import ast

def ast_to_pyg_data(code_str, target_label):
    """
    Convert python code string to PyG Data object via AST.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    node_list = []
    edge_index = [[], []] # [source_nodes, target_nodes]
    edge_attr = [] # Edge text (field name)
    
    # Map node object id to index
    node_to_idx = {}

    def visit(node, parent_idx=None, relation=None):
        nonlocal node_list, edge_index, edge_attr
        
        # Current node index
        curr_idx = len(node_list)
        node_to_idx[id(node)] = curr_idx
        
        # Node features
        node_type = type(node).__name__
        
        node_list.append({
            'type': node_type
        })
        
        # Add edge if parent exists
        if parent_idx is not None:
            edge_index[0].append(parent_idx)
            edge_index[1].append(curr_idx)
            edge_attr.append(relation if relation else "child")
            
        # Traverse children
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        visit(item, curr_idx, field)
            elif isinstance(value, ast.AST):
                visit(value, curr_idx, field)

    visit(tree)

    if not node_list:
        return None
        
    # Convert to PyG Data
    # Note: We are storing strings in lists, not tensors, as PyG tensors must be numeric.
    # Users can encode them later if needed.
    
    data = Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=len(node_list),
        
        # Custom attributes
        # node_types remapped to node_texts as requested
        node_texts=[n['type'] for n in node_list],
        edge_texts=edge_attr,
        
        # Overall code and label
        code=[code_str],
        y=target_label
    )
    
    return data