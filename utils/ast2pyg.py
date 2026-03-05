import torch
import ast
from torch_geometric.data import Data

def ast_to_pyg_data(code_str, target_label=None):
    """
    Convert python code string to PyG Data object via AST.
    Combines Node Type (e.g., 'Name') with Node Content (e.g., "id='x'")
    into a single node label.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None

    node_list = []      # Stores the combined label strings
    edge_index = [[], []] # [source_nodes, target_nodes]
    edge_attr = []      # Edge text (field name)
    
    # Map node object id to index to handle potential DAGs (though AST is a tree)
    node_to_idx = {}

    # Helper: Extract content based on node type
    def get_node_content(node):
        """Extracts the specific content (name, id, arg, value) from a node."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return f"name='{node.name}'"
        elif isinstance(node, ast.Name):
            return f"id='{node.id}'"
        elif isinstance(node, ast.arg):
            return f"arg='{node.arg}'"
        elif isinstance(node, ast.Attribute):
            return f"attr='{node.attr}'"
        elif isinstance(node, ast.keyword):
            return f"arg='{node.arg}'"
        elif isinstance(node, ast.alias):
            return f"name='{node.name}'"
        elif isinstance(node, ast.Constant):
            val = str(node.value).replace('\n', '\\n')
            if len(val) > 20: val = val[:20] + "..."
            return f"value='{val}'"
        return ""

    def visit(node, parent_idx=None, relation=None):
        nonlocal node_list, edge_index, edge_attr
        
        # Current node index
        curr_idx = len(node_list)
        node_to_idx[id(node)] = curr_idx
        
        # 1. Get Node Type (e.g., "FunctionDef")
        node_type = type(node).__name__
        
        # 2. Get Node Content (e.g., "name='my_func'")
        content = get_node_content(node)
        
        # 3. Combine them into the final label
        # Format: "[FunctionDef] name='my_func'" or just "[Return]"
        if content:
            combined_label = f"[{node_type}] {content}"
        else:
            combined_label = f"[{node_type}]"
            
        node_list.append({'label': combined_label})
        
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
    data = Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=len(node_list),
        
        # output combined labels
        node_texts=[n['label'] for n in node_list], 
        edge_texts=edge_attr,
        
        code=code_str,
        y=target_label
    )
    
    return data