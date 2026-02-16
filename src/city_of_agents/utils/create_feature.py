from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_text_graph(dataset, lm_type="tiny", batch_size=32, device=None):
    """
    Encode node_texts to data.x and edge_texts to data.edge_attr using a Transformer model.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model choices
    text_ids = {
        "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct", 
        "tiny": "sentence-transformers/all-MiniLM-L6-v2",
        "e5": "intfloat/e5-base-v2",
    }
    
    # Select model ID, handle custom paths if provided in reference mapping, but defaulting to Hub IDs for demo
    model_id = text_ids.get(lm_type, text_ids["tiny"])
    print(f"Loading model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
    except OSError:
        print(f"Model {model_id} not found locally or on Hub. Please check connection or model name.")
        # Fallback logic or exit
        return dataset
        
    model.eval()

    def get_embeddings(texts):
        if not texts:
            return None
            
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Tokenize
            encoded_input = tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            ).to(device)

            # Compute embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
                sample_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            all_embs.append(sample_embeddings.cpu())
            
        if all_embs:
            return torch.cat(all_embs, dim=0)
        return None

    print(f"Encoding features for {len(dataset)} graphs...")
    
    for i, data in enumerate(tqdm(dataset)):
        # Encode Node Texts -> x
        if hasattr(data, 'node_texts') and data.node_texts:
            data.x = get_embeddings(data.node_texts)
        
        # Encode Edge Texts -> edge_attr
        if hasattr(data, 'edge_texts') and data.edge_texts:
            data.edge_attr = get_embeddings(data.edge_texts)
            
    return dataset
