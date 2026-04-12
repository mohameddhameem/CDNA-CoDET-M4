"""CodeBERT tokenization helpers.

Notes
-----
Provides cached tokenization and embedding routines for code text.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_codebert(model_name: str = "microsoft/codebert-base", device: Optional[torch.device] = None):
    device = _get_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def encode_texts(
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
    model: Optional[AutoModel],
    device: Optional[torch.device] = None,
    batch_size: int = 128,
    max_length: int = 256,
    pool: bool = True,
) -> torch.Tensor:
    if model is None and pool:
        raise ValueError("pool=True requires a model; set pool=False when model is None")
    device = _get_device(device)
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)
    if not texts:
        return torch.empty(0)

    embs: List[torch.Tensor] = []
    iterator = range(0, len(texts), batch_size)

    for i in iterator:
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if model is None:
                # No forward pass; return token IDs directly
                embs.append(inputs["input_ids"].cpu())
            else:
                outputs = model(**inputs)
                if pool:
                    pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                    embs.append(pooled.cpu())
                else:
                    embs.append(outputs.last_hidden_state.cpu())
    return torch.cat(embs, dim=0)


# -----------------------------------------------------------------------------
# Homo graphs: expect attributes node_texts / edge_texts on each Data
# -----------------------------------------------------------------------------

def encode_homo_graphs(
    dataset: List,
    out_dir: str = "CPG",
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 128,
    max_length: int = 256,
    device: Optional[torch.device] = None,
):
    out_dir_path = Path(out_dir) / "embeddings" / "homo"
    node_path = out_dir_path / "node_embeddings.pt"
    edge_path = out_dir_path / "edge_embeddings.pt"

    # Reuse cached
    if node_path.exists() and edge_path.exists():
        node_list = torch.load(node_path)
        edge_list = torch.load(edge_path)
        if len(node_list) == len(dataset):
            for d, n_emb, e_emb in zip(dataset, node_list, edge_list):
                d.x = n_emb
                d.edge_attr = e_emb
            return dataset
        else:
            raise ValueError(f"cached embeddings have different length: {len(node_list)} != {len(dataset)}")

    tokenizer, model, device = load_codebert(model_name, device)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    node_list: List[Optional[torch.Tensor]] = []
    edge_list: List[Optional[torch.Tensor]] = []

    for d in tqdm(dataset, desc="homo-graphs"):
        n_emb = None
        e_emb = None
        if getattr(d, "node_texts", None):
            n_emb = encode_texts(d.node_texts, tokenizer, model, device, batch_size, max_length, pool=True)
            d.x = n_emb
        if getattr(d, "edge_texts", None):
            e_emb = encode_texts(d.edge_texts, tokenizer, model, device, batch_size, max_length, pool=True)
            d.edge_attr = e_emb
        node_list.append(n_emb)
        edge_list.append(e_emb)

    torch.save(node_list, node_path)
    torch.save(edge_list, edge_path)
    return dataset


# -----------------------------------------------------------------------------
# Hetero graphs: expect data[node_type].texts per graph
# -----------------------------------------------------------------------------

def encode_hetero_graphs(
    dataset: List,
    out_dir: str = "CPG",
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 128,
    max_length: int = 256,
    device: Optional[torch.device] = None,
):
    out_dir_path = Path(out_dir) / "embeddings" / "hetero"
    hetero_path = out_dir_path / "hetero_embeddings.pt"
    
    if hetero_path.exists():
        emb_list = torch.load(hetero_path)
        if len(emb_list) == len(dataset):
            for data, emb_dict in zip(dataset, emb_list):
                for n_type, emb in emb_dict.items():
                    data[n_type].x = emb
            return dataset
        else:
            raise ValueError(f"cached embeddings have different length: {len(emb_list)} != {len(dataset)}")


    tokenizer, model, device = load_codebert(model_name, device)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    emb_list: List[Dict[str, torch.Tensor]] = []

    for data in tqdm(dataset, desc="hetero-graphs"):
        per_graph: Dict[str, torch.Tensor] = {}
        for n_type in getattr(data, "node_types", []):
            texts = getattr(data[n_type], "texts", None)
            if texts:
                emb = encode_texts(texts, tokenizer, model, device, batch_size, max_length, pool=True)
                data[n_type].x = emb
                per_graph[n_type] = emb
        emb_list.append(per_graph)

    torch.save(emb_list, hetero_path)
    return dataset


# -----------------------------------------------------------------------------
# Plain code/text list encoding (e.g., raw code snippets)
# -----------------------------------------------------------------------------

def encode_code_list(
    code_texts: Iterable[str],
    out_dir: str = "CPG",
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 128,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[Path, torch.Tensor]:
    """Tokenize code list to input_ids and cache to embeddings/code_embeddings.pt."""
    device = _get_device(device)
    out_dir_path = Path(out_dir) / "embeddings"
    code_path = out_dir_path / "code_embeddings.pt"

    if code_path.exists():
        print(f"Loading cached code embeddings from {code_path}")
        cached = torch.load(code_path)
        return cached

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    texts = list(code_texts) if not isinstance(code_texts, str) else [code_texts]
    token_ids: List[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), total=(len(texts) + batch_size - 1) // batch_size, desc="code-list"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids
        token_ids.append(inputs)

    token_tensor = torch.cat(token_ids, dim=0).to(device) if token_ids else torch.empty(0, device=device)
    print(f"Saving code embeddings to {code_path}")
    torch.save(token_tensor.cpu(), code_path)
    return token_tensor
