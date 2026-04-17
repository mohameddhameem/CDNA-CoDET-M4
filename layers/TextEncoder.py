"""Text encoder layers.

Notes
-----
Wraps pretrained code language models for embedding extraction.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using the attention mask."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class TextEncoder(nn.Module):
    """Frozen CodeBERT encoder with internal tokenization."""
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        embed_dim: int = 768,
        max_length: int = 512,
        freeze: bool = True,
    ):
        """Load a pretrained text encoder and optional projection layer."""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        hidden = int(getattr(self.model.config, "hidden_size", embed_dim))
        self.proj = nn.Identity() if hidden == embed_dim else nn.Linear(hidden, embed_dim, bias=False)
        self.max_length = max_length

    def forward(self, text_input):
        """Encode raw text or tokenized inputs into pooled embeddings."""
        # Accept list/tuple of strings or already tokenized dict/tensors
        if isinstance(text_input, (list, tuple)) and text_input and isinstance(text_input[0], str):
            encoded = self.tokenizer(
                list(text_input),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        elif isinstance(text_input, dict):
            encoded = text_input
        else:
            # Assume input_ids tensor, build minimal dict
            encoded = {"input_ids": text_input}

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _mean_pool(outputs.last_hidden_state, attention_mask)
        return self.proj(pooled)

    def __del__(self):
        """Warn when the encoder object is garbage-collected."""
        print("Warning: TextEncoder instance was not explicitly released.")


class TextEncoderSimple(nn.Module):
    """Encode token IDs with a lightweight Transformer encoder."""

    def __init__(self, vocab_size=30522, embed_dim=256, num_heads=4, num_layers=2, max_len=512):
        """Build embedding, positional encoding, and Transformer layers."""
        super(TextEncoderSimple, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        """Transform token sequences and return mean pooled features."""
        # x shape: [batch_size, seq_len]
        x_emb = self.embedding(x)
        seq_len = x.shape[1]
        
        # Apply positional encoding
        x_emb = x_emb + self.pos_encoder[:, :seq_len, :]
        x_out = self.transformer(x_emb)
        
        # Global mean pooling over the sequence dimension
        return x_out.mean(dim=1)


class TextEncoderCLS(nn.Module):
    """Encode text and return CLS-token pooled representations."""

    def __init__(
        self,
        model_name="microsoft/unixcoder-base",
        embed_dim=768,
        max_length=1024,
        freeze=True,
    ):
        """Load a pretrained CLS-based encoder with optional projection."""
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        hidden = self.model.config.hidden_size
        self.proj = nn.Identity() if hidden == embed_dim else nn.Linear(hidden, embed_dim, bias=False)
        self.max_length = max_length

    def forward(self, text_input):
        """Tokenize text inputs and return projected CLS embeddings."""
        encoded = self.tokenizer(
            list(text_input),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = self.model(**encoded)

        pooled = outputs.last_hidden_state[:, 0]  # CLS pooling
        return self.proj(pooled)