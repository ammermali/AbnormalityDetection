import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class TripleEmbedding(nn.Module):
    def __init__(self, output_vocab_size, tree_vocab_size, context_vocab_size, embed_dim=64):
        super().__init__()
        self.out_emb = nn.Embedding(output_vocab_size, embed_dim)
        self.tree_emb = nn.Embedding(tree_vocab_size, embed_dim)
        self.ctx_emb = nn.Embedding(context_vocab_size, embed_dim)

    def forward(self, out_ids, tree_ids, ctx_ids):
        return (self.out_emb(out_ids) +
                self.tree_emb(tree_ids) +
                self.ctx_emb(ctx_ids))


def create_attention_mask(key_length: int, query_length: int, dtype: torch.dtype) -> torch.Tensor:
    i = torch.arange(query_length)[:, None]
    j = torch.arange(key_length)
    mask = i >= j - key_length + query_length
    mask = torch.logical_not(mask)
    return mask.to(dtype)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=key_dim, vdim=key_dim,
                                          batch_first=True)
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.ffn_1 = nn.Linear(in_features=embed_dim, out_features=ff_dim)
        self.ffn_2 = nn.Linear(in_features=ff_dim, out_features=embed_dim)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        device = inputs.device
        mask = create_attention_mask(seq_len, seq_len, torch.bool).to(device)

        attention_output, _ = self.attn(query=inputs, key=inputs, value=inputs, attn_mask=mask)
        attention_output = self.dropout_1(attention_output)
        out1 = self.layer_norm_1(inputs + attention_output)

        ffn_1 = F.relu(self.ffn_1(out1))
        ffn_output = self.dropout_2(self.ffn_2(ffn_1))
        return self.layer_norm_2(out1 + ffn_output)


class GPTModel(nn.Module):
    def __init__(self, vocab_size, tree_vocab_size, ctx_vocab_size, embed_dim, feed_forward_dim, num_heads, key_dim):
        super().__init__()
        self.embedding_layer = TripleEmbedding(vocab_size, tree_vocab_size, ctx_vocab_size, embed_dim)
        self.transformer = TransformerBlock(num_heads=num_heads, key_dim=key_dim, embed_dim=embed_dim,
                                            ff_dim=feed_forward_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, output_ids, tree_ids, context_ids):
        embedding = self.embedding_layer(output_ids, tree_ids, context_ids)
        transformer_output = self.transformer(embedding)
        return self.output_layer(transformer_output)