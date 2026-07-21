from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import math

# q.shape = (batch_size,sequence_len,embedding_dim)
# instead of X inputs renamed as q,k,v because now it supports cross-attention
# Batch size means the size of sequence length processing in parallel
# context len means how many words LLM can see
# Q_proj.shape = Q_proj @ X.transpose() (batch_size,sequence_len,embedding_dim) @ (embedding_dim x embedding_dim).T
    
# output Q and K and Vshape = (batch_size,sequence_len,embedding_dim) 
# K.Transpose shape = (batch_size,embedding_dim,sequence_len)

#scores.shape = (batch_size,sequence_len,sequence_len)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads:int=8, embedding_dim:int=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim/num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim,embedding_dim)
    
    def forward(self, q, k, v, mask = None):
        batch ,context_len, _ = q.shape
        
        Q = self.q_proj(q).view(batch,  context_len, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(k).view(batch,  context_len, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(v).view(batch,  context_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        casual_mask = torch.triu(torch.ones(context_len,context_len))

        if mask is not None:
            scores = scores.masked_fill(casual_mask , mask = float("-inf"))

        logits = torch.softmax(scores, dim = -1)
        attention_output = torch.matmul(logits,V) 
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch,context_len,-1)        
        
        output = self.out_proj(attention_output) # to communicate between heads

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim: int=512, expansion_factor: int=4):
        super().__init__()
        hidden_dim = embedding_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(embedding_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)

        )
    def forward(self,x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int=512, num_heads: int=8):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, embedding_dim)
        self.ffn = FeedForwardNetwork(embedding_dim) 
        self.pre_norm = nn.LayerNorm(embedding_dim) #pre normalization
        self.post_norm = nn.LayerNorm(embedding_dim) #post normalization

    def forward(self,X):
        
        X_norm = self.pre_norm(X)
        mha_out = self.mha(X_norm,X_norm,X_norm,mask = 1)
        X = X + mha_out

        X_norm2 = self.post_norm(X)
        ffn_out = self.ffn(X_norm2)

        X = X + ffn_out
        return X


class PositionalEncoding(nn.Module):
    pass
