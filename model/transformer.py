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

class MHA(nn.module):
    pass
class MultiHeadAttention(MHA):
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


class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim:int=512, attention_layer:nn.Module=MHA, num_heads:int=8):
        super().__init__()
        self.mha = MHA
        self.ff_dim = embedding_dim*2
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim,self.ff_dim), 
            nn.GELU(),
            nn.Linear(self.ff_dim,embedding_dim),
            nn.Dropout(0.2)
        )
        self.norm1 = nn.LayerNorm(embedding_dim) #pre normalization
        self.norm2 = nn.LayerNorm(embedding_dim) #post normalization

    def forward(self,X):
        X = self.mha.forward(X)
        X = self.norm1.forward(X)
        X = self.feed_forward.forward(X)
        X = self.norm2.forward(X)
        return X


class PositionalEncoding(nn.Module):
    pass
