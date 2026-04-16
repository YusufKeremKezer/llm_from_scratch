from torch import nn
import torch.nn.functional as F
import torch
class ClassicAttentionLayer(nn.Module): # Can see future to cheat
    def __init__(self,embedding_dim):
        super().__init__()
        self.Q = nn.Linear(embedding_dim,embedding_dim)
        self.K = nn.Linear(embedding_dim,embedding_dim)
        self.V = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,X):
        Q = self.Q(X)
        K = self.K(X)
        V = self.V(X)
        scores = Q @ K.transpose(-2,-1) # How related each query with keys.
        attention_weights = F.softmax(scores,dim=-1) # How important affair between them

        return attention_weights @ V # How important that token


class MultiHeadAttention(nn.Module):
    def __init__(self,attention_layer,num_heads,embedding_dim):
        super().__init__()
        self.attention_layers = nn.ModuleList([attention_layer(embedding_dim) for _ in range(num_heads)])

    def forward(self,X):
        head_outputs = [layer(X) for layer in self.attention_layers]
        return torch.cat(head_outputs,dim=-1) # Concatenate all heads


class CasualAttentionLayer(nn.Module):
    def __init__(self,q_size,k_size,v_size,embedding_dim):
        super().__init__()
        self.Q = nn.Linear(embedding_dim,embedding_dim)
        self.K = nn.Linear(embedding_dim,embedding_dim)
        self.V = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,X):
        # X.shape = (batch_size,sequence_len,embedding_dim)
        # Batch size means how many sentences we are processing in parallel by predicting next word.

        Q = self.Q(X) # Q.shape =   (batch_size,sequence_len,embedding_dim) * (embedding_dim x embedding_dim).T
        K = self.K(X)  # output shape = (batch_size,sequence_len,embedding_dim) 
        V = self.V(X) 
        scores = Q @ K.transpose(-2,-1) # How related each query with keys.
        # k.Transpose shape = (batch_size,embedding_dim,sequence_len)

        #scores.shape = (batch_size,sequence_len,sequence_len)
        mask = torch.triu(torch.full(scores.shape, float('-inf')), diagonal=1) #Masking future tokens with -inf to prevent cheating.
        scores +=mask
        attention_weights = F.softmax(scores,dim=-1) # How important affair between them


        return attention_weights @ V # How important that token

class PositionalEncoding(nn.Module):
    pass