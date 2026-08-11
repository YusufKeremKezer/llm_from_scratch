from torch import nn

# Standard embeddings used by BERT and early GPT models.
class Embeddings(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.truncate_embeddings(self.embedding.weight) # Truncate outliers to prevent spikes.

    def forward(self,x):
        return self.embedding(x)

    def truncate_embeddings(self,embeddings,std=0.02):
        return nn.init.trunc_normal_(embeddings,std=std)
