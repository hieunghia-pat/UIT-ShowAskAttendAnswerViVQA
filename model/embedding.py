import torch
from torch import nn

class TextEmbedding(nn.Module):
    def __init__(self, vocab, embedding_dim, d_model, dropout=0.5):
        super(TextEmbedding, self).__init__()

        self.embedding = nn.Embedding(len(vocab.stoi), embedding_dim, padding_idx=vocab.stoi["<pad>"])
        self.proj = nn.Linear(embedding_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.pe = PositionalEncoding(d_model, max_len=500)

    def forward(self, x):
        x = self.proj(self.embedding(x))

        return self.pe(self.dropout(x))

class VisualEmbedding(nn.Module):
    def __init__(self, visual_dim, d_model, dropout=0.5):
        super(VisualEmbedding, self).__init__()

        self.proj_1 = nn.Linear(visual_dim, visual_dim // (2**2))
        self.dropout_1 = nn.Dropout(dropout)

        self.proj_2 = nn.Linear(visual_dim // (2**2), visual_dim // (2**4))
        self.dropout_2 = nn.Dropout(dropout)

        self.proj_3 = nn.Linear(visual_dim // (2**4), d_model)
        self.dropout_3 = nn.Dropout(dropout)

        self.pe = PositionalEncoding(d_model, max_len=500)

    def forward(self, v):
        n, c, h, w = v.size()
        v = v.view(n, c*h, w).permute(0, 2, 1)

        v = self.dropout_1(self.proj_1(v))
        v = self.dropout_2(self.proj_2(v))
        v = self.dropout_3(self.proj_3(v))

        return self.pe(v)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.register_parameter('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] 
        return x