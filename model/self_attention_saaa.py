import torch
from torch import nn

import numpy as np

from model.latent_encoder import LatentEncoderLayer, LatentEncoder
from model.classifier import Classifier
from model.embedding import TextEmbedding, VisualEmbedding

class SAAA(nn.Module):
    def __init__(self, vocab, v_shape, d_model, embedding_dim, dff, nheads, nlayers, dropout):
        super(SAAA, self).__init__()

        self.padding_idx = vocab.stoi["<pad>"]

        self.visual_embedding = VisualEmbedding(v_shape[-3]*v_shape[-2], d_model, dropout)
        self.text_embedding = TextEmbedding(vocab, embedding_dim, d_model, dropout)
        
        self.visual_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nheads, dff, dropout, batch_first=True),
            nlayers
        )
        self.linguistic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nheads, dff, dropout, batch_first=True),
            nlayers
        )
        self.latent_encoder = LatentEncoder(
            LatentEncoderLayer(d_model, nheads, dff, dropout, batch_first=True), 
            nlayers
        )
        
        self.proj = nn.Linear(vocab.max_question_length, 1)
        self.generator = nn.Linear(d_model, len(vocab.output_cats))
        self.dropout = nn.Dropout(dropout)

    def key_padding_mask(self, x, padding_idx):
        "Mask out subsequent positions."
        return x == padding_idx

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (size, size)
        mask = np.triu(np.ones(attn_shape), k=1)
        return torch.from_numpy(mask).bool()

    def forward(self, v, q):
        device = v.device
        v_embedded = self.visual_embedding(v)
        q_embedded = self.text_embedding(q)

        attn_mask = self.subsequent_mask(q_embedded.size(1)).to(device)
        key_padding_mask = self.key_padding_mask(q, self.padding_idx).to(device)

        v_encoded = self.visual_encoder(v_embedded)
        q_encoded = self.linguistic_encoder(q_embedded, attn_mask, key_padding_mask)
        
        out = self.latent_encoder(v_encoded, q_encoded) # (n, s, e)
        out = self.proj(out.permute(0, -1, -2)).squeeze() # (n, e)

        return self.dropout(self.generator(out))