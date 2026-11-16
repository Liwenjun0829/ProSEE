import torch
from torch import nn

from .model import normalize_hidden_dims, regression_head


class BoundedStructureRegressor(nn.Module):
    """R² regression with the mathematically valid range (-infinity, 1]."""
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=96):
        super().__init__()
        self.hidden_dims = normalize_hidden_dims(hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        gru_hidden_dim = self.hidden_dims[0]
        self.gru = nn.GRU(embedding_dim, gru_hidden_dim, batch_first=True, bidirectional=True)
        self.head = regression_head(gru_hidden_dim * 2, self.hidden_dims)

    def forward(self, token_ids):
        lengths = token_ids.ne(0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(token_ids), lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        raw = self.head(torch.cat((hidden[-2], hidden[-1]), 1)).squeeze(1)
        return 1.0 - torch.nn.functional.softplus(raw)
