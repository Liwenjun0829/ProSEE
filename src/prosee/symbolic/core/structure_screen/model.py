import torch
from torch import nn


def normalize_hidden_dims(hidden_dim):
    """Return validated MLP hidden widths, accepting legacy scalar configs."""
    if isinstance(hidden_dim, int):
        hidden_dims = [hidden_dim]
    elif isinstance(hidden_dim, (list, tuple)):
        hidden_dims = list(hidden_dim)
    else:
        raise TypeError("hidden_dim must be a positive integer or a non-empty list of positive integers")
    if not hidden_dims or any(isinstance(width, bool) or not isinstance(width, int) or width <= 0
                              for width in hidden_dims):
        raise ValueError("hidden_dim must be a positive integer or a non-empty list of positive integers")
    return hidden_dims


def regression_head(input_dim, hidden_dims):
    layers = []
    for output_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout(0.1)))
        input_dim = output_dim
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)


class StructureRegressor(nn.Module):
    """Replaceable model contract: forward(token_ids) -> fitted R²."""

    def __init__(self, vocab_size: int, embedding_dim=64, hidden_dim=96):
        super().__init__()
        self.hidden_dims = normalize_hidden_dims(hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        gru_hidden_dim = self.hidden_dims[0]
        self.gru = nn.GRU(embedding_dim, gru_hidden_dim, batch_first=True, bidirectional=True)
        self.head = regression_head(gru_hidden_dim * 2, self.hidden_dims)

    def forward(self, token_ids):
        mask = token_ids.ne(0)
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(token_ids), lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(torch.cat((hidden[-2], hidden[-1]), dim=1)).squeeze(1)
