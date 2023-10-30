import torch.nn as nn
import torch

torch.manual_seed(0)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_length, d_model):
        super().__init__()
        self.positional_encoding = nn.Parameter(
            torch.empty(seq_length, d_model)
        )
        nn.init.xavier_uniform_(self.positional_encoding)

    def forward(self, x):
        return x + self.positional_encoding
