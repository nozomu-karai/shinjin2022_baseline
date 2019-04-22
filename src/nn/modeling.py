import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, word_dim: int, hidden_dim: int):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(word_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self,
                x: torch.Tensor,  # (b, len, dim)
                mask: torch.Tensor):  # (b, len)
        x_phrase = torch.sum(x, dim=1) / torch.sum(mask, dim=1).unsqueeze(dim=1).float()  # (b, dim)
        h = self.tanh(self.linear1(x_phrase))  # (b, hid)
        y = self.linear2(h)  # (b, 2)
        return y
