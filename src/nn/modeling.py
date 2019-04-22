import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, word_dim: int, hidden_size: int):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(word_dim, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self,
                x: torch.Tensor,  # (b, len, dim)
                mask: torch.Tensor):  # (b, len)
        x_phrase = torch.sum(x, dim=1) / torch.sum(mask, dim=1).unsqueeze(dim=1).float()  # (b, dim)
        h = self.tanh(self.linear1(x_phrase))  # (b, hid)
        y = self.linear2(h)  # (b, 2)
        return y


class BiLSTM(nn.Module):
    def __init__(self, word_dim: int, hidden_size: int):
        super(BiLSTM, self).__init__()
        self.bi_lstm = nn.LSTM(input_size=word_dim, hidden_size=hidden_size,
                               batch_first=True, dropout=0.1, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self,
                x: torch.Tensor,  # (b, len, dim)
                mask: torch.Tensor):  # (b, len)
        h, _ = self.bi_lstm(x)  # (b, len, hid * 2)
        h = h.sum(dim=1)  # (b, hid * 2)
        h = self.tanh(self.linear1(h))  # (b, hid)
        y = self.linear2(h)  # (b, 2)
        return y
