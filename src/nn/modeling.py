import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, word_dim: int, hidden_size: int):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(word_dim, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self,
                x: torch.Tensor,    # (b, len, dim)
                mask: torch.Tensor  # (b, len)
                ) -> torch.Tensor:  # (b, 2)
        # (b, dim) / (b, 1) -> (b, dim)
        x_phrase = torch.sum(x, dim=1) / torch.sum(mask, dim=1).unsqueeze(dim=1).float()
        h = self.tanh(self.linear1(x_phrase))  # (b, hid)
        y = self.linear2(h)  # (b, 2)
        return y


class BiLSTM(nn.Module):
    def __init__(self, word_dim: int, hidden_size: int):
        super(BiLSTM, self).__init__()
        self.bi_lstm = nn.LSTM(input_size=word_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self,
                x: torch.Tensor,      # (b, len, dim)
                mask: torch.Tensor,   # (b, len)
                ) -> torch.Tensor:    # (b, 2)
        lstm_out, _ = self.bi_lstm(x)     # (b, len, hid * 2)
        out = lstm_out.sum(dim=1)         # (b, hid * 2)
        h = self.tanh(self.linear1(out))  # (b, hid)
        y = self.linear2(h)               # (b, 2)
        return y


class BiLSTMAttn(nn.Module):
    def __init__(self, word_dim: int, hidden_size: int):
        super(BiLSTMAttn, self).__init__()
        self.bi_lstm = nn.LSTM(input_size=word_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.l_attn = nn.Linear(hidden_size * 2, 1)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self,
                x: torch.Tensor,      # (b, len, dim)
                mask: torch.Tensor,   # (b, len)
                ) -> torch.Tensor:    # (b, 2)
        lstm_out, _ = self.bi_lstm(x)               # (b, len, hid * 2)
        attn = self.l_attn(lstm_out)                # (b, len, 1)
        attn_softmax = F.softmax(attn, dim=1)       # (b, len, 1)
        out = (lstm_out * attn_softmax).sum(dim=1)  # (b, hid * 2)
        h = self.tanh(self.linear1(out))            # (b, hid)
        y = self.linear2(h)                         # (b, 2)
        return y
