import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import PAD


class Embedder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD)

    def forward(self,
                x: torch.Tensor,    # (b, len, 1) ... word ids
                ) -> torch.Tensor:  # (b, len, d_emb)
        return self.embed(x)


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


class CNN(nn.Module):
    def __init__(self,
                 word_dim: int,
                 word_lim: int,
                 num_filters: int = 128):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(3, word_dim))
        self.bn1 = nn.BatchNorm2d(num_filters, 1)
        self.pool1 = nn.MaxPool2d(word_lim - 2)
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(4, word_dim))
        self.bn2 = nn.BatchNorm2d(num_filters, 1)
        self.pool2 = nn.MaxPool2d(word_lim - 3)
        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(5, word_dim))
        self.bn3 = nn.BatchNorm2d(num_filters, 1)
        self.pool3 = nn.MaxPool2d(word_lim - 4)
        self.fc = nn.Linear(num_filters * 3, 2)

    def forward(self,
                x: torch.Tensor,    # (b, len, word_dim)
                mask: torch.Tensor  # (b, len)
                ) -> torch.Tensor:  # (b, 2)
        x = x.unsqueeze(1)                  # (b, 1, len, word_dim)
        cnn1 = self.cnn1(x)                 # (b, num_filters, len - 3 + 1, word_dim - word_dim + 1)
        bn1 = F.relu(self.bn1(cnn1))        # = (b, num_filters, len - 2, 1)
        pool1 = self.pool1(bn1).squeeze(2)  # (b, num_filters, 1)
        cnn2 = self.cnn2(x)                 # (b, num_filters, len - 4 + 1, word_dim - word_dim + 1)
        bn2 = F.relu(self.bn2(cnn2))        # = (b, num_filters, len - 3, 1)
        pool2 = self.pool2(bn2).squeeze(2)  # (b, num_filters, 1)
        cnn3 = self.cnn3(x)                 # (b, num_filters, len - 5 + 1, word_dim - word_dim + 1)
        bn3 = F.relu(self.bn3(cnn3))        # = (b, num_filters, len - 4, 1)
        pool3 = self.pool3(bn3).squeeze(2)  # (b, num_filters, 1)
        pooled = torch.cat((pool1, pool2, pool3), dim=1).squeeze(2)  # (b, num_filters * 3)
        y = self.fc(F.dropout(pooled))                               # (b, 2), dropout_rate = 0.5
        return y
