import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from constants import PAD


class MLP(nn.Module):
    def __init__(self,
                 word_dim: int,
                 hidden_size: int,
                 vocab_size: int):
        super(MLP, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim, padding_idx=PAD)
        self.linear1 = nn.Linear(word_dim, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self,
                x: torch.Tensor,    # (b, len)
                mask: torch.Tensor  # (b, len)
                ) -> torch.Tensor:  # (b, 2)
        x = self.embed(x)                      # (b, len, dim)
        # (b, dim) / (b, 1) -> (b, dim)
        x_phrase = torch.sum(x, dim=1) / torch.sum(mask, dim=1).unsqueeze(dim=1).float()
        h = self.tanh(self.linear1(x_phrase))  # (b, hid)
        y = self.linear2(h)                    # (b, 2)
        return y

    def set_init_embedding(self, init_embedding: np.array):
        self.embed.weight = nn.Parameter(torch.Tensor(init_embedding))


class BiLSTM(nn.Module):
    def __init__(self,
                 word_dim: int,
                 hidden_size: int,
                 vocab_size: int,
                 pretrained_embeddings: torch.Tensor or None = None):
        super(BiLSTM, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim,
                                  padding_idx=PAD, _weight=pretrained_embeddings)
        self.bi_lstm = RNNWrapper(rnn=nn.LSTM(input_size=word_dim, hidden_size=hidden_size,
                                              batch_first=True, bidirectional=True))
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self,
                x: torch.Tensor,      # (b, len)
                mask: torch.Tensor,   # (b, len)
                ) -> torch.Tensor:    # (b, 2)
        x = self.embed(x)                 # (b, len, dim)
        lstm_out = self.bi_lstm(x, mask)  # (b, len, hid * 2)
        out = lstm_out.sum(dim=1)         # (b, hid * 2)
        h = self.tanh(self.linear1(out))  # (b, hid)
        y = self.linear2(h)               # (b, 2)
        return y

    def set_init_embedding(self, init_embedding: np.array):
        self.embed.weight = nn.Parameter(torch.Tensor(init_embedding))


class BiLSTMAttn(nn.Module):
    def __init__(self,
                 word_dim: int,
                 hidden_size: int,
                 vocab_size: int,
                 pretrained_embeddings: torch.Tensor or None = None):
        super(BiLSTMAttn, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim,
                                  padding_idx=PAD, _weight=pretrained_embeddings)
        self.bi_lstm = RNNWrapper(rnn=nn.LSTM(input_size=word_dim, hidden_size=hidden_size,
                                              batch_first=True, bidirectional=True))
        self.l_attn = nn.Linear(hidden_size * 2, 1)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self,
                x: torch.Tensor,      # (b, len)
                mask: torch.Tensor,   # (b, len)
                ) -> torch.Tensor:    # (b, 2)
        x = self.embed(x)                           # (b, len, dim)
        lstm_out = self.bi_lstm(x, mask)            # (b, len, hid * 2)
        attn = self.l_attn(lstm_out)                # (b, len, 1)
        attn_mask = mask.unsqueeze(-1).type(attn.dtype)
        attn.masked_fill_(attn_mask[:, :lstm_out.size(1), :].ne(1), -1e6)
        attn_softmax = F.softmax(attn, dim=1)       # (b, len, 1)
        out = (lstm_out * attn_softmax).sum(dim=1)  # (b, hid * 2)
        h = self.tanh(self.linear1(out))            # (b, hid)
        y = self.linear2(h)                         # (b, 2)
        return y

    def set_init_embedding(self, init_embedding: np.array):
        self.embed.weight = nn.Parameter(torch.Tensor(init_embedding))


class RNNWrapper(nn.Module):
    def __init__(self,
                 rnn: nn.Module):
        super(RNNWrapper, self).__init__()
        self.rnn = rnn

    def forward(self,
                x: torch.Tensor,    # (b, len, dim)
                mask: torch.Tensor  # (b, len)
                ) -> torch.Tensor:
        lengths = mask.sum(dim=1)   # (b)
        sorted_lengths, sorted_indices = lengths.sort(0, descending=True)
        sorted_input = x[sorted_indices]
        _, unsorted_indices = sorted_indices.sort(0)

        # masking
        packed = pack_padded_sequence(sorted_input, lengths=sorted_lengths, batch_first=True)
        output, _ = self.rnn(packed)  # (sum(lengths), hid*2)
        unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)
        unsorted_input = unpacked[unsorted_indices]
        return unsorted_input         # (b, len, d_hid * 2)


class CNN(nn.Module):
    def __init__(self,
                 word_dim: int,
                 word_lim: int,
                 vocab_size: int,
                 pretrained_embeddings: torch.Tensor or None = None,
                 num_filters: int = 128):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim,
                                  padding_idx=PAD, _weight=pretrained_embeddings)
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
                x: torch.Tensor,    # (b, len)
                mask: torch.Tensor  # (b, len)
                ) -> torch.Tensor:  # (b, 2)
        x = self.embed(x)                   # (b, len, dim)
        x = x.unsqueeze(1)                  # (b, 1, len, dim)
        cnn1 = self.cnn1(x)                 # (b, num_filters, len - 3 + 1, dim - dim + 1)
        bn1 = F.relu(self.bn1(cnn1))        # = (b, num_filters, len - 2, 1)
        pool1 = self.pool1(bn1).squeeze(2)  # (b, num_filters, 1)
        cnn2 = self.cnn2(x)                 # (b, num_filters, len - 4 + 1, dim - dim + 1)
        bn2 = F.relu(self.bn2(cnn2))        # = (b, num_filters, len - 3, 1)
        pool2 = self.pool2(bn2).squeeze(2)  # (b, num_filters, 1)
        cnn3 = self.cnn3(x)                 # (b, num_filters, len - 5 + 1, dim - dim + 1)
        bn3 = F.relu(self.bn3(cnn3))        # = (b, num_filters, len - 4, 1)
        pool3 = self.pool3(bn3).squeeze(2)  # (b, num_filters, 1)
        pooled = torch.cat((pool1, pool2, pool3), dim=1).squeeze(2)  # (b, num_filters * 3)
        y = self.fc(F.dropout(pooled))                               # (b, 2), dropout_rate = 0.5
        return y

    def set_init_embedding(self, init_embedding: np.array):
        self.embed.weight = nn.Parameter(torch.Tensor(init_embedding))
