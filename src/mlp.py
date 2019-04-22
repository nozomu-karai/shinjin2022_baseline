import os
from argparse import ArgumentParser
from typing import Tuple, List

import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


TRAIN_FILE = '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/train/samples.txt'
VALID_FILE = '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/valid/samples.txt'

model_w2v = KeyedVectors.load_word2vec_format(
    '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/w2v.midasi.128.100K.bin', binary=True)
PAD = 0


class PNDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.sources, self.targets = self._load(path)
        self.max_phrase_len: int = max(len(phrase) for phrase in self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        phrase_len = len(self.sources[idx])
        pad: List[np.ndarray] = \
            [np.full(model_w2v.vector_size, PAD, dtype=np.float32)] * (self.max_phrase_len - phrase_len)
        source = np.array(self.sources[idx] + pad)  # (len, dim)
        mask = np.array([1] * phrase_len + [0] * (self.max_phrase_len - phrase_len))  # (len)
        target = np.array(self.targets[idx])  # ()
        return source, mask, target

    @staticmethod
    def _load(path: str) -> Tuple[List[List[np.ndarray]], List[int]]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                tag, body = line.strip().split('\t')
                assert tag in ['1', '-1']
                targets.append(int(tag == '1'))
                vectors: List[np.ndarray] = []
                for mrph in body.split():
                    if mrph in model_w2v:
                        vectors.append(model_w2v[mrph])
                    else:
                        vectors.append(model_w2v['<UNK>'])
                sources.append(vectors)
        assert len(sources) == len(targets)
        return sources, targets


class PNDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int
                 ):
        self.dataset = PNDataset(path)
        self.n_samples = len(self.dataset)
        super(PNDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers)


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


def loss_fn(output: torch.Tensor,  # (b, 2)
            target: torch.Tensor   # (b)
            ) -> torch.Tensor:     # ()
    softmax = F.softmax(output, dim=1)  # (b, 2)
    loss = F.binary_cross_entropy(softmax[:, 1], target.float(), reduction='sum')
    return loss


def metric_fn(output: torch.Tensor,  # (b, 2)
              target: torch.Tensor   # (b)
              ) -> int:
    prediction = torch.argmax(output, dim=1)
    return (prediction == target).sum().item()


def main():
    # Training settings
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: None)')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--save-dir', type=str, default='result/',
                        help='directory where trained model is saved')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device is not None else 'cpu')

    # setup data_loader instances
    train_data_loader = PNDataLoader(TRAIN_FILE, args.batch_size, shuffle=True, num_workers=2)
    valid_data_loader = PNDataLoader(VALID_FILE, args.batch_size, shuffle=False, num_workers=2)

    # build model architecture
    model = MLP(word_dim=128, hidden_dim=100)
    model.to(device)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(args.epochs):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        total_correct = 0
        for batch_idx, (source, mask, target) in tqdm(enumerate(train_data_loader)):
            source = source.to(device)  # (b, len, dim)
            mask = mask.to(device)      # (b, len)
            target = target.to(device)  # (b)
            optimizer.zero_grad()

            output = model(source, mask)  # (b, 2)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += metric_fn(output, target)
        print(f'train_loss={total_loss / train_data_loader.n_samples:.3f}', end=' ')
        print(f'train_accuracy={total_correct / train_data_loader.n_samples:.3f}')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for batch_idx, (source, mask, target) in tqdm(enumerate(valid_data_loader)):
                source = source.to(device)  # (b, len, dim)
                mask = mask.to(device)      # (b, len)
                target = target.to(device)  # (b)

                output = model(source, mask)  # (b, 2)

                total_loss += loss_fn(output, target)
                total_correct += metric_fn(output, target)
        valid_acc = total_correct / valid_data_loader.n_samples
        print(f'valid_loss={total_loss / valid_data_loader.n_samples:.3f}', end=' ')
        print(f'valid_accuracy={valid_acc:.3f}\n')
        if valid_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))


if __name__ == '__main__':
    main()
