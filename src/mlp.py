import os
from argparse import ArgumentParser
from typing import Tuple, List

import numpy as np
from gensim.models import KeyedVectors

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


model_w2v = KeyedVectors.load_word2vec_format(
    '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/w2v.midasi.128.100K.bin', binary=True)
PAD = 0


class PNDataset(Dataset):
    def __init__(self, path: str):
        self.sources, self.targets = self._load(path)
        self.max_phrase_len: int = max(len(phrase) for phrase in self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx):
        phrase_len = len(self.sources[idx])
        pad: List[np.ndarray] = [np.full(model_w2v.vector_size, PAD)] * (self.max_phrase_len - phrase_len)
        source = np.array(self.sources[idx] + pad)  # (len, dim)
        mask = np.array([True] * phrase_len + [False] * (self.max_phrase_len - phrase_len))  # (len)
        target = np.array(self.targets[idx])  # ()
        return source, mask, target

    @staticmethod
    def _load(path: str) -> Tuple[List[List[np.ndarray]], List[int]]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                tag, body = line.strip().split('\t')
                assert int(tag) in [1, -1]
                targets.append(int(tag) == 1)
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
    def __init__(self):
        super(MLP, self).__init__()

    def forward(self, x: torch.Tensor):
        pass


def main():
    # Training settings
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: None)')
    parser.add_argument('--batch-size', type=int, default=256,
                            help='number of batch size for training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setup data_loader instances
    train_data_loader = PNDataLoader(
        '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/train/samples.txt',
        args.batch_size, shuffle=True, num_workers=2)
    valid_data_loader = PNDataLoader(
        '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/valid/samples.txt',
        args.batch_size, shuffle=True, num_workers=2)

    # build model architecture
    model = MLP()
    model.train()
    model.to(device)

    # train
    for epoch in range(5):
        for batch_idx, (source, target) in enumerate(train_data_loader):
            source = source.to(device)
            target = target.to(device)

            output = model(source)

            loss = loss_fn(output, target)

    # valid
    for batch_idx, (source, target) in enumerate(valid_data_loader):
        source = source.to(device)
        target = target.to(device)

        output = model(source)
        metrics = metric_fn(output, target)


if __name__ == '__main__':
    main()

