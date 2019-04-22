from typing import Tuple, List

import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader

from nn.utils import W2V_MODEL_FILE

model_w2v = KeyedVectors.load_word2vec_format(W2V_MODEL_FILE, binary=True)
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
