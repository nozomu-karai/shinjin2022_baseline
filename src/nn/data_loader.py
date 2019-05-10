from typing import Dict, List, Optional, Tuple

import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
from constants import PAD


class PNDataset(Dataset):
    def __init__(self,
                 path: str,
                 model_w2v: KeyedVectors,
                 word2id: Dict[str, int],
                 wlim: Optional[int]):
        self.model_w2v = model_w2v
        self.word2id = word2id
        self.wlim = wlim
        self.sources, self.targets = self._load(path)
        self.max_phrase_len: int = max(len(phrase) for phrase in self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        phrase_len = len(self.sources[idx])
        pad: List[np.ndarray] = [PAD] * (self.max_phrase_len - phrase_len)
        source = np.array(self.sources[idx] + pad)  # (len, dim)
        mask = np.array([1] * phrase_len + [0] * (self.max_phrase_len - phrase_len))  # (len)
        target = np.array(self.targets[idx])  # ()
        return source, mask, target

    def _load(self, path: str) -> Tuple[List[List[int]], List[int]]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                tag, body = line.strip().split('\t')
                assert tag in ['1', '-1']
                targets.append(int(tag == '1'))
                ids: List[int] = []
                for mrph in body.split():
                    if mrph in self.word2id:
                        ids.append(self.word2id[mrph])
                    else:
                        ids.append(self.word2id['<UNK>'])
                if self.wlim is not None and len(ids) > self.wlim:
                    ids = ids[-self.wlim:]  # limit word length
                sources.append(ids)
        assert len(sources) == len(targets)
        return sources, targets


class PNDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 model_w2v: KeyedVectors,
                 word2id: Dict[str, int],
                 wlim: Optional[int],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int):
        self.dataset = PNDataset(path, model_w2v, word2id, wlim)
        self.n_samples = len(self.dataset)
        super(PNDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers)
